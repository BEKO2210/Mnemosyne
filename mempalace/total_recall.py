"""
total_recall.py — Unified Knowledge API across all repos
=========================================================

A meta-layer that searches Mnemosyne (MemPalace), Firstbrain (Obsidian),
and Cricket-Brain (neuromorphic signals) through a single query interface.

Architecture:
    TotalRecall
    ├── MnemosyneAdapter   — ChromaDB semantic search + KG (always available)
    ├── FirstbrainAdapter  — Obsidian vault search via file system (optional)
    └── CricketAdapter     — Rust signal pattern matching (optional)

Each adapter returns normalized SearchHit objects. TotalRecall fuses results
using weighted scoring: semantic_similarity * w1 + pagerank * w2 + recency * w3.

If an adapter is unavailable (not installed, path not set), it degrades
gracefully — Mnemosyne is always the fallback.
"""

import os
import re
import math
import logging
from datetime import datetime, date
from pathlib import Path
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("total_recall")


# ─── Data Types ──────────────────────────────────────────────────────────────


@dataclass
class SearchHit:
    """Normalized search result from any source."""
    text: str
    source: str            # "mnemosyne", "firstbrain", "cricket"
    location: str          # wing/room, vault/folder, or signal channel
    similarity: float      # 0.0-1.0 semantic similarity
    pagerank: float = 0.0  # 0.0-1.0 importance (Firstbrain graph rank)
    recency: float = 0.0   # 0.0-1.0 temporal freshness
    metadata: dict = field(default_factory=dict)

    @property
    def fused_score(self):
        """Weighted combination — set by TotalRecall at merge time."""
        return self.metadata.get("fused_score", self.similarity)


# ─── Base Adapter ────────────────────────────────────────────────────────────


class SourceAdapter:
    """Interface for knowledge source adapters."""

    name: str = "base"

    def available(self) -> bool:
        """Return True if this source is configured and reachable."""
        return False

    def search(self, query: str, limit: int = 5) -> list:
        """Return list of SearchHit objects."""
        return []

    def status(self) -> dict:
        """Return status info about this source."""
        return {"name": self.name, "available": self.available()}


# ─── Mnemosyne Adapter (always available) ────────────────────────────────────


class MnemosyneAdapter(SourceAdapter):
    """
    Searches the MemPalace ChromaDB + Knowledge Graph.
    Always available — this is the core system.
    """

    name = "mnemosyne"

    def __init__(self, palace_path: str = None, collection_name: str = None):
        from .config import MempalaceConfig
        cfg = MempalaceConfig()
        self._palace_path = palace_path or cfg.palace_path
        self._collection_name = collection_name or cfg.collection_name

    def available(self) -> bool:
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self._palace_path)
            client.get_collection(self._collection_name)
            return True
        except Exception:
            return False

    def search(self, query: str, limit: int = 5, wing: str = None, room: str = None) -> list:
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self._palace_path)
            col = client.get_collection(self._collection_name)
        except Exception:
            return []

        where = {}
        if wing and room:
            where = {"$and": [{"wing": wing}, {"room": room}]}
        elif wing:
            where = {"wing": wing}
        elif room:
            where = {"room": room}

        kwargs = {
            "query_texts": [query],
            "n_results": limit,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        try:
            results = col.query(**kwargs)
        except Exception:
            return []

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            filed_at = meta.get("filed_at", "")
            recency = _compute_recency(filed_at) if filed_at else 0.0

            hits.append(SearchHit(
                text=doc,
                source="mnemosyne",
                location=f"{meta.get('wing', '?')}/{meta.get('room', '?')}",
                similarity=round(1 - dist, 4),
                recency=recency,
                metadata={
                    "wing": meta.get("wing", ""),
                    "room": meta.get("room", ""),
                    "source_file": meta.get("source_file", ""),
                    "filed_at": filed_at,
                },
            ))
        return hits

    def kg_search(self, entity: str, as_of: str = None) -> list:
        """Search the Knowledge Graph for entity relationships."""
        try:
            from .knowledge_graph import KnowledgeGraph
            kg = KnowledgeGraph()
            facts = kg.query_entity(entity, as_of=as_of, direction="both")
            hits = []
            for fact in facts:
                text = f"{fact['subject']} → {fact['predicate']} → {fact['object']}"
                if fact.get("valid_from"):
                    text += f" (since {fact['valid_from']})"
                if fact.get("valid_to"):
                    text += f" (until {fact['valid_to']})"

                recency = 0.0
                if fact.get("valid_from"):
                    recency = _compute_recency(fact["valid_from"])

                hits.append(SearchHit(
                    text=text,
                    source="mnemosyne_kg",
                    location="knowledge_graph",
                    similarity=1.0,  # exact entity match
                    recency=recency,
                    metadata=fact,
                ))
            return hits
        except Exception:
            return []

    def status(self) -> dict:
        info = {"name": self.name, "available": self.available(), "palace_path": self._palace_path}
        if self.available():
            try:
                import chromadb
                client = chromadb.PersistentClient(path=self._palace_path)
                col = client.get_collection(self._collection_name)
                info["drawers"] = col.count()
            except Exception:
                pass
        return info


# ─── Firstbrain Adapter (optional — Obsidian vault) ─────────────────────────


class FirstbrainAdapter(SourceAdapter):
    """
    Searches an Obsidian vault on the filesystem.
    Reads markdown files, scores by backlink count (poor-man's PageRank).

    Enable by setting FIRSTBRAIN_VAULT_PATH env var or passing vault_path.
    """

    name = "firstbrain"

    def __init__(self, vault_path: str = None):
        self._vault_path = vault_path or os.environ.get("FIRSTBRAIN_VAULT_PATH", "")

    def available(self) -> bool:
        if not self._vault_path:
            return False
        p = Path(self._vault_path)
        return p.exists() and p.is_dir()

    def search(self, query: str, limit: int = 5) -> list:
        if not self.available():
            return []

        vault = Path(self._vault_path)
        md_files = list(vault.rglob("*.md"))
        if not md_files:
            return []

        # Build backlink index for PageRank approximation
        backlink_counts = _count_backlinks(md_files)
        max_backlinks = max(backlink_counts.values()) if backlink_counts else 1

        # Simple keyword search (no embedding needed)
        query_terms = set(query.lower().split())
        scored = []

        for md_file in md_files:
            try:
                content = md_file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            content_lower = content.lower()
            # Term frequency scoring
            term_hits = sum(1 for t in query_terms if t in content_lower)
            if term_hits == 0:
                continue

            similarity = term_hits / len(query_terms) if query_terms else 0
            rel_path = md_file.relative_to(vault)
            fname = md_file.stem

            # PageRank approximation from backlink count
            bl = backlink_counts.get(fname, 0)
            pagerank = bl / max_backlinks if max_backlinks > 0 else 0

            # Recency from file modification time
            mtime = md_file.stat().st_mtime
            recency = _compute_recency(datetime.fromtimestamp(mtime).isoformat())

            # Extract relevant snippet
            snippet = _extract_snippet(content, query_terms, max_chars=500)

            scored.append(SearchHit(
                text=snippet,
                source="firstbrain",
                location=str(rel_path),
                similarity=round(similarity, 4),
                pagerank=round(pagerank, 4),
                recency=recency,
                metadata={
                    "file": str(rel_path),
                    "backlinks": bl,
                    "modified": datetime.fromtimestamp(mtime).isoformat(),
                },
            ))

        # Sort by term match density
        scored.sort(key=lambda h: h.similarity, reverse=True)
        return scored[:limit]

    def status(self) -> dict:
        info = {"name": self.name, "available": self.available()}
        if self.available():
            vault = Path(self._vault_path)
            info["vault_path"] = self._vault_path
            info["notes"] = len(list(vault.rglob("*.md")))
        return info


# ─── Cricket-Brain Adapter (optional — Rust signal engine) ───────────────────


class CricketAdapter(SourceAdapter):
    """
    Connects to cricket-brain for real-time pattern detection.
    Checks if the cricket_brain Python binding is installed.

    Enable by installing cricket-brain Python bindings
    or setting CRICKET_BRAIN_PATH.
    """

    name = "cricket"

    def __init__(self, engine_path: str = None):
        self._engine_path = engine_path or os.environ.get("CRICKET_BRAIN_PATH", "")
        self._engine = None

    def available(self) -> bool:
        # Check for cricket_brain Python bindings
        try:
            import cricket_brain  # noqa: F401
            return True
        except ImportError:
            pass
        # Check for engine binary
        if self._engine_path and Path(self._engine_path).exists():
            return True
        return False

    def search(self, query: str, limit: int = 5) -> list:
        """
        Cricket-brain doesn't do text search — it does signal pattern matching.
        This adapter translates text queries into signal pattern lookups.

        Returns active resonance patterns that match query keywords.
        """
        if not self.available():
            return []

        try:
            import cricket_brain
            # Attempt to use the Python bindings for pattern lookup
            if hasattr(cricket_brain, "query_patterns"):
                patterns = cricket_brain.query_patterns(query, limit=limit)
                return [
                    SearchHit(
                        text=p.get("description", str(p)),
                        source="cricket",
                        location=f"channel/{p.get('channel', '?')}",
                        similarity=p.get("confidence", 0.5),
                        recency=1.0,  # real-time signals are always fresh
                        metadata=p,
                    )
                    for p in patterns
                ]
        except Exception:
            pass

        return []

    def status(self) -> dict:
        info = {"name": self.name, "available": self.available()}
        if self._engine_path:
            info["engine_path"] = self._engine_path
        return info


# ─── TotalRecall — The Meta-Layer ────────────────────────────────────────────


class TotalRecall:
    """
    Unified knowledge query across all sources.

    Usage:
        tr = TotalRecall()
        results = tr.search("Auth-Entscheidungen")
        # → fused results from Mnemosyne + Firstbrain + Cricket-Brain

    Scoring weights (configurable):
        similarity_weight: how much semantic match matters (default 0.5)
        pagerank_weight:   how much link importance matters (default 0.3)
        recency_weight:    how much freshness matters (default 0.2)
    """

    def __init__(
        self,
        palace_path: str = None,
        vault_path: str = None,
        cricket_path: str = None,
        similarity_weight: float = 0.5,
        pagerank_weight: float = 0.3,
        recency_weight: float = 0.2,
    ):
        self.weights = {
            "similarity": similarity_weight,
            "pagerank": pagerank_weight,
            "recency": recency_weight,
        }

        self.adapters = {
            "mnemosyne": MnemosyneAdapter(palace_path=palace_path),
            "firstbrain": FirstbrainAdapter(vault_path=vault_path),
            "cricket": CricketAdapter(engine_path=cricket_path),
        }

    def search(
        self,
        query: str,
        limit: int = 10,
        sources: list = None,
        wing: str = None,
        room: str = None,
        include_kg: bool = True,
    ) -> dict:
        """
        Unified search across all available sources.

        Args:
            query:      What to search for
            limit:      Max total results
            sources:    Which sources to query (default: all available)
            wing/room:  Filter for Mnemosyne only
            include_kg: Also search Knowledge Graph for entity matches

        Returns:
            dict with "results" (fused & ranked), "sources_queried", "query"
        """
        target_sources = sources or list(self.adapters.keys())
        all_hits = []
        sources_queried = []

        for name in target_sources:
            adapter = self.adapters.get(name)
            if not adapter or not adapter.available():
                continue

            sources_queried.append(name)

            if name == "mnemosyne":
                hits = adapter.search(query, limit=limit, wing=wing, room=room)
                if include_kg:
                    # Also search KG for entity matches
                    kg_hits = adapter.kg_search(query)
                    hits.extend(kg_hits)
            else:
                hits = adapter.search(query, limit=limit)

            all_hits.extend(hits)

        # Fuse scores
        ranked = self._fuse_and_rank(all_hits, limit)

        return {
            "query": query,
            "sources_queried": sources_queried,
            "total_hits": len(all_hits),
            "results": [_hit_to_dict(h) for h in ranked],
        }

    def status(self) -> dict:
        """Status of all sources — what's connected, what's not."""
        source_status = {}
        for name, adapter in self.adapters.items():
            source_status[name] = adapter.status()

        available = [n for n, a in self.adapters.items() if a.available()]
        return {
            "sources": source_status,
            "available": available,
            "weights": self.weights,
        }

    def configure(self, **kwargs) -> dict:
        """Update scoring weights at runtime."""
        for key in ("similarity_weight", "pagerank_weight", "recency_weight"):
            if key in kwargs:
                w_name = key.replace("_weight", "")
                self.weights[w_name] = float(kwargs[key])
        return {"weights": self.weights}

    def _fuse_and_rank(self, hits: list, limit: int) -> list:
        """
        Combine scores from different sources into a single ranking.

        Formula: fused = sim * w_sim + pagerank * w_pr + recency * w_rec
        """
        w_sim = self.weights["similarity"]
        w_pr = self.weights["pagerank"]
        w_rec = self.weights["recency"]

        for hit in hits:
            fused = (
                hit.similarity * w_sim
                + hit.pagerank * w_pr
                + hit.recency * w_rec
            )
            hit.metadata["fused_score"] = round(fused, 4)

        hits.sort(key=lambda h: h.metadata.get("fused_score", 0), reverse=True)

        # Deduplicate by text similarity (avoid showing same content from KG + ChromaDB)
        seen_texts = set()
        deduped = []
        for hit in hits:
            # Use first 100 chars as dedup key
            key = hit.text[:100].strip().lower()
            if key not in seen_texts:
                seen_texts.add(key)
                deduped.append(hit)

        return deduped[:limit]


# ─── Utility Functions ──────────────────────────────────────────────────────


def _compute_recency(date_str: str) -> float:
    """
    Convert a date/datetime string to a 0.0-1.0 recency score.
    Today = 1.0, 1 year ago = ~0.0. Uses exponential decay.
    """
    try:
        if "T" in date_str:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(date_str)

        if dt.tzinfo:
            dt = dt.replace(tzinfo=None)

        days_ago = (datetime.now() - dt).days
        if days_ago < 0:
            return 1.0
        # Exponential decay: half-life of 90 days
        return round(math.exp(-0.693 * days_ago / 90), 4)
    except (ValueError, TypeError):
        return 0.0


def _count_backlinks(md_files: list) -> dict:
    """
    Count how many files link to each note (wikilinks).
    Returns dict of {note_name: backlink_count}.
    """
    link_pattern = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
    counts = {}
    for md_file in md_files:
        try:
            content = md_file.read_text(encoding="utf-8", errors="ignore")
            for match in link_pattern.finditer(content):
                target = match.group(1).strip()
                counts[target] = counts.get(target, 0) + 1
        except Exception:
            continue
    return counts


def _extract_snippet(content: str, query_terms: set, max_chars: int = 500) -> str:
    """Extract the most relevant snippet around query term matches."""
    lines = content.split("\n")
    best_line_idx = 0
    best_score = 0

    for i, line in enumerate(lines):
        line_lower = line.lower()
        score = sum(1 for t in query_terms if t in line_lower)
        if score > best_score:
            best_score = score
            best_line_idx = i

    # Take context around best match
    start = max(0, best_line_idx - 2)
    end = min(len(lines), best_line_idx + 5)
    snippet = "\n".join(lines[start:end])

    if len(snippet) > max_chars:
        snippet = snippet[:max_chars] + "..."
    return snippet


def _hit_to_dict(hit: SearchHit) -> dict:
    """Convert SearchHit to a JSON-serializable dict."""
    return {
        "text": hit.text,
        "source": hit.source,
        "location": hit.location,
        "similarity": hit.similarity,
        "pagerank": hit.pagerank,
        "recency": hit.recency,
        "fused_score": hit.fused_score,
        "metadata": hit.metadata,
    }
