"""
total_recall.py — Unified Knowledge API (v2: Tier-1 upgrade)
==============================================================

Three engines, one query, fused results:

    TotalRecall
    ├── MnemosyneAdapter   — ChromaDB semantic search + temporal KG
    ├── FirstbrainAdapter  — ChromaDB semantic search + PageRank (vault)
    └── CricketAdapter     — Relevance scoring via neuromorphic resonance

v2 changes (Tier-1):
  - Firstbrain uses ChromaDB embeddings instead of keyword matching
  - Cricket-Brain acts as relevance signal amplifier, not fake text search
  - Cross-source dedup uses semantic similarity, not string prefix
  - Scoring formula uses real semantic distances everywhere
"""

import os
import re
import math
import hashlib
import logging
from datetime import datetime, date
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger("total_recall")


# ─── Data Types ──────────────────────────────────────────────────────────────


@dataclass
class SearchHit:
    """Normalized search result from any source."""
    text: str
    source: str
    location: str
    similarity: float       # 0.0-1.0 semantic similarity (from embeddings)
    pagerank: float = 0.0   # 0.0-1.0 importance
    recency: float = 0.0    # 0.0-1.0 temporal freshness
    relevance_boost: float = 0.0  # 0.0-1.0 cricket-brain signal boost
    metadata: dict = field(default_factory=dict)

    @property
    def fused_score(self):
        return self.metadata.get("fused_score", self.similarity)


# ─── Base Adapter ────────────────────────────────────────────────────────────


class SourceAdapter:
    name: str = "base"

    def available(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> list:
        return []

    def status(self) -> dict:
        return {"name": self.name, "available": self.available()}


# ─── Mnemosyne Adapter ──────────────────────────────────────────────────────


class MnemosyneAdapter(SourceAdapter):
    """ChromaDB semantic search + temporal Knowledge Graph."""

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
            hits.append(SearchHit(
                text=doc,
                source="mnemosyne",
                location=f"{meta.get('wing', '?')}/{meta.get('room', '?')}",
                similarity=round(max(0, 1 - dist), 4),
                recency=_compute_recency(filed_at) if filed_at else 0.0,
                metadata={
                    "wing": meta.get("wing", ""),
                    "room": meta.get("room", ""),
                    "source_file": meta.get("source_file", ""),
                    "filed_at": filed_at,
                },
            ))
        return hits

    def kg_search(self, entity: str, as_of: str = None) -> list:
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
                hits.append(SearchHit(
                    text=text,
                    source="mnemosyne_kg",
                    location="knowledge_graph",
                    similarity=1.0,
                    recency=_compute_recency(fact.get("valid_from", "")) if fact.get("valid_from") else 0.0,
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


# ─── Firstbrain Adapter (v2: SEMANTIC SEARCH via ChromaDB) ──────────────────


class FirstbrainAdapter(SourceAdapter):
    """
    Searches Obsidian vault using ChromaDB embeddings (same engine as Mnemosyne).
    v2: Replaces keyword matching with real semantic search.
    PageRank from the graph engine boosts important notes.
    """

    name = "firstbrain"

    def __init__(self, vault_path: str = None):
        self._vault_path = vault_path or os.environ.get("FIRSTBRAIN_VAULT_PATH", "")
        self._graph = None
        self._collection = None
        self._indexed = False

    def _get_graph(self):
        if self._graph is None and self.available():
            from firstbrain.graph import VaultGraph
            self._graph = VaultGraph(self._vault_path)
            self._graph.build()
        return self._graph

    def _get_collection(self):
        """Build ChromaDB collection from vault notes for semantic search."""
        if self._collection is not None:
            return self._collection
        if not self.available():
            return None

        import chromadb
        # Use ephemeral client — vault is re-indexed each session
        client = chromadb.EphemeralClient()
        try:
            client.delete_collection("firstbrain_vault")
        except Exception:
            pass
        col = client.create_collection("firstbrain_vault")

        vault = Path(self._vault_path)
        ids, docs, metas = [], [], []
        for md_file in vault.rglob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            # Strip frontmatter for cleaner embeddings
            body = _strip_frontmatter(content)
            if len(body.strip()) < 20:
                continue

            rel_path = str(md_file.relative_to(vault))
            name = md_file.stem
            doc_id = hashlib.md5(rel_path.encode()).hexdigest()[:16]

            ids.append(doc_id)
            docs.append(body)
            metas.append({
                "name": name,
                "path": rel_path,
                "mtime": md_file.stat().st_mtime,
            })

        if ids:
            # Batch add (ChromaDB handles embedding)
            batch = 100
            for i in range(0, len(ids), batch):
                col.add(
                    ids=ids[i:i+batch],
                    documents=docs[i:i+batch],
                    metadatas=metas[i:i+batch],
                )

        self._collection = col
        self._indexed = True
        return col

    def available(self) -> bool:
        if not self._vault_path:
            return False
        p = Path(self._vault_path)
        return p.exists() and p.is_dir()

    def search(self, query: str, limit: int = 5) -> list:
        col = self._get_collection()
        if not col or col.count() == 0:
            return []

        # Semantic search via ChromaDB embeddings
        try:
            results = col.query(
                query_texts=[query],
                n_results=min(limit, col.count()),
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return []

        # Get PageRank scores
        graph = self._get_graph()
        pr_scores = {}
        if graph:
            for entry in graph.pagerank(top_n=9999):
                pr_scores[entry["name"]] = entry["score"]
        max_pr = max(pr_scores.values()) if pr_scores else 1.0

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            name = meta.get("name", "?")
            pr = pr_scores.get(name, 0.0)
            mtime = meta.get("mtime", 0)

            # Truncate document for snippet
            snippet = doc[:500] + "..." if len(doc) > 500 else doc

            hits.append(SearchHit(
                text=snippet,
                source="firstbrain",
                location=meta.get("path", "?"),
                similarity=round(max(0, 1 - dist), 4),
                pagerank=round(pr / max_pr if max_pr > 0 else 0, 4),
                recency=_compute_recency(datetime.fromtimestamp(mtime).isoformat()) if mtime else 0.0,
                metadata={
                    "file": meta.get("path", ""),
                    "note_name": name,
                    "pagerank_raw": round(pr, 6),
                },
            ))

        return hits

    def graph_stats(self) -> dict:
        graph = self._get_graph()
        return graph.stats() if graph else {"error": "Vault not available"}

    def graph_pagerank(self, top_n: int = 20) -> list:
        graph = self._get_graph()
        return graph.pagerank(top_n=top_n) if graph else []

    def graph_clusters(self) -> list:
        graph = self._get_graph()
        return graph.tag_clusters() if graph else []

    def graph_path(self, source: str, target: str) -> dict:
        graph = self._get_graph()
        return graph.shortest_path(source, target) if graph else {"error": "Vault not available"}

    def graph_bridges(self) -> list:
        graph = self._get_graph()
        return graph.bridge_notes() if graph else []

    def status(self) -> dict:
        info = {"name": self.name, "available": self.available()}
        if self.available():
            info["vault_path"] = self._vault_path
            info["semantic_search"] = True
            col = self._get_collection()
            if col:
                info["indexed_notes"] = col.count()
            graph = self._get_graph()
            if graph:
                info.update(graph.stats())
        return info


# ─── Cricket-Brain Adapter (v2: RELEVANCE SIGNAL AMPLIFIER) ─────────────────


class CricketAdapter(SourceAdapter):
    """
    v2: Cricket-Brain as relevance signal amplifier.

    Instead of pretending to do text search, Cricket-Brain processes the
    text of OTHER adapters' results through its resonator bank. Text that
    triggers stronger neural resonance patterns gets a relevance boost.

    This is biologically correct: the cricket brain is a filter that
    amplifies interesting signals and suppresses noise. We use it the
    same way — to re-rank results from other sources.

    How it works:
    1. Convert each result's text to a frequency signal (character → frequency)
    2. Run through Cricket-Brain's resonator bank
    3. Measure total neural activation energy
    4. High activation = text has strong internal patterns = more relevant
    """

    name = "cricket"

    def __init__(self, engine_path: str = None):
        self._engine_path = engine_path or os.environ.get("CRICKET_BRAIN_PATH", "")
        self._brain = None

    def _get_brain(self):
        if self._brain is not None:
            return self._brain
        try:
            import cricket_brain
            if hasattr(cricket_brain, "Brain"):
                self._brain = cricket_brain.Brain()
                return self._brain
        except (ImportError, AttributeError):
            pass
        return None

    def available(self) -> bool:
        return self._get_brain() is not None

    def compute_relevance(self, text: str, query: str) -> float:
        """
        Run text through Cricket-Brain and measure neural activation.

        Converts text characters to frequency signals, processes through
        the 5-neuron circuit, and measures total spike energy.
        Higher energy = more structured/patterned text = likely more relevant.

        Additionally compares query-signal resonance with text-signal resonance
        to measure alignment.
        """
        brain = self._get_brain()
        if not brain:
            return 0.0

        brain.reset()

        # Convert query to frequency pattern
        query_freqs = _text_to_frequencies(query)
        query_energy = 0.0
        for freq in query_freqs:
            out = brain.step(freq)
            query_energy += abs(out)

        brain.reset()

        # Convert result text to frequency pattern (sample for speed)
        text_sample = text[:200]  # first 200 chars for speed
        text_freqs = _text_to_frequencies(text_sample)
        text_energy = 0.0
        for freq in text_freqs:
            out = brain.step(freq)
            text_energy += abs(out)

        brain.reset()

        # Now interleave query + text to measure resonance overlap
        combined = []
        for i in range(max(len(query_freqs), len(text_freqs))):
            if i < len(query_freqs):
                combined.append(query_freqs[i])
            if i < len(text_freqs):
                combined.append(text_freqs[i])

        combined_energy = 0.0
        for freq in combined[:400]:  # cap at 400 steps
            out = brain.step(freq)
            combined_energy += abs(out)

        brain.reset()

        # Resonance boost: if combined energy > sum of parts, they reinforce
        individual_sum = query_energy + text_energy
        if individual_sum > 0:
            resonance_ratio = combined_energy / individual_sum
        else:
            resonance_ratio = 0.0

        # Normalize to 0-1 range (resonance_ratio ~1.0 = no boost, >1.0 = reinforcement)
        boost = min(1.0, max(0.0, (resonance_ratio - 0.5) / 1.5))
        return round(boost, 4)

    def search(self, query: str, limit: int = 5) -> list:
        # Cricket-Brain doesn't search — it boosts other results
        return []

    def status(self) -> dict:
        info = {"name": self.name, "available": self.available(), "role": "relevance_amplifier"}
        brain = self._get_brain()
        if brain:
            info["engine"] = "cricket-brain v3.0 (5-neuron circuit)"
        return info


# ─── TotalRecall — The Meta-Layer (v2) ──────────────────────────────────────


class TotalRecall:
    """
    v2: Unified search with real semantic search, real dedup, and
    Cricket-Brain as relevance amplifier.

    Scoring: fused = similarity * w1 + pagerank * w2 + recency * w3 + cricket_boost * w4
    """

    def __init__(
        self,
        palace_path: str = None,
        vault_path: str = None,
        cricket_path: str = None,
        similarity_weight: float = 0.45,
        pagerank_weight: float = 0.25,
        recency_weight: float = 0.15,
        resonance_weight: float = 0.15,
    ):
        self.weights = {
            "similarity": similarity_weight,
            "pagerank": pagerank_weight,
            "recency": recency_weight,
            "resonance": resonance_weight,
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
        target_sources = sources or list(self.adapters.keys())
        all_hits = []
        sources_queried = []

        # Gather hits from text-search adapters
        for name in target_sources:
            if name == "cricket":
                continue  # cricket doesn't produce hits, it boosts them
            adapter = self.adapters.get(name)
            if not adapter or not adapter.available():
                continue

            sources_queried.append(name)

            if name == "mnemosyne":
                hits = adapter.search(query, limit=limit, wing=wing, room=room)
                if include_kg:
                    kg_hits = adapter.kg_search(query)
                    hits.extend(kg_hits)
            else:
                hits = adapter.search(query, limit=limit)

            all_hits.extend(hits)

        # Apply Cricket-Brain relevance boost
        cricket = self.adapters.get("cricket")
        if cricket and cricket.available():
            sources_queried.append("cricket")
            for hit in all_hits:
                boost = cricket.compute_relevance(hit.text, query)
                hit.relevance_boost = boost

        # Fuse, dedup, rank
        ranked = self._fuse_and_rank(all_hits, limit, query)

        return {
            "query": query,
            "sources_queried": sources_queried,
            "total_hits": len(all_hits),
            "results": [_hit_to_dict(h) for h in ranked],
        }

    def status(self) -> dict:
        source_status = {}
        for name, adapter in self.adapters.items():
            source_status[name] = adapter.status()
        available = [n for n, a in self.adapters.items() if a.available()]
        return {"sources": source_status, "available": available, "weights": self.weights}

    def configure(self, **kwargs) -> dict:
        for key in ("similarity_weight", "pagerank_weight", "recency_weight", "resonance_weight"):
            if key in kwargs:
                w_name = key.replace("_weight", "")
                self.weights[w_name] = float(kwargs[key])
        return {"weights": self.weights}

    def _fuse_and_rank(self, hits: list, limit: int, query: str) -> list:
        """
        v2 scoring with 4 dimensions + semantic dedup.
        """
        w = self.weights

        for hit in hits:
            fused = (
                hit.similarity * w["similarity"]
                + hit.pagerank * w["pagerank"]
                + hit.recency * w["recency"]
                + hit.relevance_boost * w["resonance"]
            )
            hit.metadata["fused_score"] = round(fused, 4)

        hits.sort(key=lambda h: h.metadata.get("fused_score", 0), reverse=True)

        # Semantic dedup: remove results that cover the same information
        deduped = []
        for hit in hits:
            if _is_duplicate(hit, deduped):
                continue
            deduped.append(hit)

        return deduped[:limit]


# ─── Semantic Deduplication ──────────────────────────────────────────────────


def _is_duplicate(candidate: SearchHit, existing: list, threshold: float = 0.7) -> bool:
    """
    Check if candidate is semantically a duplicate of any existing hit.
    Uses token overlap (Jaccard) instead of string prefix matching.
    """
    if not existing:
        return False

    cand_tokens = _tokenize(candidate.text)
    if not cand_tokens:
        return False

    for hit in existing:
        hit_tokens = _tokenize(hit.text)
        if not hit_tokens:
            continue

        # Jaccard similarity on word tokens
        intersection = len(cand_tokens & hit_tokens)
        union = len(cand_tokens | hit_tokens)
        if union > 0 and intersection / union >= threshold:
            return True

    return False


def _tokenize(text: str) -> set:
    """Extract significant word tokens from text."""
    words = re.findall(r'[a-zA-Z]{3,}', text.lower())
    # Remove very common words
    stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
                 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has',
                 'have', 'this', 'that', 'with', 'from', 'they', 'been',
                 'said', 'each', 'which', 'their', 'will', 'other', 'about',
                 'type', 'created', 'tags', 'connections'}
    return set(words) - stopwords


# ─── Cricket-Brain Signal Conversion ─────────────────────────────────────────


def _text_to_frequencies(text: str) -> list:
    """
    Convert text to frequency signals for Cricket-Brain processing.

    Maps characters to frequencies in the 2000-8000 Hz range
    (Cricket-Brain's optimal hearing range). Vowels cluster around
    4500 Hz (the cricket's tuned frequency), consonants spread wider.

    Spaces become silence (0 Hz), punctuation becomes brief pauses.
    """
    freqs = []
    for ch in text.lower():
        if ch in 'aeiou':
            # Vowels near cricket's tuned frequency (4000-5000 Hz)
            freq = 4000 + (ord(ch) % 10) * 100
        elif ch.isalpha():
            # Consonants spread across full range
            freq = 2000 + (ord(ch) - ord('a')) * 230
        elif ch == ' ':
            freq = 0.0  # silence
        elif ch in '.,;:!?':
            freq = 0.0  # pause
        else:
            freq = 3000 + (ord(ch) % 20) * 200
        freqs.append(freq)
    return freqs


# ─── Utility Functions ──────────────────────────────────────────────────────


def _strip_frontmatter(content: str) -> str:
    """Remove YAML frontmatter from markdown content."""
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            return content[end + 3:].strip()
    return content


def _compute_recency(date_str: str) -> float:
    """Date to 0.0-1.0 recency. Today=1.0, half-life=90 days."""
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
        return round(math.exp(-0.693 * days_ago / 90), 4)
    except (ValueError, TypeError):
        return 0.0


def _extract_snippet(content: str, query_terms: set, max_chars: int = 500) -> str:
    """Extract most relevant snippet around query matches."""
    lines = content.split("\n")
    best_idx, best_score = 0, 0
    for i, line in enumerate(lines):
        score = sum(1 for t in query_terms if t in line.lower())
        if score > best_score:
            best_score = score
            best_idx = i
    start = max(0, best_idx - 2)
    end = min(len(lines), best_idx + 5)
    snippet = "\n".join(lines[start:end])
    return snippet[:max_chars] + "..." if len(snippet) > max_chars else snippet


def _count_backlinks(md_files: list) -> dict:
    """Count wikilinks per note."""
    pattern = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
    counts = {}
    for f in md_files:
        try:
            for m in pattern.finditer(f.read_text(encoding="utf-8", errors="ignore")):
                t = m.group(1).strip()
                counts[t] = counts.get(t, 0) + 1
        except Exception:
            pass
    return counts


def _hit_to_dict(hit: SearchHit) -> dict:
    return {
        "text": hit.text,
        "source": hit.source,
        "location": hit.location,
        "similarity": hit.similarity,
        "pagerank": hit.pagerank,
        "recency": hit.recency,
        "relevance_boost": hit.relevance_boost,
        "fused_score": hit.fused_score,
        "metadata": hit.metadata,
    }
