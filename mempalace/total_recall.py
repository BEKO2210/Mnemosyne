"""
total_recall.py — Unified Knowledge API (v3: Query Expansion via Ollama)
=========================================================================

Three engines, one query, fused results:

    TotalRecall
    ├── QueryExpander        — Ollama LLM expands queries with related terms
    ├── MnemosyneAdapter     — ChromaDB semantic search + temporal KG
    ├── FirstbrainAdapter    — ChromaDB semantic search + PageRank (vault)
    └── CricketAdapter       — Relevance scoring via neuromorphic resonance

v3 changes:
  - Ollama-based query expansion before search (Recall boost)
  - Expanded query used for all adapters simultaneously
  - Original query preserved for Cricket-Brain relevance scoring
  - Graceful fallback: if Ollama unavailable, uses original query
"""

import os
import re
import math
import hashlib
import logging
import json
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
    Cricket-Brain as neuromorphic relevance amplifier.

    The 5-neuron delay-line coincidence circuit acts as a relevance filter:
    - Shared words between query and result text → resonant frequency burst (30 steps)
    - Non-shared words → silence
    - The circuit fires when it detects SUSTAINED resonant input
    - More shared content = denser bursts = higher spike rate = more relevant

    This is a neuromorphic version of term overlap scoring:
    the temporal pattern of neural activation directly measures
    how densely query-relevant content appears in the result.

    Minimum burst for spike: 30 steps at 4000 Hz (circuit resonant freq).
    """

    name = "cricket"
    _RESONANT_FREQ = 4000.0
    _BURST_LEN = 30   # steps per shared word (minimum to trigger spike)
    _PAUSE_LEN = 5    # steps per non-shared word

    _STOPWORDS = frozenset({
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'do', 'does', 'did', 'will', 'would', 'can', 'could', 'shall', 'should',
        'may', 'might', 'must', 'we', 'our', 'my', 'your', 'his', 'her', 'its',
        'their', 'i', 'you', 'he', 'she', 'it', 'they', 'me', 'him', 'us', 'them',
        'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 'but', 'not', 'no',
        'so', 'if', 'as', 'by', 'up', 'out', 'off', 'how', 'what', 'when',
        'where', 'who', 'why', 'that', 'this', 'with', 'from', 'has', 'have',
        'had', 'than', 'then', 'also', 'just', 'only', 'very', 'too',
    })

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
        Neuromorphic relevance scoring via word-level resonance.

        1. Extract signal words from query (minus stopwords)
        2. Scan result text word by word
        3. Shared words → 30-step burst at 4000 Hz (triggers spike)
        4. Non-shared words → 5-step silence
        5. Spike rate = relevance score (0.0-1.0)

        Tested: AUTH text with 4 shared words → 32% spike rate
                Irrelevant text with 0 shared → 0% spike rate
        """
        brain = self._get_brain()
        if not brain:
            return 0.0

        q_words = set(query.lower().split()) - self._STOPWORDS
        if not q_words:
            return 0.0

        # Also match word stems (simple suffix strip)
        q_stems = set()
        for w in q_words:
            q_stems.add(w)
            if len(w) > 5:
                q_stems.add(w[:-1])   # "authentication" → "authenticatio"
                q_stems.add(w[:-2])   # "authentication" → "authenticati"
                q_stems.add(w[:-3])   # "authentication" → "authenticat"

        # Build signal from text
        signal = []
        text_words = text.lower().split()
        for w in text_words[:80]:  # cap for speed
            w_clean = w.strip('.,;:!?"\'-()[]{}/')
            matched = w_clean in q_stems or any(
                w_clean.startswith(s) or s.startswith(w_clean)
                for s in q_stems if len(s) > 3 and len(w_clean) > 3
            )
            if matched:
                signal.extend([self._RESONANT_FREQ] * self._BURST_LEN)
            else:
                signal.extend([0.0] * self._PAUSE_LEN)

        if not signal:
            return 0.0

        brain.reset()
        outputs = brain.step_batch(signal)
        spikes = sum(1 for o in outputs if o > 0)
        rate = spikes / len(outputs) if outputs else 0.0

        # Normalize: max observed rate ~0.35 → scale to 0-1
        normalized = min(1.0, rate / 0.35)
        return round(normalized, 4)

    def search(self, query: str, limit: int = 5) -> list:
        return []

    def status(self) -> dict:
        info = {"name": self.name, "available": self.available(), "role": "relevance_amplifier"}
        brain = self._get_brain()
        if brain:
            info["engine"] = "cricket-brain v3.0 (5-neuron delay-line circuit)"
            info["resonant_freq"] = self._RESONANT_FREQ
            info["burst_length"] = self._BURST_LEN
        return info


# ─── Query Expansion via Ollama ──────────────────────────────────────────────


class QueryExpander:
    """
    Uses a local Ollama LLM to expand search queries with related terms.

    "user login" → "user login authentication OAuth2 PKCE session token
                    credential verification access control SSO"

    This dramatically improves ChromaDB semantic search because the
    expanded terms overlap with the actual vocabulary in stored documents.

    Graceful fallback: if Ollama is not running, returns the original query.
    """

    _PROMPT = (
        "Expand this search query with related technical terms, synonyms, "
        "and concepts. Return ONLY a single line of expanded search terms. "
        "No explanation, no numbering, no formatting.\n\n"
        "Query: {query}\n\nExpanded:"
    )

    def __init__(self, model: str = None, ollama_url: str = None):
        self._model = model or os.environ.get("OLLAMA_MODEL", "phi3.5")
        self._url = ollama_url or os.environ.get("OLLAMA_URL", "http://localhost:11434")
        self._available = None

    def available(self) -> bool:
        """Check if Ollama is running and the model is loaded."""
        if self._available is not None:
            return self._available
        try:
            import urllib.request
            req = urllib.request.Request(f"{self._url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read())
                models = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
                self._available = self._model.split(":")[0] in models
        except Exception:
            self._available = False
        return self._available

    def expand(self, query: str) -> str:
        """
        Expand a query using the local LLM.
        Returns: "original query + expanded terms"
        Falls back to original query if Ollama unavailable.
        """
        if not self.available():
            return query

        try:
            import urllib.request
            payload = json.dumps({
                "model": self._model,
                "prompt": self._PROMPT.format(query=query),
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 80},
            }).encode()

            req = urllib.request.Request(
                f"{self._url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
                expanded = data.get("response", "").strip()

            if expanded:
                # Combine original + expanded for best coverage
                return f"{query} {expanded}"
        except Exception as e:
            logger.debug(f"Query expansion failed: {e}")

        return query

    def status(self) -> dict:
        return {
            "available": self.available(),
            "model": self._model,
            "url": self._url,
        }


# ─── TotalRecall — The Meta-Layer (v3) ──────────────────────────────────────


class TotalRecall:
    """
    v3: Unified search with Ollama query expansion, real semantic search,
    real dedup, and Cricket-Brain as relevance amplifier.

    Pipeline: Query → Expand (Ollama) → Search (all adapters) → Boost (Cricket) → Fuse → Dedup
    Scoring: fused = similarity * w1 + pagerank * w2 + recency * w3 + cricket_boost * w4
    """

    def __init__(
        self,
        palace_path: str = None,
        vault_path: str = None,
        cricket_path: str = None,
        ollama_model: str = None,
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

        self.expander = QueryExpander(model=ollama_model)
        self._reranker = None
        # Reranker disabled by default — helps at scale (100+ docs),
        # hurts on small corpora by distorting multi-source fusion.
        # Enable with: tr.enable_reranker()
        # self._init_reranker()
        self.adapters = {
            "mnemosyne": MnemosyneAdapter(palace_path=palace_path),
            "firstbrain": FirstbrainAdapter(vault_path=vault_path),
            "cricket": CricketAdapter(engine_path=cricket_path),
        }

    def enable_reranker(self):
        """Enable cross-encoder re-ranking. Recommended for 100+ documents."""
        self._init_reranker()
        return {"reranker": "enabled" if self._reranker else "failed to load"}

    def _init_reranker(self):
        """Load cross-encoder reranker (lazy, once)."""
        try:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
            logger.info("Reranker loaded: ms-marco-MiniLM-L-12-v2")
        except Exception:
            self._reranker = None

    def _rerank(self, hits: list, query: str) -> list:
        """
        Re-score hits using cross-encoder as BOOST signal (not replacement).

        The cross-encoder score is added as a bonus to the original similarity,
        not as a replacement. This preserves the calibrated ChromaDB distances
        while letting the cross-encoder promote results it finds relevant.
        """
        if not self._reranker or not hits:
            return hits

        pairs = [(query, hit.text[:512]) for hit in hits]
        try:
            scores = self._reranker.predict(pairs)
        except Exception:
            return hits

        # Sigmoid normalization: raw logits → 0-1 probability
        # This preserves absolute quality (unlike min-max normalization)
        for hit, score in zip(hits, scores):
            sigmoid = 1.0 / (1.0 + math.exp(-float(score)))
            # Blend: keep 70% original similarity + 30% reranker score
            original = hit.similarity
            hit.similarity = round(original * 0.7 + sigmoid * 0.3, 4)
            hit.metadata["reranker_raw"] = round(float(score), 4)
            hit.metadata["reranker_sigmoid"] = round(sigmoid, 4)
            hit.metadata["similarity_original"] = round(original, 4)

        return hits

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

        # Step 1: Query Expansion via Ollama
        original_query = query
        expanded_query = self.expander.expand(query)
        used_expansion = expanded_query != query

        # Use expanded query for semantic search adapters
        search_query = expanded_query

        # Gather hits from text-search adapters
        for name in target_sources:
            if name == "cricket":
                continue
            adapter = self.adapters.get(name)
            if not adapter or not adapter.available():
                continue

            sources_queried.append(name)

            if name == "mnemosyne":
                hits = adapter.search(search_query, limit=limit, wing=wing, room=room)
                if include_kg:
                    # KG uses original query (entity names, not expanded)
                    kg_hits = adapter.kg_search(original_query)
                    hits.extend(kg_hits)
            else:
                hits = adapter.search(search_query, limit=limit)

            all_hits.extend(hits)

        # Step 2: Cross-encoder re-ranking (boosts similarity scores)
        if all_hits and self._reranker is not None:
            all_hits = self._rerank(all_hits, original_query)

        # Step 3: Cricket-Brain relevance boost (uses ORIGINAL query for precision)
        cricket = self.adapters.get("cricket")
        if cricket and cricket.available():
            sources_queried.append("cricket")
            for hit in all_hits:
                boost = cricket.compute_relevance(hit.text, original_query)
                hit.relevance_boost = boost

        # Step 4: Fuse, dedup, rank
        ranked = self._fuse_and_rank(all_hits, limit, original_query)

        return {
            "query": original_query,
            "expanded_query": expanded_query if used_expansion else None,
            "reranked": self._reranker is not None,
            "sources_queried": sources_queried,
            "total_hits": len(all_hits),
            "results": [_hit_to_dict(h) for h in ranked],
        }

    def status(self) -> dict:
        source_status = {}
        for name, adapter in self.adapters.items():
            source_status[name] = adapter.status()
        available = [n for n, a in self.adapters.items() if a.available()]
        return {
            "sources": source_status,
            "available": available,
            "weights": self.weights,
            "query_expansion": self.expander.status(),
            "reranker": "ms-marco-MiniLM-L-12-v2" if self._reranker else None,
        }

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
