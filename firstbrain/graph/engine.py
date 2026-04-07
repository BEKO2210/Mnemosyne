"""
engine.py — Graph analysis engine for Obsidian vaults.

Builds a directed graph from markdown files and wiki-links, then runs
PageRank, clustering, path finding, bridge detection, and structural
similarity analysis. Zero external dependencies (no networkx, no numpy).

Usage:
    from firstbrain.graph import VaultGraph

    g = VaultGraph("/path/to/obsidian/vault")
    g.build()

    top = g.pagerank(top_n=10)
    clusters = g.tag_clusters()
    path = g.shortest_path("Note A", "Note B")
    bridges = g.bridge_notes()
    similar = g.structural_similarity("Note A", top_n=5)
"""

import re
import os
import math
from pathlib import Path
from collections import defaultdict


# ─── Wiki-link parsing ──────────────────────────────────────────────────────

_WIKILINK = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)
_TAGS = re.compile(r"tags:\s*\[([^\]]*)\]")
_TAG_INLINE = re.compile(r"(?:^|\s)#([a-zA-Z][a-zA-Z0-9_/-]+)", re.MULTILINE)


def _parse_tags(content: str) -> set:
    """Extract tags from frontmatter and inline hashtags."""
    tags = set()
    # Frontmatter tags
    fm = _FRONTMATTER.match(content)
    if fm:
        m = _TAGS.search(fm.group(1))
        if m:
            for t in m.group(1).split(","):
                t = t.strip().strip('"').strip("'")
                if t:
                    tags.add(t.lower())
    # Inline tags
    for m in _TAG_INLINE.finditer(content):
        tags.add(m.group(1).lower())
    return tags


def _parse_links(content: str) -> list:
    """Extract wiki-link targets from markdown content."""
    return [m.group(1).strip() for m in _WIKILINK.finditer(content)]


# ─── VaultGraph ──────────────────────────────────────────────────────────────


class VaultGraph:
    """
    Directed graph built from an Obsidian vault's wiki-links.

    Nodes = note names (without .md extension)
    Edges = wiki-links (A links to B → directed edge A→B)
    """

    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path)
        self.nodes = {}       # name → {path, tags, content_preview, mtime}
        self.edges = {}       # name → set of linked note names
        self.backlinks = {}   # name → set of notes that link TO this note
        self._built = False

    def build(self) -> dict:
        """
        Scan the vault and build the graph.
        Returns stats: {nodes, edges, density, components, orphans}.
        """
        self.nodes.clear()
        self.edges.clear()
        self.backlinks.clear()

        md_files = list(self.vault_path.rglob("*.md"))

        for md_file in md_files:
            name = md_file.stem
            try:
                content = md_file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            rel_path = str(md_file.relative_to(self.vault_path))
            tags = _parse_tags(content)
            links = _parse_links(content)
            mtime = md_file.stat().st_mtime

            self.nodes[name] = {
                "path": rel_path,
                "tags": tags,
                "content_preview": content[:200],
                "mtime": mtime,
            }
            self.edges[name] = set(links)

        # Build backlinks index
        for name, targets in self.edges.items():
            for target in targets:
                if target not in self.backlinks:
                    self.backlinks[target] = set()
                self.backlinks[target].add(name)

        self._built = True

        n_nodes = len(self.nodes)
        n_edges = sum(len(v) for v in self.edges.values())
        max_edges = n_nodes * (n_nodes - 1) if n_nodes > 1 else 1
        density = n_edges / max_edges if max_edges > 0 else 0
        orphans = [n for n in self.nodes if not self.edges.get(n) and not self.backlinks.get(n)]

        return {
            "nodes": n_nodes,
            "edges": n_edges,
            "density": round(density, 4),
            "orphans": len(orphans),
            "orphan_names": orphans[:20],
        }

    # ─── PageRank ────────────────────────────────────────────────────────

    def pagerank(self, damping: float = 0.85, iterations: int = 50, top_n: int = 20) -> list:
        """
        Compute PageRank for all notes. Returns sorted list of (name, score).

        Pure Python implementation — no dependencies.
        """
        if not self._built:
            self.build()

        all_nodes = list(self.nodes.keys())
        n = len(all_nodes)
        if n == 0:
            return []

        # Initialize uniform
        scores = {name: 1.0 / n for name in all_nodes}

        for _ in range(iterations):
            new_scores = {}
            for name in all_nodes:
                # Sum of scores from nodes that link to this one
                incoming = self.backlinks.get(name, set())
                rank_sum = 0.0
                for source in incoming:
                    if source in scores:
                        out_degree = len(self.edges.get(source, set()))
                        if out_degree > 0:
                            rank_sum += scores[source] / out_degree

                new_scores[name] = (1 - damping) / n + damping * rank_sum
            scores = new_scores

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [{"name": name, "score": round(score, 6), "tags": list(self.nodes[name]["tags"])}
                for name, score in ranked[:top_n]
                if name in self.nodes]

    # ─── Tag Clustering ──────────────────────────────────────────────────

    def tag_clusters(self, min_cluster_size: int = 3) -> list:
        """
        Group notes by shared tags. Returns clusters sorted by size.
        Uses tag co-occurrence as the clustering signal.
        """
        if not self._built:
            self.build()

        # Build tag → notes index
        tag_index = defaultdict(set)
        for name, info in self.nodes.items():
            for tag in info["tags"]:
                tag_index[tag].add(name)

        # Group notes that share the same tag set
        clusters = []
        seen_notes = set()

        for tag, notes in sorted(tag_index.items(), key=lambda x: len(x[1]), reverse=True):
            cluster_notes = notes - seen_notes
            if len(cluster_notes) >= min_cluster_size:
                clusters.append({
                    "primary_tag": tag,
                    "notes": sorted(cluster_notes),
                    "size": len(cluster_notes),
                    "all_tags": sorted(set().union(*(self.nodes[n]["tags"] for n in cluster_notes if n in self.nodes))),
                })
                seen_notes.update(cluster_notes)

        return clusters

    # ─── Shortest Path ───────────────────────────────────────────────────

    def shortest_path(self, source: str, target: str) -> dict:
        """
        BFS shortest path between two notes via wiki-links.
        Returns path as list of note names, or empty if unreachable.
        """
        if not self._built:
            self.build()

        if source not in self.nodes and source not in self.edges:
            return {"path": [], "error": f"Note not found: {source}"}
        if target not in self.nodes and target not in self.edges:
            return {"path": [], "error": f"Note not found: {target}"}

        # BFS
        visited = {source}
        queue = [(source, [source])]

        while queue:
            current, path = queue.pop(0)
            for neighbor in self.edges.get(current, set()):
                if neighbor == target:
                    full_path = path + [neighbor]
                    return {"path": full_path, "hops": len(full_path) - 1}
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return {"path": [], "hops": -1, "message": f"No path from {source} to {target}"}

    # ─── Bridge Detection ────────────────────────────────────────────────

    def bridge_notes(self) -> list:
        """
        Find articulation points — notes whose removal would disconnect
        parts of the graph. These are critical knowledge hubs.
        """
        if not self._built:
            self.build()

        # Build undirected adjacency for bridge detection
        adj = defaultdict(set)
        for name, targets in self.edges.items():
            for t in targets:
                adj[name].add(t)
                adj[t].add(name)

        # Tarjan's articulation point algorithm
        disc = {}
        low = {}
        parent = {}
        ap = set()
        timer = [0]

        def dfs(u):
            children = 0
            disc[u] = low[u] = timer[0]
            timer[0] += 1

            for v in adj[u]:
                if v not in disc:
                    children += 1
                    parent[v] = u
                    dfs(v)
                    low[u] = min(low[u], low[v])

                    if parent.get(u) is None and children > 1:
                        ap.add(u)
                    if parent.get(u) is not None and low[v] >= disc[u]:
                        ap.add(u)
                elif v != parent.get(u):
                    low[u] = min(low[u], disc[v])

        for node in adj:
            if node not in disc:
                parent[node] = None
                dfs(node)

        bridges = []
        for name in sorted(ap):
            info = self.nodes.get(name, {})
            bridges.append({
                "name": name,
                "connections": len(adj.get(name, set())),
                "tags": sorted(info.get("tags", set())),
            })

        bridges.sort(key=lambda b: b["connections"], reverse=True)
        return bridges

    # ─── Structural Similarity ───────────────────────────────────────────

    def structural_similarity(self, note_name: str, top_n: int = 5) -> list:
        """
        Find notes with similar link neighborhoods (Jaccard similarity).
        Notes that link to the same things or are linked by the same things
        are structurally similar — even without shared tags.
        """
        if not self._built:
            self.build()

        if note_name not in self.nodes:
            return []

        # Get this note's neighborhood
        out_links = self.edges.get(note_name, set())
        in_links = self.backlinks.get(note_name, set())
        neighborhood = out_links | in_links

        if not neighborhood:
            return []

        similarities = []
        for other_name in self.nodes:
            if other_name == note_name:
                continue

            other_out = self.edges.get(other_name, set())
            other_in = self.backlinks.get(other_name, set())
            other_neighborhood = other_out | other_in

            if not other_neighborhood:
                continue

            # Jaccard similarity
            intersection = len(neighborhood & other_neighborhood)
            union = len(neighborhood | other_neighborhood)
            jaccard = intersection / union if union > 0 else 0

            if jaccard > 0:
                similarities.append({
                    "name": other_name,
                    "similarity": round(jaccard, 4),
                    "shared_neighbors": sorted(neighborhood & other_neighborhood)[:10],
                })

        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_n]

    # ─── Multi-hop Discovery ─────────────────────────────────────────────

    def multi_hop(self, note_name: str, max_hops: int = 3) -> list:
        """
        Find notes reachable within N hops but not directly linked.
        These are hidden connections — ideas that are close but not obvious.
        """
        if not self._built:
            self.build()

        direct = self.edges.get(note_name, set()) | self.backlinks.get(note_name, set())
        direct.add(note_name)

        # BFS to max_hops
        visited = {note_name}
        current_layer = {note_name}
        discoveries = []

        for hop in range(1, max_hops + 1):
            next_layer = set()
            for node in current_layer:
                for neighbor in self.edges.get(node, set()) | self.backlinks.get(node, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_layer.add(neighbor)

                        if neighbor not in direct and neighbor in self.nodes:
                            discoveries.append({
                                "name": neighbor,
                                "hops": hop,
                                "tags": sorted(self.nodes[neighbor]["tags"]),
                            })
            current_layer = next_layer

        discoveries.sort(key=lambda x: x["hops"])
        return discoveries

    # ─── Stats ───────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Complete graph statistics."""
        if not self._built:
            self.build()

        n = len(self.nodes)
        e = sum(len(v) for v in self.edges.values())
        degrees = {name: len(self.edges.get(name, set())) + len(self.backlinks.get(name, set()))
                   for name in self.nodes}
        avg_degree = sum(degrees.values()) / n if n > 0 else 0

        return {
            "nodes": n,
            "edges": e,
            "density": round(e / (n * (n - 1)) if n > 1 else 0, 4),
            "avg_degree": round(avg_degree, 2),
            "max_degree": max(degrees.values()) if degrees else 0,
            "hub": max(degrees, key=degrees.get) if degrees else None,
            "orphans": len([name for name in self.nodes
                          if not self.edges.get(name) and not self.backlinks.get(name)]),
        }
