"""
Firstbrain Graph Engine — Knowledge graph analysis for Obsidian vaults.

Algorithms:
  - PageRank: importance ranking of notes by link structure
  - Clustering: topic clusters by shared tags (Jaccard similarity)
  - Path finding: shortest path between notes via wiki-links
  - Bridge detection: notes whose removal would split the graph
  - Structural similarity: notes linking to/from the same neighbors
"""

from .engine import VaultGraph

__all__ = ["VaultGraph"]
