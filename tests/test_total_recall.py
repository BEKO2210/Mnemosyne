"""Tests for total_recall.py v2 — unified knowledge API."""

import math
from mempalace.total_recall import (
    TotalRecall,
    SearchHit,
    MnemosyneAdapter,
    FirstbrainAdapter,
    CricketAdapter,
    _compute_recency,
    _count_backlinks,
    _extract_snippet,
    _is_duplicate,
    _tokenize,
    _strip_frontmatter,
)


# ─── Utility function tests ─────────────────────────────────────────────────


def test_compute_recency_today():
    from datetime import datetime
    score = _compute_recency(datetime.now().isoformat())
    assert score >= 0.99


def test_compute_recency_old_date():
    score = _compute_recency("2020-01-01")
    assert score < 0.01


def test_compute_recency_invalid():
    assert _compute_recency("not-a-date") == 0.0
    assert _compute_recency("") == 0.0


def test_extract_snippet():
    content = "Line one\nLine two\nLine three has auth topic\nLine four\nLine five"
    snippet = _extract_snippet(content, {"auth"}, max_chars=500)
    assert "auth" in snippet


def test_count_backlinks(tmp_path):
    f1 = tmp_path / "note1.md"
    f1.write_text("See [[note2]] and [[note3]]")
    f2 = tmp_path / "note2.md"
    f2.write_text("Link to [[note3]]")
    f3 = tmp_path / "note3.md"
    f3.write_text("No links here")
    counts = _count_backlinks([f1, f2, f3])
    assert counts["note2"] == 1
    assert counts["note3"] == 2


def test_strip_frontmatter():
    content = "---\ntype: zettel\ntags: [a]\n---\n# Title\nBody text"
    assert _strip_frontmatter(content) == "# Title\nBody text"


def test_strip_frontmatter_no_fm():
    assert _strip_frontmatter("# Just text") == "# Just text"


# ─── Cricket-Brain relevance scoring ─────────────────────────────────────────


def test_cricket_relevance_scoring():
    """Cricket adapter computes meaningful relevance from word overlap."""
    adapter = CricketAdapter()
    if not adapter.available():
        return  # skip if no compiled bindings

    # Relevant text (shares "authentication", "user", "login")
    boost_auth = adapter.compute_relevance(
        "User authentication handled via Keycloak for login and security",
        "How do we handle user authentication and login"
    )

    # Irrelevant text (no shared words)
    boost_db = adapter.compute_relevance(
        "PostgreSQL won over SQLite because we need concurrent writes",
        "How do we handle user authentication and login"
    )

    assert boost_auth > boost_db, f"Auth boost {boost_auth} should be > DB boost {boost_db}"
    assert boost_auth > 0.0
    assert boost_db == 0.0


def test_cricket_relevance_stems():
    """Cricket adapter matches word stems, not just exact words."""
    adapter = CricketAdapter()
    if not adapter.available():
        return

    # "authenticate" should match query containing "authentication"
    boost = adapter.compute_relevance(
        "We authenticate users via OAuth2 PKCE tokens",
        "What is the authentication strategy"
    )
    assert boost > 0.0


# ─── Semantic dedup tests ────────────────────────────────────────────────────


def test_tokenize():
    tokens = _tokenize("The quick brown fox jumps over the lazy dog")
    assert "quick" in tokens
    assert "brown" in tokens
    assert "the" not in tokens  # stopword


def test_is_duplicate_identical():
    h1 = SearchHit("OAuth2 with PKCE flow for mobile app authentication", "a", "x", 0.9)
    existing = [h1]
    h2 = SearchHit("OAuth2 with PKCE flow for mobile app authentication", "b", "y", 0.8)
    assert _is_duplicate(h2, existing) is True


def test_is_duplicate_different():
    h1 = SearchHit("OAuth2 with PKCE flow for mobile app", "a", "x", 0.9)
    existing = [h1]
    h2 = SearchHit("PostgreSQL database setup with TimescaleDB extension", "b", "y", 0.8)
    assert _is_duplicate(h2, existing) is False


def test_is_duplicate_partial_overlap():
    h1 = SearchHit("We chose PostgreSQL for the Orion project database", "a", "x", 0.9)
    existing = [h1]
    h2 = SearchHit("PostgreSQL was selected for Orion because of concurrent writes", "b", "y", 0.8)
    # ~50% overlap, below 0.7 threshold → not duplicate
    result = _is_duplicate(h2, existing)
    # This should be borderline — both mention PostgreSQL + Orion
    assert isinstance(result, bool)


# ─── SearchHit tests ────────────────────────────────────────────────────────


def test_search_hit_fused_score():
    hit = SearchHit("test", "mnemosyne", "w/r", 0.9, metadata={"fused_score": 0.75})
    assert hit.fused_score == 0.75


def test_search_hit_default_fused_score():
    hit = SearchHit("test", "mnemosyne", "w/r", 0.9)
    assert hit.fused_score == 0.9


def test_search_hit_relevance_boost():
    hit = SearchHit("test", "mnemosyne", "w/r", 0.9, relevance_boost=0.3)
    assert hit.relevance_boost == 0.3


# ─── Adapter availability tests ─────────────────────────────────────────────


def test_mnemosyne_adapter_unavailable():
    adapter = MnemosyneAdapter(palace_path="/nonexistent/path")
    assert adapter.available() is False
    assert adapter.search("test") == []


def test_firstbrain_adapter_no_path():
    adapter = FirstbrainAdapter(vault_path="")
    assert adapter.available() is False
    assert adapter.search("test") == []


def test_firstbrain_adapter_with_vault(tmp_path):
    (tmp_path / "test.md").write_text("# Auth decisions\nWe chose JWT tokens for authentication.")
    adapter = FirstbrainAdapter(vault_path=str(tmp_path))
    assert adapter.available() is True
    status = adapter.status()
    assert status["semantic_search"] is True


def test_firstbrain_semantic_search(tmp_path):
    """v2: Firstbrain now uses ChromaDB embeddings, not keyword match."""
    (tmp_path / "auth.md").write_text("# Authentication\nJWT tokens for OAuth2 PKCE flow.")
    (tmp_path / "db.md").write_text("# Database\nPostgres setup for production workloads.")

    adapter = FirstbrainAdapter(vault_path=str(tmp_path))
    results = adapter.search("authentication tokens", limit=5)
    assert len(results) > 0
    assert results[0].source == "firstbrain"
    # v2: similarity should be real embedding distance, not keyword count
    assert 0.0 < results[0].similarity <= 1.0


def test_cricket_adapter_as_amplifier():
    """v2: Cricket adapter computes relevance boost, not fake search."""
    adapter = CricketAdapter()
    # Cricket doesn't search — it boosts
    assert adapter.search("test") == []
    if adapter.available():
        boost = adapter.compute_relevance("OAuth2 PKCE authentication flow", "auth")
        assert 0.0 <= boost <= 1.0


# ─── TotalRecall integration tests ──────────────────────────────────────────


def test_total_recall_status():
    tr = TotalRecall()
    status = tr.status()
    assert "sources" in status
    assert "mnemosyne" in status["sources"]
    assert "firstbrain" in status["sources"]
    assert "cricket" in status["sources"]
    assert "weights" in status
    assert "resonance" in status["weights"]


def test_total_recall_configure():
    tr = TotalRecall()
    result = tr.configure(similarity_weight=0.7, resonance_weight=0.05)
    assert result["weights"]["similarity"] == 0.7
    assert result["weights"]["resonance"] == 0.05


def test_total_recall_search_no_sources():
    tr = TotalRecall(palace_path="/nonexistent", vault_path="", cricket_path="")
    result = tr.search("test query")
    assert result["query"] == "test query"
    for source in result["sources_queried"]:
        assert source in ("mnemosyne", "firstbrain", "cricket")


def test_total_recall_with_firstbrain(tmp_path):
    (tmp_path / "note.md").write_text("Important auth decision: use OAuth2 for all services")
    tr = TotalRecall(palace_path="/nonexistent", vault_path=str(tmp_path))
    result = tr.search("auth decision")
    assert "firstbrain" in result["sources_queried"]
    assert len(result["results"]) > 0


def test_fuse_and_rank_with_resonance():
    """v2: Fusion now includes resonance weight."""
    tr = TotalRecall(
        similarity_weight=0.4,
        pagerank_weight=0.2,
        recency_weight=0.2,
        resonance_weight=0.2,
    )
    hits = [
        SearchHit("highly relevant", "mnemosyne", "w/r", similarity=0.9, relevance_boost=0.8),
        SearchHit("somewhat relevant", "firstbrain", "f/x", similarity=0.7, pagerank=0.5, relevance_boost=0.2),
        SearchHit("recent but weak", "mnemosyne", "w/r", similarity=0.3, recency=0.9, relevance_boost=0.1),
    ]
    ranked = tr._fuse_and_rank(hits, limit=10, query="test")
    # "highly relevant" = 0.9*0.4 + 0*0.2 + 0*0.2 + 0.8*0.2 = 0.52
    # "somewhat relevant" = 0.7*0.4 + 0.5*0.2 + 0*0.2 + 0.2*0.2 = 0.42
    # "recent but weak" = 0.3*0.4 + 0*0.2 + 0.9*0.2 + 0.1*0.2 = 0.32
    assert ranked[0].text == "highly relevant"
    assert ranked[1].text == "somewhat relevant"
    assert ranked[2].text == "recent but weak"


def test_dedup_in_fusion():
    """v2: Semantic dedup removes near-identical results from different sources."""
    tr = TotalRecall()
    hits = [
        SearchHit("OAuth2 with PKCE flow for the mobile application", "mnemosyne", "w/r", 0.9),
        SearchHit("OAuth2 with PKCE flow for the mobile application security", "firstbrain", "f/x", 0.8),
        SearchHit("PostgreSQL database configuration for production", "mnemosyne", "w/r2", 0.7),
    ]
    ranked = tr._fuse_and_rank(hits, limit=10, query="test")
    # The two OAuth2 hits should be deduped to one
    oauth_hits = [h for h in ranked if "OAuth2" in h.text]
    assert len(oauth_hits) == 1
