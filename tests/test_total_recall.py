"""Tests for total_recall.py — unified knowledge API."""

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
)


# ─── Utility function tests ─────────────────────────────────────────────────


def test_compute_recency_today():
    """Today should be ~1.0."""
    from datetime import datetime
    score = _compute_recency(datetime.now().isoformat())
    assert score >= 0.99


def test_compute_recency_old_date():
    """A year ago should be very low."""
    score = _compute_recency("2020-01-01")
    assert score < 0.01


def test_compute_recency_invalid():
    """Invalid date returns 0.0."""
    assert _compute_recency("not-a-date") == 0.0
    assert _compute_recency("") == 0.0


def test_extract_snippet():
    content = "Line one\nLine two\nLine three has auth topic\nLine four\nLine five"
    snippet = _extract_snippet(content, {"auth"}, max_chars=500)
    assert "auth" in snippet


def test_count_backlinks(tmp_path):
    """Count [[wikilinks]] across files."""
    f1 = tmp_path / "note1.md"
    f1.write_text("See [[note2]] and [[note3]]")
    f2 = tmp_path / "note2.md"
    f2.write_text("Link to [[note3]]")
    f3 = tmp_path / "note3.md"
    f3.write_text("No links here")

    counts = _count_backlinks([f1, f2, f3])
    assert counts["note2"] == 1
    assert counts["note3"] == 2


# ─── SearchHit tests ────────────────────────────────────────────────────────


def test_search_hit_fused_score():
    hit = SearchHit(
        text="test",
        source="mnemosyne",
        location="wing_code/auth",
        similarity=0.9,
        metadata={"fused_score": 0.75},
    )
    assert hit.fused_score == 0.75


def test_search_hit_default_fused_score():
    hit = SearchHit(
        text="test",
        source="mnemosyne",
        location="wing_code/auth",
        similarity=0.9,
    )
    # Without explicit fused_score, falls back to similarity
    assert hit.fused_score == 0.9


# ─── Adapter availability tests ─────────────────────────────────────────────


def test_mnemosyne_adapter_unavailable():
    """Mnemosyne adapter returns unavailable when palace doesn't exist."""
    adapter = MnemosyneAdapter(palace_path="/nonexistent/path")
    assert adapter.available() is False
    assert adapter.search("test") == []


def test_firstbrain_adapter_no_path():
    """Firstbrain adapter unavailable without vault path."""
    adapter = FirstbrainAdapter(vault_path="")
    assert adapter.available() is False
    assert adapter.search("test") == []


def test_firstbrain_adapter_with_vault(tmp_path):
    """Firstbrain adapter works with a real directory."""
    (tmp_path / "test.md").write_text("# Auth decisions\nWe chose JWT tokens.")
    adapter = FirstbrainAdapter(vault_path=str(tmp_path))
    assert adapter.available() is True

    status = adapter.status()
    assert status["name"] == "firstbrain"
    assert status["available"] is True


def test_firstbrain_search(tmp_path):
    """Search finds matching notes and scores them."""
    (tmp_path / "auth.md").write_text("# Authentication\nJWT tokens for auth flow.")
    (tmp_path / "db.md").write_text("# Database\nPostgres setup for production.")
    (tmp_path / "links.md").write_text("See [[auth]] for details on [[auth]].")

    adapter = FirstbrainAdapter(vault_path=str(tmp_path))
    results = adapter.search("authentication JWT", limit=5)
    assert len(results) > 0
    assert results[0].source == "firstbrain"
    assert results[0].similarity > 0


def test_cricket_adapter_unavailable():
    """Cricket adapter unavailable without bindings."""
    adapter = CricketAdapter(engine_path="/nonexistent")
    # Will be False unless cricket_brain is actually installed
    assert adapter.search("test") == []


# ─── TotalRecall integration tests ──────────────────────────────────────────


def test_total_recall_status():
    """Status should always return all three sources."""
    tr = TotalRecall()
    status = tr.status()
    assert "sources" in status
    assert "mnemosyne" in status["sources"]
    assert "firstbrain" in status["sources"]
    assert "cricket" in status["sources"]
    assert "weights" in status


def test_total_recall_configure():
    """Configure should update weights."""
    tr = TotalRecall()
    result = tr.configure(similarity_weight=0.7, recency_weight=0.1)
    assert result["weights"]["similarity"] == 0.7
    assert result["weights"]["recency"] == 0.1
    assert result["weights"]["pagerank"] == 0.3  # unchanged


def test_total_recall_search_no_sources():
    """Search with no available sources returns empty or minimal results."""
    tr = TotalRecall(
        palace_path="/nonexistent",
        vault_path="",
        cricket_path="",
    )
    result = tr.search("test query")
    assert result["query"] == "test query"
    # Only sources that are truly available will be queried
    for source in result["sources_queried"]:
        assert source in ("mnemosyne", "firstbrain", "cricket")


def test_total_recall_with_firstbrain(tmp_path):
    """Search with Firstbrain vault returns results."""
    (tmp_path / "note.md").write_text("Important auth decision: use OAuth2")
    tr = TotalRecall(
        palace_path="/nonexistent",
        vault_path=str(tmp_path),
        cricket_path="",
    )
    result = tr.search("auth decision")
    assert len(result["sources_queried"]) >= 1
    assert "firstbrain" in result["sources_queried"]
    assert len(result["results"]) > 0


def test_fuse_and_rank():
    """Fused scoring ranks results correctly."""
    tr = TotalRecall(similarity_weight=0.5, pagerank_weight=0.3, recency_weight=0.2)
    hits = [
        SearchHit("old but relevant", "mnemosyne", "w/r", similarity=0.9, recency=0.1),
        SearchHit("new but weak", "firstbrain", "notes/x", similarity=0.3, pagerank=0.5, recency=0.9),
        SearchHit("balanced", "mnemosyne", "w/r", similarity=0.7, pagerank=0.2, recency=0.6),
    ]
    ranked = tr._fuse_and_rank(hits, limit=10)
    # "old but relevant" should score: 0.9*0.5 + 0*0.3 + 0.1*0.2 = 0.47
    # "new but weak": 0.3*0.5 + 0.5*0.3 + 0.9*0.2 = 0.48
    # "balanced": 0.7*0.5 + 0.2*0.3 + 0.6*0.2 = 0.53
    assert ranked[0].text == "balanced"
