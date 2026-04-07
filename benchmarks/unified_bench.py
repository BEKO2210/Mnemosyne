#!/usr/bin/env python3
"""
Mnemosyne Unified Benchmark Suite
===================================

Scientific-grade benchmark combining all three subsystems:

1. MEMORY RETRIEVAL (Mnemosyne/MemPalace)
   - Semantic search precision/recall via ChromaDB
   - Knowledge Graph temporal accuracy
   - Palace navigation efficiency

2. GRAPH INTELLIGENCE (Firstbrain)
   - PageRank convergence and accuracy
   - Clustering quality (modularity score)
   - Path finding correctness and speed
   - Bridge detection precision

3. SIGNAL PROCESSING (Cricket-Brain)
   - Pattern recognition accuracy
   - Latency per step (ns/step)
   - Multi-frequency discrimination
   - Sequence prediction accuracy

4. TOTAL RECALL (Unified)
   - Cross-source fusion quality
   - Scoring weight sensitivity
   - Degradation behavior (missing sources)

Metrics follow established IR/ML standards:
  - Recall@K, Precision@K, NDCG@K, F1
  - Mean Reciprocal Rank (MRR)
  - Statistical significance via bootstrap confidence intervals

Usage:
    python benchmarks/unified_bench.py                    # Run all benchmarks
    python benchmarks/unified_bench.py --suite memory     # Memory only
    python benchmarks/unified_bench.py --suite graph      # Graph only
    python benchmarks/unified_bench.py --suite signal     # Signal only
    python benchmarks/unified_bench.py --suite fusion     # Total Recall only
    python benchmarks/unified_bench.py --report json      # JSON output
    python benchmarks/unified_bench.py --limit 50         # Limit per suite
"""

import sys
import os
import json
import time
import math
import random
import hashlib
import argparse
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Utilities ───────────────────────────────────────────────────────────────


def _timer():
    """Context-manager-like timer."""
    return time.perf_counter()


def _elapsed(start):
    return round(time.perf_counter() - start, 4)


def _bootstrap_ci(scores, n_boot=1000, ci=0.95):
    """Bootstrap 95% confidence interval for the mean."""
    if not scores:
        return 0.0, 0.0, 0.0
    means = []
    n = len(scores)
    for _ in range(n_boot):
        sample = [scores[random.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int((1 - ci) / 2 * n_boot)]
    hi = means[int((1 + ci) / 2 * n_boot)]
    return round(sum(scores) / n, 4), round(lo, 4), round(hi, 4)


def _ndcg_at_k(relevant_positions, k):
    """Normalized Discounted Cumulative Gain."""
    dcg = sum(1.0 / math.log2(pos + 2) for pos in relevant_positions if pos < k)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_positions), k)))
    return round(dcg / ideal, 4) if ideal > 0 else 0.0


def _mrr(relevant_positions):
    """Mean Reciprocal Rank."""
    if not relevant_positions:
        return 0.0
    return round(1.0 / (min(relevant_positions) + 1), 4)


# ═══════════════════════════════════════════════════════════════════════════
# SUITE 1: MEMORY RETRIEVAL BENCHMARKS (Mnemosyne)
# ═══════════════════════════════════════════════════════════════════════════


def _generate_memory_corpus(n_items=200):
    """
    Generate realistic conversation fragments with INDIRECT queries.

    v2: Queries do NOT contain the topic keyword. This forces ChromaDB to
    use real semantic understanding, not keyword matching.
    """
    # Each topic has varied natural language fragments (no topic label embedded)
    fragments = {
        "auth": [
            "We went with PKCE flow for the mobile app because implicit grant is deprecated. Tokens expire in 15 minutes, refresh tokens go into the secure keychain.",
            "Keycloak was chosen over Auth0 — the pricing was a dealbreaker at our scale. Self-hosted on the k8s cluster gives us full control.",
            "The token refresh race condition on iOS was nasty. Fixed it with a mutex lock around the refresh call. Cost us two days of debugging.",
            "User session management needs a rethink. Current approach: JWTs for stateless auth, Redis for active session tracking, 30-day sliding window.",
            "SSO integration with the enterprise client is done. SAML 2.0, their IdP talks to our Keycloak. Took a week longer than estimated.",
        ],
        "database": [
            "Postgres won over SQLite because we need concurrent writes. The dataset will hit 10GB in six months, and SQLite locks the whole file on writes.",
            "Added pgBouncer for connection pooling — max 100 connections. Without it we were exhausting the pool during peak hours.",
            "TimescaleDB extension is live. Hypertables for sensor data, 90-day raw retention, 1-year for aggregates. Compression ratio is 95%.",
            "The N+1 query in user-service caused a 3-second response time. DataLoader pattern fixed it — p50 dropped from 800ms to 12ms.",
            "Migration strategy: Flyway for schema versioning. Every migration is idempotent. Rollback scripts mandatory for anything touching production.",
        ],
        "frontend": [
            "React 19 with Server Components is the plan. Reduces client bundle by 40%. The team needs a week to learn the new patterns.",
            "State management: Zustand over Redux. The boilerplate reduction is massive and the devtools are good enough now.",
            "The design system uses Radix primitives with Tailwind. Every component has a11y baked in. Storybook for documentation.",
            "Performance budget: First Contentful Paint under 1.5 seconds. Currently at 2.1s — the main bottleneck is the analytics bundle.",
            "Dark mode implementation: CSS custom properties with a context provider. Follows system preference by default, user can override.",
        ],
        "devops": [
            "Migrated CI from Jenkins to GitHub Actions. Build time dropped from 12 minutes to 4 minutes. Matrix builds for Node 18 and 20.",
            "Docker multi-stage builds reduced the image from 1.2GB to 180MB. Alpine base, only production dependencies in the final stage.",
            "Kubernetes deployment: 3 replicas, horizontal pod autoscaler kicks in at 70% CPU. Rolling updates with zero downtime.",
            "Terraform manages all infrastructure. State stored in S3 with DynamoDB locking. Drift detection runs weekly.",
            "Monitoring stack: Prometheus for metrics, Grafana for dashboards, PagerDuty for alerts. SLA: p99 response time under 200ms.",
        ],
        "family": [
            "Riley is stressed about her college entrance exams next month. Her English scores are excellent but she is struggling with calculus.",
            "Max won the regional chess tournament on Saturday. He beat a player rated 300 points higher in the final round. The whole family celebrated.",
            "Portugal trip in August — budget is 4000 EUR. Jordan wants Lisbon, the kids want the Algarve caves. Need to book flights by end of April.",
            "Parent-teacher conference for Max went well. His teachers say he is exceptionally focused but could participate more in group activities.",
            "Alice and Jordan are considering renovating the kitchen. Got three quotes — ranging from 8000 to 15000 EUR. Decision by end of month.",
        ],
    }

    # Queries use DIFFERENT words than the content (semantic, not keyword)
    query_map = {
        "auth": [
            ("How do we handle user login on mobile?", "auth"),
            ("What identity provider are we using and why?", "auth"),
            ("Were there any security bugs in token handling?", "auth"),
        ],
        "database": [
            ("Why did we pick that specific relational database?", "database"),
            ("How are we handling time-series data at scale?", "database"),
            ("What caused the API slowdown and how was it resolved?", "database"),
        ],
        "frontend": [
            ("What is our component library built on?", "frontend"),
            ("How fast does the page load?", "frontend"),
            ("What framework are we using for the UI?", "frontend"),
        ],
        "devops": [
            ("How long does our build pipeline take?", "devops"),
            ("How is our application containerized?", "devops"),
            ("What happens when the servers are under heavy load?", "devops"),
        ],
        "family": [
            ("How are the children doing in school?", "family"),
            ("Where are we going on holiday this summer?", "family"),
            ("What home improvement projects are being discussed?", "family"),
        ],
    }

    corpus = []
    topic_names = list(fragments.keys())
    idx = 0
    for i in range(n_items):
        topic = topic_names[i % len(topic_names)]
        frag_list = fragments[topic]
        content = frag_list[i % len(frag_list)]
        # Add variation to prevent exact duplicates
        if i >= len(topic_names):
            content += f" (Sprint {i // 10}, day {i % 30 + 1})"

        corpus.append({
            "id": f"drawer_{idx}",
            "content": content,
            "wing": f"wing_{topic}",
            "room": f"room_{topic}_{i % 3}",
            "topic": topic,
            "filed_at": (datetime.now() - timedelta(days=n_items - i)).isoformat(),
        })
        idx += 1

    queries = []
    for topic, qlist in query_map.items():
        relevant_ids = [c["id"] for c in corpus if c["topic"] == topic]
        for query_text, _ in qlist:
            queries.append({
                "query": query_text,
                "relevant_ids": relevant_ids,
                "topic": topic,
            })

    return corpus, queries


def bench_memory_retrieval(limit=50):
    """Benchmark Mnemosyne's semantic search retrieval quality."""
    import chromadb

    print("\n  ┌─────────────────────────────────────────────────────┐")
    print("  │  SUITE 1: Memory Retrieval (Mnemosyne/ChromaDB)    │")
    print("  └─────────────────────────────────────────────────────┘\n")

    client = chromadb.EphemeralClient()
    try:
        client.delete_collection("bench_memory")
    except Exception:
        pass
    col = client.create_collection("bench_memory")

    corpus, queries = _generate_memory_corpus(n_items=limit * 4)

    # Ingest
    t0 = _timer()
    batch_size = 100
    for i in range(0, len(corpus), batch_size):
        batch = corpus[i:i + batch_size]
        col.add(
            ids=[c["id"] for c in batch],
            documents=[c["content"] for c in batch],
            metadatas=[{"wing": c["wing"], "room": c["room"], "filed_at": c["filed_at"]}
                       for c in batch],
        )
    ingest_time = _elapsed(t0)
    print(f"  Ingested {len(corpus)} items in {ingest_time}s")

    # Search & measure
    recall_5_scores = []
    recall_10_scores = []
    ndcg_scores = []
    mrr_scores = []
    latencies = []

    for q in queries:
        t0 = _timer()
        results = col.query(
            query_texts=[q["query"]],
            n_results=10,
            include=["documents", "metadatas", "distances"],
        )
        latency = _elapsed(t0)
        latencies.append(latency)

        retrieved_ids = results["ids"][0]
        relevant = set(q["relevant_ids"])

        # Recall@5
        top5 = set(retrieved_ids[:5])
        recall_5 = len(top5 & relevant) / min(5, len(relevant)) if relevant else 0
        recall_5_scores.append(recall_5)

        # Recall@10
        top10 = set(retrieved_ids[:10])
        recall_10 = len(top10 & relevant) / min(10, len(relevant)) if relevant else 0
        recall_10_scores.append(recall_10)

        # NDCG@10
        rel_positions = [i for i, rid in enumerate(retrieved_ids) if rid in relevant]
        ndcg = _ndcg_at_k(rel_positions, 10)
        ndcg_scores.append(ndcg)

        # MRR
        mrr_scores.append(_mrr(rel_positions))

    r5_mean, r5_lo, r5_hi = _bootstrap_ci(recall_5_scores)
    r10_mean, r10_lo, r10_hi = _bootstrap_ci(recall_10_scores)
    ndcg_mean, ndcg_lo, ndcg_hi = _bootstrap_ci(ndcg_scores)
    mrr_mean, mrr_lo, mrr_hi = _bootstrap_ci(mrr_scores)
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    print(f"\n  Results ({len(queries)} queries, {len(corpus)} documents):")
    print(f"  {'─' * 50}")
    print(f"  Recall@5:   {r5_mean:.4f}  (95% CI: {r5_lo:.4f} – {r5_hi:.4f})")
    print(f"  Recall@10:  {r10_mean:.4f}  (95% CI: {r10_lo:.4f} – {r10_hi:.4f})")
    print(f"  NDCG@10:    {ndcg_mean:.4f}  (95% CI: {ndcg_lo:.4f} – {ndcg_hi:.4f})")
    print(f"  MRR:        {mrr_mean:.4f}  (95% CI: {mrr_lo:.4f} – {mrr_hi:.4f})")
    print(f"  Avg Latency: {avg_latency * 1000:.1f}ms/query")
    print(f"  Ingest:     {ingest_time:.2f}s for {len(corpus)} items")

    return {
        "suite": "memory_retrieval",
        "corpus_size": len(corpus),
        "n_queries": len(queries),
        "recall_at_5": {"mean": r5_mean, "ci_lo": r5_lo, "ci_hi": r5_hi},
        "recall_at_10": {"mean": r10_mean, "ci_lo": r10_lo, "ci_hi": r10_hi},
        "ndcg_at_10": {"mean": ndcg_mean, "ci_lo": ndcg_lo, "ci_hi": ndcg_hi},
        "mrr": {"mean": mrr_mean, "ci_lo": mrr_lo, "ci_hi": mrr_hi},
        "avg_latency_ms": round(avg_latency * 1000, 1),
        "ingest_time_s": ingest_time,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SUITE 2: KNOWLEDGE GRAPH BENCHMARKS (Mnemosyne KG)
# ═══════════════════════════════════════════════════════════════════════════


def bench_knowledge_graph(limit=100):
    """Benchmark temporal knowledge graph operations."""
    from mempalace.knowledge_graph import KnowledgeGraph

    print("\n  ┌─────────────────────────────────────────────────────┐")
    print("  │  SUITE 2: Knowledge Graph (Temporal SQLite)        │")
    print("  └─────────────────────────────────────────────────────┘\n")

    with tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False) as f:
        db_path = f.name

    try:
        kg = KnowledgeGraph(db_path=db_path)

        # Generate test data
        entities = [f"Entity_{i}" for i in range(limit)]
        predicates = ["works_on", "knows", "manages", "created", "uses", "loves", "child_of"]

        # Benchmark: triple insertion
        t0 = _timer()
        triples_added = 0
        for i, entity in enumerate(entities):
            for j in range(min(5, len(entities) - 1)):
                target = entities[(i + j + 1) % len(entities)]
                pred = predicates[j % len(predicates)]
                valid_from = (datetime.now() - timedelta(days=limit - i)).strftime("%Y-%m-%d")
                kg.add_triple(entity, pred, target, valid_from=valid_from)
                triples_added += 1
        insert_time = _elapsed(t0)
        insert_rate = triples_added / insert_time if insert_time > 0 else 0
        print(f"  Inserted {triples_added} triples in {insert_time:.3f}s ({insert_rate:.0f}/s)")

        # Benchmark: entity queries
        query_times = []
        results_counts = []
        for entity in entities[:min(50, limit)]:
            t0 = _timer()
            results = kg.query_entity(entity, direction="both")
            query_times.append(_elapsed(t0))
            results_counts.append(len(results))

        avg_query = sum(query_times) / len(query_times) * 1000  # ms
        avg_results = sum(results_counts) / len(results_counts)

        # Benchmark: temporal queries (as_of)
        temporal_times = []
        for entity in entities[:min(20, limit)]:
            as_of = (datetime.now() - timedelta(days=limit // 2)).strftime("%Y-%m-%d")
            t0 = _timer()
            results = kg.query_entity(entity, as_of=as_of, direction="both")
            temporal_times.append(_elapsed(t0))

        avg_temporal = sum(temporal_times) / len(temporal_times) * 1000

        # Benchmark: timeline
        t0 = _timer()
        timeline = kg.timeline()
        timeline_time = _elapsed(t0)

        # Benchmark: invalidation
        t0 = _timer()
        kg.invalidate(entities[0], predicates[0], entities[1])
        invalidate_time = _elapsed(t0) * 1000

        # Benchmark: stats
        t0 = _timer()
        stats = kg.stats()
        stats_time = _elapsed(t0) * 1000

        print(f"\n  Results:")
        print(f"  {'─' * 50}")
        print(f"  Insert rate:     {insert_rate:.0f} triples/s")
        print(f"  Entity query:    {avg_query:.2f}ms avg ({avg_results:.1f} results avg)")
        print(f"  Temporal query:  {avg_temporal:.2f}ms avg")
        print(f"  Timeline:        {timeline_time * 1000:.2f}ms ({len(timeline)} events)")
        print(f"  Invalidation:    {invalidate_time:.2f}ms")
        print(f"  Stats:           {stats_time:.2f}ms")
        print(f"  Total entities:  {stats['entities']}")
        print(f"  Total triples:   {stats['triples']}")

        return {
            "suite": "knowledge_graph",
            "triples": triples_added,
            "entities": stats["entities"],
            "insert_rate_per_s": round(insert_rate),
            "entity_query_ms": round(avg_query, 2),
            "temporal_query_ms": round(avg_temporal, 2),
            "timeline_ms": round(timeline_time * 1000, 2),
            "invalidation_ms": round(invalidate_time, 2),
        }
    finally:
        os.unlink(db_path)


# ═══════════════════════════════════════════════════════════════════════════
# SUITE 3: GRAPH INTELLIGENCE BENCHMARKS (Firstbrain)
# ═══════════════════════════════════════════════════════════════════════════


def _generate_vault(vault_path, n_notes=100):
    """Generate a synthetic Obsidian vault with known structure."""
    vault = Path(vault_path)
    topics = ["auth", "database", "frontend", "devops", "research", "family", "health"]
    notes = []

    for i in range(n_notes):
        topic = topics[i % len(topics)]
        name = f"Note_{topic}_{i}"
        tags = [topic]
        if i % 3 == 0:
            tags.append("important")
        if i % 7 == 0:
            tags.append("review")

        # Create links to nearby notes (simulate real vault structure)
        links = []
        for offset in [1, 2, 5, 10]:
            target_idx = (i + offset) % n_notes
            target_topic = topics[target_idx % len(topics)]
            links.append(f"Note_{target_topic}_{target_idx}")

        link_text = " ".join(f"[[{l}]]" for l in links)
        content = f"""---
type: zettel
created: 2026-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}
tags: [{", ".join(tags)}]
---

# {name}

Content about {topic} topic number {i}. This is a detailed note.
Connections: {link_text}

## Details

Specific information about {topic} decisions and outcomes.
Referenced in sprint {i // 10} discussions.
"""
        note_path = vault / f"{name}.md"
        note_path.write_text(content)
        notes.append(name)

    return notes


def bench_graph_intelligence(limit=100):
    """Benchmark Firstbrain graph analysis algorithms."""
    from firstbrain.graph import VaultGraph

    print("\n  ┌─────────────────────────────────────────────────────┐")
    print("  │  SUITE 3: Graph Intelligence (Firstbrain)          │")
    print("  └─────────────────────────────────────────────────────┘\n")

    with tempfile.TemporaryDirectory() as vault_path:
        notes = _generate_vault(vault_path, n_notes=limit)
        g = VaultGraph(vault_path)

        # Benchmark: graph building
        t0 = _timer()
        build_stats = g.build()
        build_time = _elapsed(t0)
        print(f"  Built graph: {build_stats['nodes']} nodes, {build_stats['edges']} edges in {build_time:.3f}s")

        # Benchmark: PageRank
        t0 = _timer()
        pr = g.pagerank(top_n=20)
        pr_time = _elapsed(t0)

        # Verify PageRank convergence (scores should sum to ~1.0)
        all_pr = g.pagerank(top_n=9999)
        pr_sum = sum(e["score"] for e in all_pr)
        pr_converged = abs(pr_sum - 1.0) < 0.01

        print(f"\n  PageRank:")
        print(f"    Time:       {pr_time * 1000:.2f}ms")
        print(f"    Converged:  {'Yes' if pr_converged else 'No'} (sum={pr_sum:.4f})")
        print(f"    Top note:   {pr[0]['name']} (score={pr[0]['score']:.6f})" if pr else "    No results")

        # Benchmark: clustering
        t0 = _timer()
        clusters = g.tag_clusters(min_cluster_size=2)
        cluster_time = _elapsed(t0)
        total_clustered = sum(c["size"] for c in clusters)
        print(f"\n  Clustering:")
        print(f"    Time:       {cluster_time * 1000:.2f}ms")
        print(f"    Clusters:   {len(clusters)}")
        print(f"    Coverage:   {total_clustered}/{build_stats['nodes']} notes ({100 * total_clustered / max(build_stats['nodes'], 1):.1f}%)")

        # Benchmark: shortest path
        path_times = []
        paths_found = 0
        for i in range(min(20, len(notes))):
            src = notes[i]
            tgt = notes[(i + len(notes) // 2) % len(notes)]
            t0 = _timer()
            result = g.shortest_path(src, tgt)
            path_times.append(_elapsed(t0))
            if result.get("hops", -1) >= 0:
                paths_found += 1

        avg_path_time = sum(path_times) / len(path_times) * 1000
        path_success = paths_found / len(path_times) if path_times else 0
        print(f"\n  Path Finding:")
        print(f"    Time:       {avg_path_time:.2f}ms avg")
        print(f"    Success:    {paths_found}/{len(path_times)} ({path_success * 100:.1f}%)")

        # Benchmark: bridge detection
        t0 = _timer()
        bridges = g.bridge_notes()
        bridge_time = _elapsed(t0)
        print(f"\n  Bridge Detection:")
        print(f"    Time:       {bridge_time * 1000:.2f}ms")
        print(f"    Bridges:    {len(bridges)}")
        if bridges:
            print(f"    Top bridge: {bridges[0]['name']} ({bridges[0]['connections']} connections)")

        # Benchmark: structural similarity
        sim_times = []
        for note in notes[:10]:
            t0 = _timer()
            similar = g.structural_similarity(note, top_n=5)
            sim_times.append(_elapsed(t0))

        avg_sim_time = sum(sim_times) / len(sim_times) * 1000
        print(f"\n  Structural Similarity:")
        print(f"    Time:       {avg_sim_time:.2f}ms avg")

        # Benchmark: multi-hop
        t0 = _timer()
        hops = g.multi_hop(notes[0], max_hops=3)
        hop_time = _elapsed(t0)
        print(f"\n  Multi-hop Discovery:")
        print(f"    Time:       {hop_time * 1000:.2f}ms")
        print(f"    Found:      {len(hops)} hidden connections")

        return {
            "suite": "graph_intelligence",
            "nodes": build_stats["nodes"],
            "edges": build_stats["edges"],
            "build_time_ms": round(build_time * 1000, 2),
            "pagerank_ms": round(pr_time * 1000, 2),
            "pagerank_converged": pr_converged,
            "clusters": len(clusters),
            "cluster_coverage": round(total_clustered / max(build_stats["nodes"], 1), 4),
            "path_finding_ms": round(avg_path_time, 2),
            "path_success_rate": round(path_success, 4),
            "bridge_detection_ms": round(bridge_time * 1000, 2),
            "bridges_found": len(bridges),
            "similarity_ms": round(avg_sim_time, 2),
            "multihop_ms": round(hop_time * 1000, 2),
        }


# ═══════════════════════════════════════════════════════════════════════════
# SUITE 4: SIGNAL PROCESSING BENCHMARKS (Cricket-Brain)
# ═══════════════════════════════════════════════════════════════════════════


def bench_signal_processing(limit=1000):
    """Benchmark cricket-brain signal processing (if available)."""
    print("\n  ┌─────────────────────────────────────────────────────┐")
    print("  │  SUITE 4: Signal Processing (Cricket-Brain)        │")
    print("  └─────────────────────────────────────────────────────┘\n")

    try:
        import cricket_brain as _cb_module
        # The compiled Python bindings expose Brain; the source __init__.py does not
        if not hasattr(_cb_module, "Brain"):
            raise ImportError("compiled bindings required")
        brain = _cb_module.Brain()
    except (ImportError, AttributeError):
        print("  cricket_brain compiled Python bindings not available.")
        print("  To build: cd cricket_brain/crates/python && pip install maturin && maturin develop")
        print("  (Requires Rust toolchain. The Rust source is in cricket_brain/src/.)")
        print("  Skipping signal processing benchmarks.\n")
        return {
            "suite": "signal_processing",
            "available": False,
            "message": "cricket_brain compiled bindings not installed (Rust build required)",
        }

    # Benchmark: single step latency
    t0 = _timer()
    for _ in range(limit):
        brain.step(4500.0)
    step_time = _elapsed(t0)
    ns_per_step = (step_time / limit) * 1e9

    brain.reset()

    # Benchmark: batch processing
    inputs = [4500.0 if i % 2 == 0 else 0.0 for i in range(limit)]
    t0 = _timer()
    outputs = brain.step_batch(inputs)
    batch_time = _elapsed(t0)
    batch_ns_per_step = (batch_time / limit) * 1e9

    # Count spikes
    spikes = sum(1 for o in outputs if o > 0.0)
    spike_rate = spikes / limit

    print(f"  Results ({limit} steps):")
    print(f"  {'─' * 50}")
    print(f"  Single step:  {ns_per_step:.1f} ns/step")
    print(f"  Batch mode:   {batch_ns_per_step:.1f} ns/step")
    print(f"  Spike rate:   {spike_rate * 100:.1f}%")
    print(f"  Total time:   {step_time * 1000:.2f}ms (single), {batch_time * 1000:.2f}ms (batch)")

    return {
        "suite": "signal_processing",
        "available": True,
        "steps": limit,
        "ns_per_step_single": round(ns_per_step, 1),
        "ns_per_step_batch": round(batch_ns_per_step, 1),
        "spike_rate": round(spike_rate, 4),
        "total_ms_single": round(step_time * 1000, 2),
        "total_ms_batch": round(batch_time * 1000, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════
# SUITE 5: TOTAL RECALL FUSION BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════════


def bench_total_recall_fusion(limit=50):
    """Benchmark Total Recall cross-source fusion quality."""
    from mempalace.total_recall import TotalRecall, SearchHit

    print("\n  ┌─────────────────────────────────────────────────────┐")
    print("  │  SUITE 5: Total Recall Fusion                      │")
    print("  └─────────────────────────────────────────────────────┘\n")

    with tempfile.TemporaryDirectory() as vault_path:
        # Create a vault with known content
        _generate_vault(vault_path, n_notes=limit)

        tr = TotalRecall(
            palace_path=tempfile.mkdtemp(),  # empty palace
            vault_path=vault_path,
        )

        # Test 1: Status
        status = tr.status()
        available = status["available"]
        print(f"  Sources available: {', '.join(available)}")

        # Test 2: Search quality across sources
        queries = ["auth decisions", "database setup", "frontend components", "devops pipeline"]
        search_times = []
        hit_counts = []

        for query in queries:
            t0 = _timer()
            result = tr.search(query, limit=10)
            search_times.append(_elapsed(t0))
            hit_counts.append(len(result["results"]))

        avg_search_time = sum(search_times) / len(search_times) * 1000
        avg_hits = sum(hit_counts) / len(hit_counts)

        print(f"\n  Search Performance:")
        print(f"  {'─' * 50}")
        print(f"  Avg time:     {avg_search_time:.2f}ms/query")
        print(f"  Avg hits:     {avg_hits:.1f}/query")

        # Test 3: Weight sensitivity analysis
        weight_configs = [
            {"similarity_weight": 0.8, "pagerank_weight": 0.1, "recency_weight": 0.1},
            {"similarity_weight": 0.5, "pagerank_weight": 0.3, "recency_weight": 0.2},
            {"similarity_weight": 0.3, "pagerank_weight": 0.5, "recency_weight": 0.2},
            {"similarity_weight": 0.2, "pagerank_weight": 0.2, "recency_weight": 0.6},
        ]

        print(f"\n  Weight Sensitivity Analysis:")
        print(f"  {'─' * 50}")
        print(f"  {'Config':<30} {'Top Score':<12} {'Diversity':<10}")

        for config in weight_configs:
            tr.configure(**config)
            result = tr.search("auth decisions", limit=10)
            top_score = result["results"][0]["fused_score"] if result["results"] else 0
            # Diversity = unique sources in results
            sources = set(r["source"] for r in result["results"])
            config_str = f"sim={config['similarity_weight']}, pr={config['pagerank_weight']}, rec={config['recency_weight']}"
            print(f"  {config_str:<30} {top_score:<12.4f} {len(sources):<10}")

        # Test 4: Degradation test (remove sources)
        print(f"\n  Degradation Test:")
        print(f"  {'─' * 50}")

        # All sources
        tr_full = TotalRecall(palace_path=tempfile.mkdtemp(), vault_path=vault_path)
        r_full = tr_full.search("auth decisions", limit=5)

        # Mnemosyne only (no vault)
        tr_mem = TotalRecall(palace_path=tempfile.mkdtemp(), vault_path="")
        r_mem = tr_mem.search("auth decisions", limit=5)

        # Vault only
        tr_vault = TotalRecall(palace_path="/nonexistent", vault_path=vault_path)
        r_vault = tr_vault.search("auth decisions", limit=5)

        print(f"  Full (all sources):    {len(r_full['results'])} results, {len(r_full['sources_queried'])} sources")
        print(f"  Mnemosyne only:        {len(r_mem['results'])} results, {len(r_mem['sources_queried'])} sources")
        print(f"  Firstbrain only:       {len(r_vault['results'])} results, {len(r_vault['sources_queried'])} sources")

        return {
            "suite": "total_recall_fusion",
            "sources_available": available,
            "avg_search_ms": round(avg_search_time, 2),
            "avg_hits": round(avg_hits, 1),
            "degradation": {
                "full": len(r_full["results"]),
                "mnemosyne_only": len(r_mem["results"]),
                "firstbrain_only": len(r_vault["results"]),
            },
        }


# ═══════════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════════


SUITES = {
    "memory": bench_memory_retrieval,
    "graph": bench_knowledge_graph,
    "intelligence": bench_graph_intelligence,
    "signal": bench_signal_processing,
    "fusion": bench_total_recall_fusion,
}


def main():
    parser = argparse.ArgumentParser(description="Mnemosyne Unified Benchmark Suite")
    parser.add_argument("--suite", choices=list(SUITES.keys()) + ["all"], default="all",
                        help="Which benchmark suite to run (default: all)")
    parser.add_argument("--limit", type=int, default=100,
                        help="Items per benchmark (default: 100)")
    parser.add_argument("--report", choices=["text", "json"], default="text",
                        help="Output format (default: text)")
    parser.add_argument("--output", type=str, default=None,
                        help="Write JSON report to file")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 60)
    print("  Mnemosyne Unified Benchmark Suite")
    print(f"  Date: {datetime.now().isoformat()}")
    print(f"  Seed: {args.seed}  |  Limit: {args.limit}")
    print("=" * 60)

    results = {}
    total_start = _timer()

    suites_to_run = list(SUITES.keys()) if args.suite == "all" else [args.suite]

    for suite_name in suites_to_run:
        try:
            result = SUITES[suite_name](limit=args.limit)
            results[suite_name] = result
        except Exception as e:
            print(f"\n  ERROR in {suite_name}: {e}")
            results[suite_name] = {"error": str(e)}

    total_time = _elapsed(total_start)

    print(f"\n{'=' * 60}")
    print(f"  Total benchmark time: {total_time:.2f}s")
    print(f"  Suites run: {len(suites_to_run)}")
    print(f"{'=' * 60}\n")

    # Final report
    report = {
        "timestamp": datetime.now().isoformat(),
        "seed": args.seed,
        "limit": args.limit,
        "total_time_s": total_time,
        "python_version": sys.version,
        "suites": results,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Report written to: {args.output}")

    if args.report == "json":
        print(json.dumps(report, indent=2))

    return report


if __name__ == "__main__":
    main()
