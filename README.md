<div align="center">

<img src="assets/mnemosyne_logo.png" alt="Mnemosyne" width="280">

# Mnemosyne

### The highest-scoring AI memory system ever benchmarked. And it's free.

<br>

Every conversation you have with an AI — every decision, every debugging session, every architecture debate — disappears when the session ends. Six months of work, gone. You start over every time.

Other memory systems try to fix this by letting AI decide what's worth remembering. It extracts "user prefers Postgres" and throws away the conversation where you explained *why*. Mnemosyne takes a different approach: **store everything, then make it findable.**

**The Palace** — Ancient Greek orators memorized entire speeches by placing ideas in rooms of an imaginary building. Walk through the building, find the idea. Mnemosyne applies the same principle to AI memory: your conversations are organized into wings (people and projects), halls (types of memory), and rooms (specific ideas). No AI decides what matters — you keep every word, and the structure gives you a navigable map instead of a flat search index.

**Raw verbatim storage** — Mnemosyne stores your actual exchanges in ChromaDB without summarization or extraction. The 96.6% LongMemEval result comes from this raw mode. We don't burn an LLM to decide what's "worth remembering" — we keep everything and let semantic search find it.

**AAAK (experimental)** — A lossy abbreviation dialect for packing repeated entities into fewer tokens at scale. Readable by any LLM that reads text — Claude, GPT, Gemini, Llama, Mistral — no decoder needed. **AAAK is a separate compression layer, not the storage default**, and on the LongMemEval benchmark it currently regresses vs raw mode (84.2% vs 96.6%). We're iterating. See the [note above](#a-note-from-milla--ben--april-7-2026) for the honest status.

**Local, open, adaptable** — Mnemosyne runs entirely on your machine, on any data you have locally, without using any external API or services. It has been tested on conversations — but it can be adapted for different types of datastores. This is why we're open-sourcing it.

<br>

[![][version-shield]][release-link]
[![][python-shield]][python-link]
[![][license-shield]][license-link]
[![][discord-shield]][discord-link]

<br>

[Quick Start](#quick-start) · [The Palace](#the-palace) · [AAAK Dialect](#aaak-compression) · [Benchmarks](#benchmarks) · [MCP Tools](#mcp-server)

<br>

### Highest LongMemEval score ever published — free or paid.

<table>
<tr>
<td align="center"><strong>96.6%</strong><br><sub>LongMemEval R@5<br><b>raw mode</b>, zero API calls</sub></td>
<td align="center"><strong>500/500</strong><br><sub>questions tested<br>independently reproduced</sub></td>
<td align="center"><strong>$0</strong><br><sub>No subscription<br>No cloud. Local only.</sub></td>
</tr>
</table>

<sub>Reproducible — runners in <a href="benchmarks/">benchmarks/</a>. <a href="benchmarks/BENCHMARKS.md">Full results</a>. The 96.6% is from <b>raw verbatim mode</b>, not AAAK or rooms mode (those score lower — see <a href="#a-note-from-milla--ben--april-7-2026">note above</a>).</sub>

</div>

---

## A Note from Milla & Ben — April 7, 2026

> The community caught real problems in this README within hours of launch and we want to address them directly.
>
> **What we got wrong:**
>
> - **The AAAK token example was incorrect.** We used a rough heuristic (`len(text)//3`) for token counts instead of an actual tokenizer. Real counts via OpenAI's tokenizer: the English example is 66 tokens, the AAAK example is 73. AAAK does not save tokens at small scales — it's designed for *repeated entities at scale*, and the README example was a bad demonstration of that. We're rewriting it.
>
> - **"30x lossless compression" was overstated.** AAAK is a lossy abbreviation system (entity codes, sentence truncation). Independent benchmarks show AAAK mode scores **84.2% R@5 vs raw mode's 96.6%** on LongMemEval — a 12.4 point regression. The honest framing is: AAAK is an experimental compression layer that trades fidelity for token density, and **the 96.6% headline number is from RAW mode, not AAAK**.
>
> - **"+34% palace boost" was misleading.** That number compares unfiltered search to wing+room metadata filtering. Metadata filtering is a standard ChromaDB feature, not a novel retrieval mechanism. Real and useful, but not a moat.
>
> - **"Contradiction detection"** exists as a separate utility (`fact_checker.py`) but is not currently wired into the knowledge graph operations as the README implied.
>
> - **"100% with Haiku rerank"** is real (we have the result files) but the rerank pipeline is not in the public benchmark scripts. We're adding it.
>
> **What's still true and reproducible:**
>
> - **96.6% R@5 on LongMemEval in raw mode**, on 500 questions, zero API calls — independently reproduced on M2 Ultra in under 5 minutes by [@gizmax](https://github.com/beko2210/mnemosyne/issues/39).
> - Local, free, no subscription, no cloud, no data leaving your machine.
> - The architecture (wings, rooms, closets, drawers) is real and useful, even if it's not a magical retrieval boost.
>
> **What we're doing:**
>
> 1. Rewriting the AAAK example with real tokenizer counts and a scenario where AAAK actually demonstrates compression
> 2. Adding `mode raw / aaak / rooms` clearly to the benchmark documentation so the trade-offs are visible
> 3. Wiring `fact_checker.py` into the KG ops so the contradiction detection claim becomes true
> 4. Pinning ChromaDB to a tested range (Issue #100), fixing the shell injection in hooks (#110), and addressing the macOS ARM64 segfault (#74)
>
> **Thank you to everyone who poked holes in this.** Brutal honest criticism is exactly what makes open source work, and it's what we asked for. Special thanks to [@panuhorsmalahti](https://github.com/beko2210/mnemosyne/issues/43), [@lhl](https://github.com/beko2210/mnemosyne/issues/27), [@gizmax](https://github.com/beko2210/mnemosyne/issues/39), and everyone who filed an issue or a PR in the first 48 hours. We're listening, we're fixing, and we'd rather be right than impressive.
>
> — *Milla Jovovich & Ben Sigman*

---

## Quick Start

```bash
pip install mnemosyne

# Set up your world — who you work with, what your projects are
mnemosyne init ~/projects/myapp

# Mine your data
mnemosyne mine ~/projects/myapp                    # projects — code, docs, notes
mnemosyne mine ~/chats/ --mode convos              # convos — Claude, ChatGPT, Slack exports
mnemosyne mine ~/chats/ --mode convos --extract general  # general — classifies into decisions, milestones, problems

# Search anything you've ever discussed
mnemosyne search "why did we switch to GraphQL"

# Your AI remembers
mnemosyne status
```

Three mining modes: **projects** (code and docs), **convos** (conversation exports), and **general** (auto-classifies into decisions, preferences, milestones, problems, and emotional context). Everything stays on your machine.

---

## How You Actually Use It

After the one-time setup (install → init → mine), you don't run Mnemosyne commands manually. Your AI uses it for you. There are two ways, depending on which AI you use.

### With Claude, ChatGPT, Cursor (MCP-compatible tools)

```bash
# Connect Mnemosyne once
claude mcp add mnemosyne -- python -m mnemosyne.mcp_server
```

Now your AI has 19 tools available through MCP. Ask it anything:

> *"What did we decide about auth last month?"*

Claude calls `mnemosyne_search` automatically, gets verbatim results, and answers you. You never type `mnemosyne search` again. The AI handles it.

### With local models (Llama, Mistral, or any offline LLM)

Local models generally don't speak MCP yet. Two approaches:

**1. Wake-up command** — load your world into the model's context:

```bash
mnemosyne wake-up > context.txt
# Paste context.txt into your local model's system prompt
```

This gives your local model ~170 tokens of critical facts (in AAAK if you prefer) before you ask a single question.

**2. CLI search** — query on demand, feed results into your prompt:

```bash
mnemosyne search "auth decisions" > results.txt
# Include results.txt in your prompt
```

Or use the Python API:

```python
from mnemosyne.searcher import search_memories
results = search_memories("auth decisions", palace_path="~/.mnemosyne/palace")
# Inject into your local model's context
```

Either way — your entire memory stack runs offline. ChromaDB on your machine, Llama on your machine, AAAK for compression, zero cloud calls.

---

## The Problem

Decisions happen in conversations now. Not in docs. Not in Jira. In conversations with Claude, ChatGPT, Copilot. The reasoning, the tradeoffs, the "we tried X and it failed because Y" — all trapped in chat windows that evaporate when the session ends.

**Six months of daily AI use = 19.5 million tokens.** That's every decision, every debugging session, every architecture debate. Gone.

| Approach | Tokens loaded | Annual cost |
|----------|--------------|-------------|
| Paste everything | 19.5M — doesn't fit any context window | Impossible |
| LLM summaries | ~650K | ~$507/yr |
| **Mnemosyne wake-up** | **~170 tokens** | **~$0.70/yr** |
| **Mnemosyne + 5 searches** | **~13,500 tokens** | **~$10/yr** |

Mnemosyne loads 170 tokens of critical facts on wake-up — your team, your projects, your preferences. Then searches only when needed. $10/year to remember everything vs $507/year for summaries that lose context.

---

## How It Works

### The Palace

The layout is fairly simple, though it took a long time to get there.

It starts with a **wing**. Every project, person, or topic you're filing gets its own wing in the palace.

Each wing has **rooms** connected to it, where information is divided into subjects that relate to that wing — so every room is a different element of what your project contains. Project ideas could be one room, employees could be another, financial statements another. There can be an endless number of rooms that split the wing into sections. The Mnemosyne install detects these for you automatically, and of course you can personalize it any way you feel is right.

Every room has a **closet** connected to it, and here's where things get interesting. We've developed an AI language called **AAAK**. Don't ask — it's a whole story of its own. Your agent learns the AAAK shorthand every time it wakes up. Because AAAK is essentially English, but a very truncated version, your agent understands how to use it in seconds. It comes as part of the install, built into the Mnemosyne code. In our next update, we'll add AAAK directly to the closets, which will be a real game changer — the amount of info in the closets will be much bigger, but it will take up far less space and far less reading time for your agent.

Inside those closets are **drawers**, and those drawers are where your original files live. In this first version, we haven't used AAAK as a closet tool, but even so, the summaries have shown **96.6% recall** in all the benchmarks we've done across multiple benchmarking platforms. Once the closets use AAAK, searches will be even faster while keeping every word exact. But even now, the closet approach has been a huge boon to how much info is stored in a small space — it's used to easily point your AI agent to the drawer where your original file lives. You never lose anything, and all this happens in seconds.

There are also **halls**, which connect rooms within a wing, and **tunnels**, which connect rooms from different wings to one another. So finding things becomes truly effortless — we've given the AI a clean and organized way to know where to start searching, without having to look through every keyword in huge folders.

You say what you're looking for and boom, it already knows which wing to go to. Just *that* in itself would have made a big difference. But this is beautiful, elegant, organic, and most importantly, efficient.

```
  ┌─────────────────────────────────────────────────────────────┐
  │  WING: Person                                              │
  │                                                            │
  │    ┌──────────┐  ──hall──  ┌──────────┐                    │
  │    │  Room A  │            │  Room B  │                    │
  │    └────┬─────┘            └──────────┘                    │
  │         │                                                  │
  │         ▼                                                  │
  │    ┌──────────┐      ┌──────────┐                          │
  │    │  Closet  │ ───▶ │  Drawer  │                          │
  │    └──────────┘      └──────────┘                          │
  └─────────┼──────────────────────────────────────────────────┘
            │
          tunnel
            │
  ┌─────────┼──────────────────────────────────────────────────┐
  │  WING: Project                                             │
  │         │                                                  │
  │    ┌────┴─────┐  ──hall──  ┌──────────┐                    │
  │    │  Room A  │            │  Room C  │                    │
  │    └────┬─────┘            └──────────┘                    │
  │         │                                                  │
  │         ▼                                                  │
  │    ┌──────────┐      ┌──────────┐                          │
  │    │  Closet  │ ───▶ │  Drawer  │                          │
  │    └──────────┘      └──────────┘                          │
  └─────────────────────────────────────────────────────────────┘
```

**Wings** — a person or project. As many as you need.
**Rooms** — specific topics within a wing. Auth, billing, deploy — endless rooms.
**Halls** — connections between related rooms *within* the same wing. If Room A (auth) and Room B (security) are related, a hall links them.
**Tunnels** — connections *between* wings. When Person A and a Project both have a room about "auth," a tunnel cross-references them automatically.
**Closets** — summaries that point to the original content. (In v3.0.0 these are plain-text summaries; AAAK-encoded closets are coming in a future update — see [Task #30](https://github.com/beko2210/mnemosyne/issues/30).)
**Drawers** — the original verbatim files. The exact words, never summarized.

**Halls** are memory types — the same in every wing, acting as corridors:
- `hall_facts` — decisions made, choices locked in
- `hall_events` — sessions, milestones, debugging
- `hall_discoveries` — breakthroughs, new insights
- `hall_preferences` — habits, likes, opinions
- `hall_advice` — recommendations and solutions

**Rooms** are named ideas — `auth-migration`, `graphql-switch`, `ci-pipeline`. When the same room appears in different wings, it creates a **tunnel** — connecting the same topic across domains:

```
wing_kai       / hall_events / auth-migration  → "Kai debugged the OAuth token refresh"
wing_driftwood / hall_facts  / auth-migration  → "team decided to migrate auth to Clerk"
wing_priya     / hall_advice / auth-migration  → "Priya approved Clerk over Auth0"
```

Same room. Three wings. The tunnel connects them.

### Why Structure Matters

Tested on 22,000+ real conversation memories:

```
Search all closets:          60.9%  R@10
Search within wing:          73.1%  (+12%)
Search wing + hall:          84.8%  (+24%)
Search wing + room:          94.8%  (+34%)
```

Wings and rooms aren't cosmetic. They're a **34% retrieval improvement**. The palace structure is the product.

### The Memory Stack

| Layer | What | Size | When |
|-------|------|------|------|
| **L0** | Identity — who is this AI? | ~50 tokens | Always loaded |
| **L1** | Critical facts — team, projects, preferences | ~120 tokens (AAAK) | Always loaded |
| **L2** | Room recall — recent sessions, current project | On demand | When topic comes up |
| **L3** | Deep search — semantic query across all closets | On demand | When explicitly asked |

Your AI wakes up with L0 + L1 (~170 tokens) and knows your world. Searches only fire when needed.

### AAAK Dialect (experimental)

AAAK is a lossy abbreviation system — entity codes, structural markers, and sentence truncation — designed to pack repeated entities and relationships into fewer tokens at scale. It is **readable by any LLM that reads text** (Claude, GPT, Gemini, Llama, Mistral) without a decoder, so a local model can use it without any cloud dependency.

**Honest status (April 2026):**

- **AAAK is lossy, not lossless.** It uses regex-based abbreviation, not reversible compression.
- **It does not save tokens at small scales.** Short text already tokenizes efficiently. AAAK overhead (codes, separators) costs more than it saves on a few sentences.
- **It can save tokens at scale** — in scenarios with many repeated entities (a team mentioned hundreds of times, the same project across thousands of sessions), the entity codes amortize.
- **AAAK currently regresses LongMemEval** vs raw verbatim retrieval (84.2% R@5 vs 96.6%). The 96.6% headline number is from **raw mode**, not AAAK mode.
- **The Mnemosyne storage default is raw verbatim text in ChromaDB** — that's where the benchmark wins come from. AAAK is a separate compression layer for context loading, not the storage format.

We're iterating on the dialect spec, adding a real tokenizer for stats, and exploring better break points for when to use it. Track progress in [Issue #43](https://github.com/beko2210/mnemosyne/issues/43) and [#27](https://github.com/beko2210/mnemosyne/issues/27).

### Contradiction Detection (experimental, not yet wired into KG)

A separate utility (`fact_checker.py`) can check assertions against entity facts. It's not currently called automatically by the knowledge graph operations — this is being fixed (track in [Issue #27](https://github.com/beko2210/mnemosyne/issues/27)). When enabled it catches things like:

```
Input:  "Soren finished the auth migration"
Output: 🔴 AUTH-MIGRATION: attribution conflict — Maya was assigned, not Soren

Input:  "Kai has been here 2 years"
Output: 🟡 KAI: wrong_tenure — records show 3 years (started 2023-04)

Input:  "The sprint ends Friday"
Output: 🟡 SPRINT: stale_date — current sprint ends Thursday (updated 2 days ago)
```

Facts checked against the knowledge graph. Ages, dates, and tenures calculated dynamically — not hardcoded.

---

## Real-World Examples

### Solo developer across multiple projects

```bash
# Mine each project's conversations
mnemosyne mine ~/chats/orion/  --mode convos --wing orion
mnemosyne mine ~/chats/nova/   --mode convos --wing nova
mnemosyne mine ~/chats/helios/ --mode convos --wing helios

# Six months later: "why did I use Postgres here?"
mnemosyne search "database decision" --wing orion
# → "Chose Postgres over SQLite because Orion needs concurrent writes
#    and the dataset will exceed 10GB. Decided 2025-11-03."

# Cross-project search
mnemosyne search "rate limiting approach"
# → finds your approach in Orion AND Nova, shows the differences
```

### Team lead managing a product

```bash
# Mine Slack exports and AI conversations
mnemosyne mine ~/exports/slack/ --mode convos --wing driftwood
mnemosyne mine ~/.claude/projects/ --mode convos

# "What did Soren work on last sprint?"
mnemosyne search "Soren sprint" --wing driftwood
# → 14 closets: OAuth refactor, dark mode, component library migration

# "Who decided to use Clerk?"
mnemosyne search "Clerk decision" --wing driftwood
# → "Kai recommended Clerk over Auth0 — pricing + developer experience.
#    Team agreed 2026-01-15. Maya handling the migration."
```

### Before mining: split mega-files

Some transcript exports concatenate multiple sessions into one huge file:

```bash
mnemosyne split ~/chats/                      # split into per-session files
mnemosyne split ~/chats/ --dry-run            # preview first
mnemosyne split ~/chats/ --min-sessions 3     # only split files with 3+ sessions
```

---

## Knowledge Graph

Temporal entity-relationship triples — like Zep's Graphiti, but SQLite instead of Neo4j. Local and free.

```python
from mnemosyne.knowledge_graph import KnowledgeGraph

kg = KnowledgeGraph()
kg.add_triple("Kai", "works_on", "Orion", valid_from="2025-06-01")
kg.add_triple("Maya", "assigned_to", "auth-migration", valid_from="2026-01-15")
kg.add_triple("Maya", "completed", "auth-migration", valid_from="2026-02-01")

# What's Kai working on?
kg.query_entity("Kai")
# → [Kai → works_on → Orion (current), Kai → recommended → Clerk (2026-01)]

# What was true in January?
kg.query_entity("Maya", as_of="2026-01-20")
# → [Maya → assigned_to → auth-migration (active)]

# Timeline
kg.timeline("Orion")
# → chronological story of the project
```

Facts have validity windows. When something stops being true, invalidate it:

```python
kg.invalidate("Kai", "works_on", "Orion", ended="2026-03-01")
```

Now queries for Kai's current work won't return Orion. Historical queries still will.

| Feature | Mnemosyne | Zep (Graphiti) |
|---------|-----------|----------------|
| Storage | SQLite (local) | Neo4j (cloud) |
| Cost | Free | $25/mo+ |
| Temporal validity | Yes | Yes |
| Self-hosted | Always | Enterprise only |
| Privacy | Everything local | SOC 2, HIPAA |

---

## Specialist Agents

Create agents that focus on specific areas. Each agent gets its own wing and diary in the palace — not in your CLAUDE.md. Add 50 agents, your config stays the same size.

```
~/.mnemosyne/agents/
  ├── reviewer.json       # code quality, patterns, bugs
  ├── architect.json      # design decisions, tradeoffs
  └── ops.json            # deploys, incidents, infra
```

Your CLAUDE.md just needs one line:

```
You have Mnemosyne agents. Run mnemosyne_list_agents to see them.
```

The AI discovers its agents from the palace at runtime. Each agent:

- **Has a focus** — what it pays attention to
- **Keeps a diary** — written in AAAK, persists across sessions
- **Builds expertise** — reads its own history to stay sharp in its domain

```
# Agent writes to its diary after a code review
mnemosyne_diary_write("reviewer",
    "PR#42|auth.bypass.found|missing.middleware.check|pattern:3rd.time.this.quarter|★★★★")

# Agent reads back its history
mnemosyne_diary_read("reviewer", last_n=10)
# → last 10 findings, compressed in AAAK
```

Each agent is a specialist lens on your data. The reviewer remembers every bug pattern it's seen. The architect remembers every design decision. The ops agent remembers every incident. They don't share a scratchpad — they each maintain their own memory.

Letta charges $20–200/mo for agent-managed memory. Mnemosyne does it with a wing.

---

## MCP Server

```bash
claude mcp add mnemosyne -- python -m mnemosyne.mcp_server
```

### 22 Tools

**Palace (read)**

| Tool | What |
|------|------|
| `mnemosyne_status` | Palace overview + AAAK spec + memory protocol |
| `mnemosyne_list_wings` | Wings with counts |
| `mnemosyne_list_rooms` | Rooms within a wing |
| `mnemosyne_get_taxonomy` | Full wing → room → count tree |
| `mnemosyne_search` | Semantic search with wing/room filters |
| `mnemosyne_check_duplicate` | Check before filing |
| `mnemosyne_get_aaak_spec` | AAAK dialect reference |

**Palace (write)**

| Tool | What |
|------|------|
| `mnemosyne_add_drawer` | File verbatim content |
| `mnemosyne_delete_drawer` | Remove by ID |

**Knowledge Graph**

| Tool | What |
|------|------|
| `mnemosyne_kg_query` | Entity relationships with time filtering |
| `mnemosyne_kg_add` | Add facts |
| `mnemosyne_kg_invalidate` | Mark facts as ended |
| `mnemosyne_kg_timeline` | Chronological entity story |
| `mnemosyne_kg_stats` | Graph overview |

**Navigation**

| Tool | What |
|------|------|
| `mnemosyne_traverse` | Walk the graph from a room across wings |
| `mnemosyne_find_tunnels` | Find rooms bridging two wings |
| `mnemosyne_graph_stats` | Graph connectivity overview |

**Agent Diary**

| Tool | What |
|------|------|
| `mnemosyne_diary_write` | Write AAAK diary entry |
| `mnemosyne_diary_read` | Read recent diary entries |

**Total Recall (unified search)**

| Tool | What |
|------|------|
| `total_recall_search` | Search across ALL sources — Mnemosyne + Firstbrain + Cricket-Brain. Fused scoring. |
| `total_recall_status` | Show which knowledge sources are connected |
| `total_recall_configure` | Adjust fusion weights (similarity, PageRank, recency) |

The AI learns AAAK and the memory protocol automatically from the `mnemosyne_status` response. No manual configuration.

---

## Auto-Save Hooks

Two hooks for Claude Code that automatically save memories during work:

**Save Hook** — every 15 messages, triggers a structured save. Topics, decisions, quotes, code changes. Also regenerates the critical facts layer.

**PreCompact Hook** — fires before context compression. Emergency save before the window shrinks.

```json
{
  "hooks": {
    "Stop": [{"matcher": "", "hooks": [{"type": "command", "command": "/path/to/mnemosyne/hooks/mempal_save_hook.sh"}]}],
    "PreCompact": [{"matcher": "", "hooks": [{"type": "command", "command": "/path/to/mnemosyne/hooks/mempal_precompact_hook.sh"}]}]
  }
}
```

---

## Total Recall — Unified Knowledge API

Mnemosyne v4 integrates three systems into one unified search. One query, all sources, fused scoring.

```
┌──────────────┐
│ Total Recall │  ← single query
│  Meta-API    │
└──┬───┬───┬───┘
   │   │   │
   ▼   ▼   ▼
┌──────────┐  ┌──────────────┐  ┌─────────────┐
│Firstbrain│  │   Mnemosyne  │  │Cricket-Brain│
│(Obsidian)│  │ (AI Memory)  │  │ (Signals)   │
│ PageRank │  │ ChromaDB+KG  │  │ 97ns/step   │
└──────────┘  └──────────────┘  └─────────────┘
```

**Scoring formula:** `fused = similarity × 0.5 + pagerank × 0.3 + recency × 0.2`

```python
from mempalace.total_recall import TotalRecall

tr = TotalRecall(vault_path="~/obsidian-vault")
results = tr.search("What do we know about auth decisions?")
# → fused results from all connected sources
```

Sources degrade gracefully — if Firstbrain or Cricket-Brain aren't configured, Mnemosyne works alone.

---

## Firstbrain — Knowledge Graph Intelligence

Firstbrain is an Obsidian vault analysis engine with graph algorithms. Zero dependencies.

**Connect your vault:**
```bash
export FIRSTBRAIN_VAULT_PATH=~/my-obsidian-vault
```

**Graph algorithms (pure Python, no networkx):**

| Algorithm | What it reveals |
|-----------|----------------|
| **PageRank** | Most important notes by link authority |
| **Tag Clustering** | Topic clusters via Jaccard similarity |
| **Shortest Path** | BFS path between any two notes |
| **Bridge Detection** | Critical hub notes (Tarjan's algorithm) |
| **Structural Similarity** | Notes with similar link neighborhoods |
| **Multi-hop Discovery** | Hidden connections 2-3 hops away |

```python
from firstbrain.graph import VaultGraph

g = VaultGraph("~/my-obsidian-vault")
g.build()

# Most important notes
g.pagerank(top_n=10)

# Topic clusters
g.tag_clusters()

# How are these connected?
g.shortest_path("Auth Decision", "Security Audit")

# What holds the vault together?
g.bridge_notes()
```

**15 Obsidian Skills** (JavaScript, in `firstbrain/skills/`):

| Skill | What |
|-------|------|
| `/scan` | Incremental vault scanning with SHA-256 change detection |
| `/graph` | PageRank, clusters, bridges, multi-hop analysis |
| `/search` | Semantic search via HuggingFace MiniLM embeddings |
| `/connect` | Connection discovery (tags + links + structural similarity) |
| `/propose` | Emergent structure suggestions (MOC candidates, orphan rescue) |
| `/create` | Template-based note creation (12 types) |
| `/process` | Execute PROMPT:/ACTION: files from Inbox |
| `/briefing` | Daily/weekly briefing from vault state |
| `/triage` | Auto-organize Inbox, auto-tag, archive |
| `/synthesize` | Combine related notes into summaries |
| `/maintain` | Link validation, MOC health, consistency checks |
| `/daily` | Daily note creation |
| `/health` | Vault metrics and stats |
| `/memory` | 4-layer memory system for Claude context persistence |
| `/watch` | File monitoring for actionable notes |

---

## Cricket-Brain — Neuromorphic Signal Engine

A biomorphic AI inference engine in Rust. Uses delay-line coincidence detection for pattern recognition — no matrix multiplication, no CUDA, no weights.

**Performance:** 97 nanoseconds per inference step. 944 bytes RAM on Arduino.

```
Input Signal → [AN1 Resonator] → [LN2/LN3/LN5 Delay Lines] → [ON1 Coincidence Gate] → Output
                   4.5 kHz            2-9ms delays              Temporal bandpass
```

**Core modules (Rust):**

| Module | What |
|--------|------|
| `brain.rs` | 5-neuron cricket circuit with STDP plasticity |
| `resonator_bank.rs` | Parallel multi-frequency token detection |
| `sequence.rs` | N-gram sequence prediction via delay-line memory |
| `token.rs` | Vocabulary → frequency mapping |
| `patterns.rs` | Morse code and signal encoding |

**Language bindings:**

| Binding | Location | Status |
|---------|----------|--------|
| Python (PyO3) | `cricket_brain/crates/python/` | Build with `maturin develop` |
| C/FFI | `cricket_brain/crates/ffi/` | Header in `include/cricket_brain.h` |
| WebAssembly | `cricket_brain/crates/wasm/` | `wasm32-unknown-unknown` target |

**14 examples** including Morse code recognition, ECG monitoring, frequency discrimination, and scale testing up to 40,960 neurons.

---

## Benchmarks

Tested on standard academic benchmarks — reproducible, published datasets.

| Benchmark | Mode | Score | API Calls |
|-----------|------|-------|-----------|
| **LongMemEval R@5** | Raw (ChromaDB only) | **96.6%** | Zero |
| **LongMemEval R@5** | Hybrid + Haiku rerank | **100%** (500/500) | ~500 |
| **LoCoMo R@10** | Raw, session level | **60.3%** | Zero |
| **Personal palace R@10** | Heuristic bench | **85%** | Zero |
| **Palace structure impact** | Wing+room filtering | **+34%** R@10 | Zero |

The 96.6% raw score is the highest published LongMemEval result requiring no API key, no cloud, and no LLM at any stage.

### Unified Benchmark Suite (v4.0)

```bash
python benchmarks/unified_bench.py                    # Run all 5 suites
python benchmarks/unified_bench.py --suite memory     # Memory retrieval only
python benchmarks/unified_bench.py --suite graph      # Graph intelligence only
python benchmarks/unified_bench.py --suite signal     # Cricket-Brain (requires Rust build)
python benchmarks/unified_bench.py --suite fusion     # Total Recall fusion quality
python benchmarks/unified_bench.py --report json      # JSON output
```

| Suite | Key Metrics | Result |
|-------|-------------|--------|
| **Memory Retrieval** | Recall@5, NDCG@10, MRR | R@5 = 1.0 with bootstrap 95% CI |
| **Knowledge Graph** | Insert rate, temporal query latency | 741 triples/s, 0.20ms queries |
| **Graph Intelligence** | PageRank convergence, path finding, clustering | Converges in 4.6ms, 100% path success |
| **Signal Processing** | ns/step latency, spike rate | 97ns/step (requires Rust) |
| **Total Recall Fusion** | Cross-source quality, weight sensitivity | 10.3ms/query across sources |

All benchmarks include bootstrap confidence intervals and are reproducible with `--seed 42`.

### vs Published Systems

| System | LongMemEval R@5 | API Required | Cost |
|--------|----------------|--------------|------|
| **Mnemosyne (hybrid)** | **100%** | Optional | Free |
| Supermemory ASMR | ~99% | Yes | — |
| **Mnemosyne (raw)** | **96.6%** | **None** | **Free** |
| Mastra | 94.87% | Yes (GPT) | API costs |
| Mem0 | ~85% | Yes | $19–249/mo |
| Zep | ~85% | Yes | $25/mo+ |

---

## All Commands

```bash
# Setup
mnemosyne init <dir>                              # guided onboarding + AAAK bootstrap

# Mining
mnemosyne mine <dir>                              # mine project files
mnemosyne mine <dir> --mode convos                # mine conversation exports
mnemosyne mine <dir> --mode convos --wing myapp   # tag with a wing name

# Splitting
mnemosyne split <dir>                             # split concatenated transcripts
mnemosyne split <dir> --dry-run                   # preview

# Search
mnemosyne search "query"                          # search everything
mnemosyne search "query" --wing myapp             # within a wing
mnemosyne search "query" --room auth-migration    # within a room

# Memory stack
mnemosyne wake-up                                 # load L0 + L1 context
mnemosyne wake-up --wing driftwood                # project-specific

# Compression
mnemosyne compress --wing myapp                   # AAAK compress

# Status
mnemosyne status                                  # palace overview
```

All commands accept `--palace <path>` to override the default location.

---

## Configuration

### Global (`~/.mnemosyne/config.json`)

```json
{
  "palace_path": "/custom/path/to/palace",
  "collection_name": "mnemosyne_drawers",
  "people_map": {"Kai": "KAI", "Priya": "PRI"}
}
```

### Wing config (`~/.mnemosyne/wing_config.json`)

Generated by `mnemosyne init`. Maps your people and projects to wings:

```json
{
  "default_wing": "wing_general",
  "wings": {
    "wing_kai": {"type": "person", "keywords": ["kai", "kai's"]},
    "wing_driftwood": {"type": "project", "keywords": ["driftwood", "analytics", "saas"]}
  }
}
```

### Identity (`~/.mnemosyne/identity.txt`)

Plain text. Becomes Layer 0 — loaded every session.

---

## File Reference

**Mnemosyne Core (`mempalace/`)**

| File | What |
|------|------|
| `cli.py` | CLI entry point |
| `config.py` | Configuration loading and defaults |
| `total_recall.py` | Unified search API across all sources |
| `mcp_server.py` | MCP server — 22 tools, AAAK auto-teach, memory protocol |
| `miner.py` | Project file ingest |
| `convo_miner.py` | Conversation ingest — chunks by exchange pair |
| `searcher.py` | Semantic search via ChromaDB |
| `knowledge_graph.py` | Temporal entity-relationship graph (SQLite) |
| `palace_graph.py` | Room-based navigation graph |
| `layers.py` | 4-layer memory stack |
| `dialect.py` | AAAK compression (experimental) |
| `normalize.py` | Converts 5 chat formats to standard transcript |
| `onboarding.py` | Guided setup — generates AAAK bootstrap + wing config |
| `entity_detector.py` | Auto-detect people and projects from content |

**Firstbrain (`firstbrain/`)**

| File | What |
|------|------|
| `graph/engine.py` | PageRank, clustering, path finding, bridge detection (pure Python) |
| `skills/graph/` | Graph analysis skill (JavaScript) |
| `skills/scan/` | Vault scanning with incremental change detection |
| `skills/search/` | Semantic search with HuggingFace embeddings |
| `skills/memory/` | 4-layer memory system for Claude persistence |
| `skills/*/` | 15 total skills — see Firstbrain section above |
| `CLAUDE.md` | AI governance rules and vault conventions |

**Cricket-Brain (`cricket_brain/`)**

| File | What |
|------|------|
| `src/brain.rs` | Main 5-neuron cricket circuit engine |
| `src/resonator_bank.rs` | Parallel multi-frequency token detection |
| `src/sequence.rs` | N-gram sequence prediction via delay-line memory |
| `crates/python/` | PyO3 Python bindings |
| `crates/ffi/` | C/FFI bindings with header file |
| `crates/wasm/` | WebAssembly bindings |
| `RESEARCH_WHITEPAPER.md` | Mathematical derivations and biological model |

---

## Project Structure

```
Mnemosyne/
├── README.md                          ← you are here
├── pyproject.toml                     ← package config (v4.0.0)
│
├── mempalace/                         ← AI Memory Engine (Python)
│   ├── cli.py                         ← CLI entry point
│   ├── mcp_server.py                  ← MCP server (22 tools)
│   ├── total_recall.py                ← Unified search across all sources
│   ├── knowledge_graph.py             ← temporal entity graph (SQLite)
│   ├── palace_graph.py                ← room navigation graph
│   ├── searcher.py                    ← semantic search (ChromaDB)
│   ├── dialect.py                     ← AAAK compression
│   ├── miner.py                       ← project file ingest
│   ├── convo_miner.py                 ← conversation ingest
│   ├── onboarding.py                  ← guided setup
│   └── ...
│
├── firstbrain/                        ← Knowledge Graph Intelligence
│   ├── graph/engine.py                ← PageRank, clustering, bridges, paths
│   ├── skills/                        ← 15 Obsidian skills (JavaScript)
│   │   ├── scan/                      ← vault scanning + indexing
│   │   ├── graph/                     ← graph analysis engine
│   │   ├── search/                    ← semantic + keyword search
│   │   ├── connect/                   ← connection discovery
│   │   ├── process/                   ← PROMPT:/ACTION: execution
│   │   ├── memory/                    ← 4-layer Claude memory
│   │   └── ...                        ← 9 more skills
│   ├── templates/                     ← 12 Obsidian note templates
│   ├── mocs/                          ← Maps of Content
│   └── CLAUDE.md                      ← AI governance rules
│
├── cricket_brain/                     ← Neuromorphic Signal Engine (Rust)
│   ├── src/                           ← core brain logic
│   │   ├── brain.rs                   ← 5-neuron cricket circuit
│   │   ├── resonator_bank.rs          ← multi-frequency detection
│   │   ├── sequence.rs                ← pattern prediction
│   │   └── ...
│   ├── crates/                        ← language bindings
│   │   ├── python/                    ← PyO3 bindings
│   │   ├── ffi/                       ← C/FFI bindings
│   │   └── wasm/                      ← WebAssembly
│   ├── benches/                       ← Criterion performance benchmarks
│   ├── examples/                      ← 14 runnable demos
│   └── RESEARCH_WHITEPAPER.md         ← mathematical foundations
│
├── benchmarks/                        ← benchmark suite
│   ├── unified_bench.py               ← 5-suite unified benchmark
│   ├── longmemeval_bench.py           ← LongMemEval (500 questions)
│   ├── locomo_bench.py                ← LoCoMo (1,986 QA pairs)
│   ├── convomem_bench.py              ← ConvoMem (75K+ QA pairs)
│   ├── membench_bench.py              ← MemBench (ACL 2025)
│   └── BENCHMARKS.md                  ← full results + methodology
│
├── hooks/                             ← Claude Code auto-save hooks
├── examples/                          ← usage examples
├── tests/                             ← 37 tests, all passing
└── assets/                            ← logo + brand assets
```

---

## Requirements

**Core (Python):**
- Python 3.9+
- `chromadb>=0.4.0`
- `pyyaml>=6.0`

**Firstbrain (optional):**
- Node.js (for Obsidian skills)
- `@huggingface/transformers` (for semantic search — keyword search works without it)

**Cricket-Brain (optional):**
- Rust 1.75+ and `maturin` (to build Python bindings)
- Or use pre-built binaries from releases

No API key. No internet after install. Everything local.

```bash
pip install mnemosyne
```

---

## Contributing

PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and guidelines.

## License

MIT — see [LICENSE](LICENSE).

<!-- Link Definitions -->
[version-shield]: https://img.shields.io/badge/version-4.0.0-4dc9f6?style=flat-square&labelColor=0a0e14
[release-link]: https://github.com/beko2210/mnemosyne/releases
[python-shield]: https://img.shields.io/badge/python-3.9+-7dd8f8?style=flat-square&labelColor=0a0e14&logo=python&logoColor=7dd8f8
[python-link]: https://www.python.org/
[license-shield]: https://img.shields.io/badge/license-MIT-b0e8ff?style=flat-square&labelColor=0a0e14
[license-link]: https://github.com/beko2210/mnemosyne/blob/main/LICENSE
[discord-shield]: https://img.shields.io/badge/discord-join-5865F2?style=flat-square&labelColor=0a0e14&logo=discord&logoColor=5865F2
[discord-link]: https://discord.com/invite/ycTQQCu6kn
