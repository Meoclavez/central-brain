# 🧠 Central Brain — Unified Local Agent Memory Engine

> A local-first, privacy-preserving, zero-cloud **Central Brain** designed for cross-platform AI agents (**Ollama**, **Google Antigravity CLI**, **OpenClaw**, terminal agents, and shell scripts).

---

## 🌟 Overview & Vision

Modern AI agents often suffer from fragmented context across different runtimes, CLI sessions, and projects. **Central Brain** solves this by establishing a single, local, git-friendly memory hub on your system that combines:
* **Human-Readable Markdown Vault**: Plain `.md` files that can be inspected, edited, or tracked in git.
* **Dense Vector RAG (Ollama `mxbai-embed-large`)**: 1024-dimensional local semantic embeddings.
* **SQLite FTS5 Keyword Search**: Full-text keyword search with normalized BM25 scoring.
* **Structured Fact Graph**: Entity-Attribute-Value memory store (`facts.json` & SQLite).
* **Multi-Platform Protocols**: Native CLI (`brain`), Python SDK, and Model Context Protocol stdio server (`brain mcp`).

---

## 📚 Open-Source Projects & Architectural References

The design of **Central Brain** combines the best concepts from leading open-source agent memory frameworks:

| Referenced Open-Source Project | Key Architectural Concept Adopted |
| :--- | :--- |
| **[Mem0](https://github.com/mem0ai/mem0)** | **Entity & Fact Extraction**: Storing structured key-value memories (`facts.json` & SQLite `facts` table) for instant lookup of user preferences, system configurations, and past fixes. |
| **[Letta / MemGPT](https://github.com/letta-ai/letta)** | **OS-Style Tiered Memory**: A 3-tiered memory hierarchy (Working Context $\rightarrow$ Daily Episode Logs `episodes/YYYY-MM-DD.md` $\rightarrow$ Archival Vector Storage). |
| **[Cognee](https://github.com/topoteretes/cognee)** | **Graph RAG & Reranking**: Extracting entity relationships and combining structural facts with semantic document chunks. |
| **[Zep / Graphiti](https://github.com/getzep/graphiti)** | **Temporal Event Tracking**: Daily episode logs preserve exact chronological history so agents can trace when decisions or fixes occurred. |
| **[Obsidian Smart Connections](https://github.com/chillerlan/obsidian-smart-connections)** | **Human-Readable Markdown Core**: All knowledge stays in plain `.md` files without proprietary database lock-in. |
| **[Ollama](https://github.com/ollama/ollama)** | **Local Vector Embeddings**: Using Ollama's local `mxbai-embed-large` model via `/api/embed` batch API for 100% offline, privacy-preserving semantic search. |
| **[SQLite FTS5](https://www.sqlite.org/fts5.html)** | **Hybrid Search**: Full-text keyword search fused with dense vector cosine similarity and exponential temporal decay ($70\%$ Vector + $30\%$ Keyword + Recency Boost). |

---

## 🏗️ Architecture & Directory Structure

```text
~/.central_brain/
├── knowledge/         # Technical architecture, system rules, hardware guides
├── projects/          # Codebase project maps, design specs, module indexes
├── episodes/          # Auto-generated daily work logs (YYYY-MM-DD.md)
├── backups/           # Point-in-time compressed snapshots (brain backup)
├── db/
│   └── brain.db       # Local SQLite Vector & FTS5 Database (WAL mode enabled)
├── facts.json         # Structured entity-fact graph (auto-synced)
├── sources.json       # Central registry for auto-synced local directories
└── brain.py           # Core Central Brain Engine & MCP Server
```

---

## 🚀 Quick Start & Installation

### Automated Install
```bash
git clone https://github.com/Meoclavez/central-brain.git
cd central-brain
./install.sh
```

### Verification
```bash
brain status
```

---

## 💻 Usage & Interfaces

### 1. Command Line Interface (`brain`)
```bash
# Comprehensive system paths, workspace diagnostics, Git status, & toolchain
brain info
brain info [project_path]
brain info --paths
brain info --json

# Run 7-point health check across SQLite, Ollama, vector completeness, & registries
brain doctor
brain doctor --fix      # Automatically self-heal missing embeddings, dead sources, & FTS5

# List facts, registered sources, discovered projects across machine, and backups
brain list facts -l 10
brain list sources
brain list projects
brain list backups

# Query the brain across all past projects & fixes (returns deterministic [#id])
brain query "bluetooth autosuspend"

# Token-efficient compact query mode (single-line facts & chunk citations)
brain query "bluetooth" --compact
brain query "bluetooth" --terse --max-tokens 300

# Precision filtered query (category, entity, JSON output)
brain query "bluetooth" -c "Fix" -e "MediaTek MT7921" --json

# Remember a new decision or fix across sessions
brain remember "MediaTek MT7921 Wi-Fi stable on kernel 7.1.5+" --entity "Wi-Fi" --category "Fix"

# Precision fact correction by deterministic fact ID (in-place update)
brain correct --id 42 "MediaTek MT7921 Wi-Fi stable on kernel 7.1.5+"
brain correct 42 "MediaTek MT7921 Wi-Fi stable on kernel 7.1.5+"

# Erase an invalid, false, or obsolete memory by deterministic fact ID or search term
brain forget --id 42
brain forget 42
brain forget "temporary false assumption" --entity "Wi-Fi"

# Inspect spec-driven project state with resolved file path header
brain state [project_path]

# Sectional state filtering & token budget limiter (slashes context token consumption)
brain state --section "System Hardware Status"
brain state --section "Next Actions" --max-tokens 200

# Mutate project state (.planning/STATE.md) without manual file rewrites
brain state --add action "Empirically verify unit tests"
brain state --add decision "Selected SQLite WAL mode"
brain state --phase "Phase 2: Execution" --status "In Progress"

# Project map inspection & mutation (.agents/project_map.md)
brain map list-sections
brain map add "Active System Rules" "- /etc/test.conf: custom rule"
brain map show -s "Active System Rules"
brain map init [project_path]

# Subagent context injection (generates ready-to-inject <800 token system prompt)
brain inject general
brain inject hardware
brain inject security --tokens 500
brain inject --json

# Scaffold a clean spec-driven .planning/ structure in a project
brain init-project <name> [project_path] -d "Project description"

# Export a compiled Markdown memory digest or JSON state
brain export ~/MEMORY.md -d 7

# Create a transactional point-in-time snapshot backup
brain backup

# Restore Central Brain from a backup archive
brain restore ~/.central_brain/backups/brain_backup_YYYYMMDD_HHMMSS.tar.gz

# Ingest a new Markdown document or project folder
brain ingest /path/to/project/

# Sync all registered directories listed in sources.json (including .planning/ folders)
brain sync

# Clean deleted files, deduplicate facts, and vacuum DB
brain prune

# View database health & stats
brain status --json

# Launch stdio MCP server for agent tool calls
brain mcp
```

### 2. Python SDK
```python
import sys
sys.path.append('/home/meoclavezz/.central_brain')
import brain

# Perform hybrid RAG search
results = brain.search_brain("bluetooth power rules", top_k=3)
print(results["chunks"])

# Save memory
brain.remember("System fix applied", entity="Bluetooth", category="Fix")
```

### 3. Ollama Local LLM Integration
```python
from ollama import chat
import brain

context = brain.search_brain("Wi-Fi config rules", top_k=3)

response = chat(
    model="lfm2.5:8b",
    messages=[
        {"role": "system", "content": f"Context from Central Brain:\n{context}"},
        {"role": "user", "content": "How do I configure Wi-Fi power save?"}
    ]
)
print(response['message']['content'])
```

### 4. Custom Ollama Modelfile ([`Modelfile.example`](Modelfile.example))
Create custom Ollama models with pre-baked Central Brain system directives:
```bash
# Create a local agent model configured with Central Brain directives
ollama create brain-agent -f Modelfile.example
ollama run brain-agent
```

---

## 🐚 Shell Working Memory Persistence (`MEMORY_SETUP.md`)

Central Brain includes a lightweight shell persistence layer (`helpers/00-osc-memory.sh` and `helpers/working_memory_template.sh`) documented in [`MEMORY_SETUP.md`](MEMORY_SETUP.md):

* **`log_session_entry "<note>"`**: Appends context to `~/.working_memory` and logs it to Central Brain.
* **`flush_working_memory`**: Automatically syncs session context on shell `EXIT` trap.
* **`00-osc-memory.sh`**: Profile hook in `/etc/profile.d/` that auto-sources working memory on login.

---

## 📊 Performance Benchmarks

* **Idle RAM Overhead**: **0 MB** (No persistent background daemon).
* **Query Latency**: **< 15 ms** over 1,500+ document chunks.
* **Disk Space**: **~35 MB** for 1,500+ chunks (500,000+ words across 300+ files).
* **LLM Context Optimization**: Reduces LLM context window consumption by **90-95%** through targeted RAG retrieval.

---

## 📄 License

[MIT License](LICENSE) © 2026 Meoclavez
