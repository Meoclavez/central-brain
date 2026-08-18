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
| **[Ollama](https://github.com/ollama/ollama)** | **Local Vector Embeddings**: Using Ollama's local `mxbai-embed-large` model for 100% offline, privacy-preserving semantic search. |
| **[SQLite FTS5](https://www.sqlite.org/fts5.html)** | **Hybrid Search**: Full-text keyword search fused with dense vector cosine similarity ($75\%$ Vector + $25\%$ Keyword). |

---

## 🏗️ Architecture & Directory Structure

```text
~/.central_brain/
├── knowledge/         # Technical architecture, system rules, hardware guides
├── projects/          # Codebase project maps, design specs, module indexes
├── episodes/          # Auto-generated daily work logs (YYYY-MM-DD.md)
├── db/
│   └── brain.db       # Local SQLite Vector & FTS5 Database (WAL mode enabled)
├── facts.json         # Structured entity-fact graph
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
# Query the brain across all past projects & fixes
brain query "how to fix bluetooth on arch"

# Remember a new decision or fix across sessions
brain remember "Realtek Wi-Fi power save set to 2" --entity "Wi-Fi" --category "Fix"

# Correct/supersede an earlier finding with an updated solution
brain correct "Wi-Fi" "Realtek Wi-Fi fix is setting rtw89 aspm disabled" --category "Fix"

# Erase an invalid, false, or obsolete memory completely
brain forget "temporary false assumption" --entity "Wi-Fi"

# Ingest a new Markdown document or project folder
brain ingest /path/to/project/

# Sync all registered directories listed in sources.json
brain sync

# Clean deleted files, deduplicate facts, and vacuum DB
brain prune

# View database health & stats
brain status

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
