#!/usr/bin/env python3
"""
Central Brain - Unified Local Memory Engine for Multi-Platform AI Agents
Integrates Markdown vault, Ollama vector embeddings (mxbai-embed-large),
SQLite FTS5 keyword search, structured facts graph, and MCP tool interface.
"""

import sys
import os
import sqlite3
import json
import hashlib
import time
import math
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime

# Base Directory Setup
BRAIN_DIR = Path.home() / ".central_brain"
KNOWLEDGE_DIR = BRAIN_DIR / "knowledge"
PROJECTS_DIR = BRAIN_DIR / "projects"
EPISODES_DIR = BRAIN_DIR / "episodes"
DB_DIR = BRAIN_DIR / "db"
DB_PATH = DB_DIR / "brain.db"
FACTS_PATH = BRAIN_DIR / "facts.json"

OLLAMA_URL = "http://localhost:11434/api/embeddings"
DEFAULT_EMBED_MODEL = "mxbai-embed-large"

def ensure_dirs():
    for d in [BRAIN_DIR, KNOWLEDGE_DIR, PROJECTS_DIR, EPISODES_DIR, DB_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    if not FACTS_PATH.exists():
        with open(FACTS_PATH, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)

def get_db():
    ensure_dirs()
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=10000;")
    # Initialize tables
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                header TEXT,
                content TEXT NOT NULL,
                embedding BLOB,
                hash TEXT UNIQUE,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # FTS table for keyword search
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                file_path, header, content, content='chunks', content_rowid='id'
            )
        """)
        # Triggers to keep FTS table in sync
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, file_path, header, content)
                VALUES (new.id, new.file_path, new.header, new.content);
            END;
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, file_path, header, content)
                VALUES('delete', old.id, old.file_path, old.header, old.content);
            END;
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, file_path, header, content)
                VALUES('delete', old.id, old.file_path, old.header, old.content);
                INSERT INTO chunks_fts(rowid, file_path, header, content)
                VALUES (new.id, new.file_path, new.header, new.content);
            END;
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity TEXT,
                category TEXT,
                fact TEXT NOT NULL,
                source TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    return conn

def get_embedding(text: str, model: str = DEFAULT_EMBED_MODEL, max_retries: int = 3):
    text_sample = text[:1500] if text else ""
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                OLLAMA_URL,
                data=json.dumps({"model": model, "prompt": text_sample}).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                emb = data.get("embedding")
                if emb:
                    return emb
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.2 * (attempt + 1))
            else:
                print(f"[Brain Warning] Ollama embedding failed ({e}). Falling back to keyword search.", file=sys.stderr)
    return None

def cosine_similarity(v1, v2):
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0

def chunk_markdown(content: str, file_path: str):
    """Splits markdown content into logical header sections or chunks."""
    lines = content.splitlines()
    chunks = []
    current_header = "Header / General"
    current_lines = []

    for line in lines:
        if line.startswith("#"):
            if current_lines:
                chunk_text = "\n".join(current_lines).strip()
                if chunk_text:
                    chunks.append((current_header, chunk_text))
                current_lines = []
            current_header = line.lstrip("#").strip()
        current_lines.append(line)

    if current_lines:
        chunk_text = "\n".join(current_lines).strip()
        if chunk_text:
            chunks.append((current_header, chunk_text))

    return chunks

def ingest_file(file_path: Path):
    file_path = file_path.resolve()
    if not file_path.exists() or file_path.suffix.lower() not in ['.md', '.txt', '.json']:
        return 0

    try:
        content = file_path.read_text(encoding='utf-8', errors='ignore')
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return 0

    conn = get_db()
    str_path = str(file_path)
    file_content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()

    # Incremental check: if file content hasn't changed, skip re-embedding
    existing = conn.execute("SELECT COUNT(*), hash FROM chunks WHERE file_path = ?", (str_path,)).fetchall()
    if existing and existing[0][0] > 0:
        first_hash = existing[0][1] or ""
        if first_hash.startswith(f"{file_content_hash}:"):
            return existing[0][0] # No changes detected, return existing chunk count

    chunks = chunk_markdown(content, str_path)
    ingested_count = 0

    with conn:
        # Atomic replacement: Delete old chunks for this file (FTS triggers auto-delete from FTS index)
        conn.execute("DELETE FROM chunks WHERE file_path = ?", (str_path,))

        for idx, (header, chunk_text) in enumerate(chunks):
            chunk_hash = f"{file_content_hash}:{idx}:{hashlib.sha256(chunk_text.encode('utf-8')).hexdigest()[:16]}"
            vec = get_embedding(f"{header}\n{chunk_text}")
            vec_blob = json.dumps(vec) if vec else None

            conn.execute(
                "INSERT INTO chunks (file_path, header, content, embedding, hash) VALUES (?, ?, ?, ?, ?)",
                (str_path, header, chunk_text, vec_blob, chunk_hash)
            )
            ingested_count += 1

    return ingested_count

def ingest_directory(dir_path: Path):
    total = 0
    dir_path = dir_path.resolve()
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.md') or file.endswith('.txt'):
                fp = Path(root) / file
                total += ingest_file(fp)
    return total

def search_brain(query: str, top_k: int = 5):
    conn = get_db()
    query_vec = get_embedding(query)

    # 1. Vector Search
    vector_results = []
    if query_vec:
        rows = conn.execute("SELECT id, file_path, header, content, embedding FROM chunks WHERE embedding IS NOT NULL").fetchall()
        for r in rows:
            vec = json.loads(r['embedding'])
            sim = cosine_similarity(query_vec, vec)
            vector_results.append((sim, dict(r)))
        vector_results.sort(key=lambda x: x[0], reverse=True)

    # 2. FTS Keyword Search
    fts_results = {}
    try:
        # Sanitize query for FTS
        clean_q = "".join([c if c.isalnum() or c.isspace() else " " for c in query]).strip()
        if clean_q:
            fts_rows = conn.execute("""
                SELECT rowid as id, file_path, header, content, rank
                FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT 10
            """, (clean_q,)).fetchall()
            for rank_idx, r in enumerate(fts_rows):
                # Normalized keyword score
                fts_results[r['id']] = 1.0 / (rank_idx + 1)
    except Exception as e:
        pass

    # 3. Hybrid Reranking
    hybrid_scores = {}
    doc_map = {}

    for sim, doc in vector_results[:20]:
        doc_id = doc['id']
        doc_map[doc_id] = doc
        hybrid_scores[doc_id] = hybrid_scores.get(doc_id, 0.0) + (0.75 * sim)

    for doc_id, kw_score in fts_results.items():
        if doc_id not in doc_map:
            r = conn.execute("SELECT id, file_path, header, content FROM chunks WHERE id = ?", (doc_id,)).fetchone()
            if r:
                doc_map[doc_id] = dict(r)
        hybrid_scores[doc_id] = hybrid_scores.get(doc_id, 0.0) + (0.25 * kw_score)

    sorted_docs = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # 4. Also retrieve matching facts
    facts_rows = conn.execute(
        "SELECT entity, category, fact, source, timestamp FROM facts WHERE fact LIKE ? OR entity LIKE ? ORDER BY id DESC LIMIT 5",
        (f"%{query}%", f"%{query}%")
    ).fetchall()

    return {
        "chunks": [{"score": round(score, 4), **doc_map[doc_id]} for doc_id, score in sorted_docs],
        "facts": [dict(f) for f in facts_rows]
    }

def remember(fact: str, entity: str = "General", category: str = "Knowledge", source: str = "Agent"):
    conn = get_db()
    with conn:
        conn.execute(
            "INSERT INTO facts (entity, category, fact, source) VALUES (?, ?, ?, ?)",
            (entity, category, fact, source)
        )

    # Append to today's episode file
    today_str = datetime.now().strftime("%Y-%m-%d")
    today_file = EPISODES_DIR / f"{today_str}.md"

    mode = "a" if today_file.exists() else "w"
    with open(today_file, mode, encoding="utf-8") as f:
        if mode == "w":
            f.write(f"# Agent Episode Log - {today_str}\n\n")
        time_str = datetime.now().strftime("%H:%M:%S")
        f.write(f"- [{time_str}] **[{category}]** ({entity}): {fact} (via {source})\n")

    # Ingest today's episode file to vector DB
    ingest_file(today_file)
    return True

def forget(target: str, entity: str = None):
    """Deletes matching facts and records an invalidation log in today's episode file."""
    conn = get_db()
    deleted_count = 0
    with conn:
        if entity and entity != "General":
            cur = conn.execute("DELETE FROM facts WHERE entity = ? AND (fact LIKE ? OR ? = '')", (entity, f"%{target}%", target))
            deleted_count = cur.rowcount
        else:
            cur = conn.execute("DELETE FROM facts WHERE fact LIKE ? OR entity LIKE ?", (f"%{target}%", f"%{target}%"))
            deleted_count = cur.rowcount

    # Append invalidation log to today's episode file
    today_str = datetime.now().strftime("%Y-%m-%d")
    today_file = EPISODES_DIR / f"{today_str}.md"
    mode = "a" if today_file.exists() else "w"
    with open(today_file, mode, encoding="utf-8") as f:
        if mode == "w":
            f.write(f"# Agent Episode Log - {today_str}\n\n")
        time_str = datetime.now().strftime("%H:%M:%S")
        target_info = f"'{target}'" if target else f"entity '{entity}'"
        f.write(f"- [{time_str}] **[Invalidated/Forgotten]** ({entity or 'General'}): Purged outdated/incorrect memory matching {target_info}\n")

    ingest_file(today_file)
    return deleted_count

def correct(entity: str, new_fact: str, old_fact_search: str = None, category: str = "Fix", source: str = "Agent"):
    """Corrects/supersedes an existing memory with a new finding."""
    conn = get_db()
    with conn:
        if old_fact_search:
            conn.execute("DELETE FROM facts WHERE entity = ? AND fact LIKE ?", (entity, f"%{old_fact_search}%"))
        else:
            conn.execute("DELETE FROM facts WHERE entity = ? AND category = ?", (entity, category))

        conn.execute(
            "INSERT INTO facts (entity, category, fact, source) VALUES (?, ?, ?, ?)",
            (entity, category, new_fact, source)
        )

    # Append correction log to today's episode file
    today_str = datetime.now().strftime("%Y-%m-%d")
    today_file = EPISODES_DIR / f"{today_str}.md"
    mode = "a" if today_file.exists() else "w"
    with open(today_file, mode, encoding="utf-8") as f:
        if mode == "w":
            f.write(f"# Agent Episode Log - {today_str}\n\n")
        time_str = datetime.now().strftime("%H:%M:%S")
        f.write(f"- [{time_str}] **[Correction]** ({entity}): {new_fact} (Supersedes prior finding, via {source})\n")

    ingest_file(today_file)
    return True

def run_graphify(target_path: Path = None, code_only: bool = True, backend: str = "ollama"):
    """Extracts a deterministic AST code knowledge graph using Graphify and indexes it into the Central Brain."""
    import subprocess
    target = Path(target_path).resolve() if target_path else Path.cwd()
    if not target.exists():
        return False, f"Target path {target} does not exist."

    cmd = ["graphify", "extract", str(target)]
    if code_only:
        cmd.append("--code-only")
    else:
        cmd.extend(["--backend", backend])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        out = proc.stdout + proc.stderr
        graph_out = target / "graphify-out"
        graph_json = graph_out / "graph.json"

        indexed_chunks = 0
        if graph_json.exists():
            indexed_chunks = ingest_file(graph_json)
            report_md = graph_out / "GRAPH_REPORT.md"
            if report_md.exists():
                indexed_chunks += ingest_file(report_md)
        return True, f"Graphify extraction complete. Generated graph.json ({indexed_chunks} chunks indexed into Central Brain).\n{out.strip()}"
    except Exception as e:
        return False, f"Graphify execution failed: {e}"

def get_god_nodes(target_path: Path = None, top_n: int = 10):
    """Retrieves the most connected architectural hub nodes from graph.json."""
    import subprocess
    g_path = None
    if target_path:
        p = Path(target_path).resolve()
        if (p / "graphify-out" / "graph.json").exists():
            g_path = p / "graphify-out" / "graph.json"
        elif p.name == "graph.json" and p.exists():
            g_path = p

    cmd = ["graphify", "god-nodes", "--top", str(top_n)]
    if g_path:
        cmd.extend(["--graph", str(g_path)])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return proc.stdout.strip()
    except Exception as e:
        return f"Error querying god nodes: {e}"

def query_graph_connections(symbol: str, target_path: Path = None):
    """Finds callers, callees, and dependencies for a symbol from graph.json."""
    g_path = None
    if target_path:
        p = Path(target_path).resolve()
        g_path = (p / "graphify-out" / "graph.json") if (p / "graphify-out" / "graph.json").exists() else (p if p.name == "graph.json" else None)

    if not g_path or not g_path.exists():
        matches = list(Path.home().glob("**/graphify-out/graph.json"))
        if matches:
            g_path = matches[0]
        else:
            return f"No graph.json found. Run 'brain graph <project_path>' first."

    try:
        with open(g_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        nodes = {n["id"]: n.get("label", n["id"]) for n in data.get("nodes", [])}
        clean_sym = symbol.strip().lower().replace("()", "")
        callers = []
        callees = []

        links = data.get("links", data.get("edges", []))
        for e in links:
            src_id = str(e.get("source", "")).lower()
            tgt_id = str(e.get("target", "")).lower()
            src_name = nodes.get(e.get("source"), e.get("source"))
            tgt_name = nodes.get(e.get("target"), e.get("target"))
            rel = e.get("relation", "calls")
            loc = f" ({e.get('source_file','')}:{e.get('source_location','')})" if e.get('source_file') else ""

            if clean_sym in tgt_id or clean_sym in str(tgt_name).lower():
                callers.append(f"{src_name} --[{rel}]--> {tgt_name}{loc}")
            elif clean_sym in src_id or clean_sym in str(src_name).lower():
                callees.append(f"{src_name} --[{rel}]--> {tgt_name}{loc}")

        res = [f"📊 GRAPH CONNECTIONS FOR '{symbol}' (from {g_path.parent.parent.name}):"]
        if callers:
            res.append("\nIncoming Calls / Usages (Who calls/uses this):")
            for c in callers[:15]:
                res.append(f"  • {c}")
        if callees:
            res.append("\nOutgoing Calls / Dependencies (What this calls):")
            for c in callees[:15]:
                res.append(f"  • {c}")
        if not callers and not callees:
            res.append(f"No direct edges found for '{symbol}'.")

        return "\n".join(res)
    except Exception as e:
        return f"Error reading graph connections: {e}"

def get_status():
    conn = get_db()
    total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    total_files = conn.execute("SELECT COUNT(DISTINCT file_path) FROM chunks").fetchone()[0]
    total_facts = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]

    # Test Ollama
    ollama_ok = False
    try:
        vec = get_embedding("test")
        if vec:
            ollama_ok = True
    except Exception:
        pass

    return {
        "brain_directory": str(BRAIN_DIR),
        "total_indexed_files": total_files,
        "total_chunks": total_chunks,
        "total_facts": total_facts,
        "ollama_embedding_status": "Connected (mxbai-embed-large)" if ollama_ok else "Unavailable / Fallback to FTS",
        "database_size_bytes": DB_PATH.stat().st_size if DB_PATH.exists() else 0
    }

# MCP Server Implementation for Antigravity & Agent Protocols
def run_mcp_server():
    """Runs a standard Model Context Protocol (MCP) JSON-RPC stdio server."""
    sys.stderr.write("Starting Central Brain MCP Server...\n")
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            req = json.loads(line.strip())
            req_id = req.get("id")
            method = req.get("method")

            if method == "initialize":
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "central-brain-mcp", "version": "1.0.0"}
                    }
                }
            elif method == "tools/list":
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "tools": [
                            {
                                "name": "brain_query",
                                "description": "Search the local Central Brain across all project knowledge, architecture notes, and past agent facts.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "query": {"type": "string", "description": "Search query or natural language question"},
                                        "top_k": {"type": "integer", "default": 5}
                                    },
                                    "required": ["query"]
                                }
                            },
                            {
                                "name": "brain_remember",
                                "description": "Save a new fact, decision, or learned rule to the Central Brain so all agents know it permanently.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "fact": {"type": "string", "description": "Fact or memory to save"},
                                        "entity": {"type": "string", "default": "General"},
                                        "category": {"type": "string", "default": "Knowledge"}
                                    },
                                    "required": ["fact"]
                                }
                            },
                            {
                                "name": "brain_forget",
                                "description": "Remove wrong or outdated facts from the Central Brain.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "target": {"type": "string", "description": "Search term or keyword to delete"},
                                        "entity": {"type": "string", "description": "Optional entity name"}
                                    },
                                    "required": ["target"]
                                }
                            },
                            {
                                "name": "brain_correct",
                                "description": "Correct/supersede an existing memory or fact with a new finding.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "entity": {"type": "string", "description": "Entity or topic to correct"},
                                        "new_fact": {"type": "string", "description": "The new corrected fact or solution"},
                                        "old_fact_search": {"type": "string", "description": "Optional keyword of the old fact to replace"},
                                        "category": {"type": "string", "default": "Fix"}
                                    },
                                    "required": ["entity", "new_fact"]
                                }
                            },
                            {
                                "name": "brain_graph",
                                "description": "Extract a deterministic AST code knowledge graph using Graphify and index it into the Central Brain.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "path": {"type": "string", "description": "Project directory path to graph"},
                                        "code_only": {"type": "boolean", "default": True}
                                    }
                                }
                            },
                            {
                                "name": "brain_god_nodes",
                                "description": "List the most connected architectural hub nodes in a project codebase.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "path": {"type": "string", "description": "Optional project path"},
                                        "top_n": {"type": "integer", "default": 10}
                                    }
                                }
                            },
                            {
                                "name": "brain_status",
                                "description": "Get current status and statistics of the Central Brain.",
                                "inputSchema": {"type": "object", "properties": {}}
                            }
                        ]
                    }
                }
            elif method == "tools/call":
                params = req.get("params", {})
                name = params.get("name")
                args = params.get("arguments", {})

                if name == "brain_query":
                    res = search_brain(args.get("query"), args.get("top_k", 5))
                    resp = {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}}
                elif name == "brain_remember":
                    remember(args.get("fact"), args.get("entity", "General"), args.get("category", "Knowledge"))
                    resp = {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": "Fact saved to Central Brain successfully."}]}}
                elif name == "brain_forget":
                    cnt = forget(args.get("target"), args.get("entity"))
                    resp = {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": f"Purged {cnt} matching facts from Central Brain."}]}}
                elif name == "brain_correct":
                    correct(args.get("entity"), args.get("new_fact"), args.get("old_fact_search"), args.get("category", "Fix"))
                    resp = {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": f"Successfully corrected memory for entity '{args.get('entity')}'."}]}}
                elif name == "brain_graph":
                    ok, msg = run_graphify(args.get("path"), args.get("code_only", True))
                    resp = {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": msg}]}}
                elif name == "brain_god_nodes":
                    res = get_god_nodes(args.get("path"), args.get("top_n", 10))
                    resp = {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": res}]}}
                elif name == "brain_status":
                    res = get_status()
                    resp = {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}}
                else:
                    resp = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}}
            else:
                resp = {"jsonrpc": "2.0", "id": req_id, "result": {}}

            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stderr.write(f"MCP Exception: {e}\n")

def clean_orphans():
    """Finds indexed files that no longer exist on disk and purges their chunks & vectors."""
    conn = get_db()
    rows = conn.execute("SELECT DISTINCT file_path FROM chunks").fetchall()
    deleted_files = 0
    deleted_chunks = 0

    with conn:
        for r in rows:
            fp_str = r['file_path']
            if not Path(fp_str).exists():
                c_cnt = conn.execute("SELECT COUNT(*) FROM chunks WHERE file_path = ?", (fp_str,)).fetchone()[0]
                conn.execute("DELETE FROM chunks WHERE file_path = ?", (fp_str,))
                deleted_files += 1
                deleted_chunks += c_cnt

    return deleted_files, deleted_chunks

def prune_brain():
    """Cleans orphan files, deduplicates facts, and vacuums the SQLite database."""
    conn = get_db()
    orphans_files, orphan_chunks = clean_orphans()

    # Deduplicate facts table
    with conn:
        conn.execute("""
            DELETE FROM facts WHERE id NOT IN (
                SELECT MIN(id) FROM facts GROUP BY entity, category, fact
            )
        """)

    conn.isolation_level = None
    conn.execute("VACUUM;")
    conn.isolation_level = ""

    return orphans_files, orphan_chunks

def sync_brain():
    """Scans all registered directories/files and updates modified content in vector DB."""
    sources_file = BRAIN_DIR / "sources.json"
    default_sources = [
        str(KNOWLEDGE_DIR),
        str(PROJECTS_DIR),
        str(EPISODES_DIR),
        str(Path.home() / "Documents" / "Configs"),
        str(Path.home() / ".agents" / "project_map.md"),
        str(Path.home() / "AGENTS.md")
    ]

    if not sources_file.exists():
        with open(sources_file, "w", encoding="utf-8") as f:
            json.dump(default_sources, f, indent=2)
        sources = default_sources
    else:
        try:
            with open(sources_file, "r", encoding="utf-8") as f:
                sources = json.load(f)
        except Exception:
            sources = default_sources

    total_chunks = 0
    synced_paths = 0
    for src in sources:
        p = Path(src)
        if p.is_file():
            cnt = ingest_file(p)
            total_chunks += cnt
            synced_paths += 1
        elif p.is_dir():
            cnt = ingest_directory(p)
            total_chunks += cnt
            synced_paths += 1

    # Clean up files deleted from disk
    orphan_files, orphan_chunks = clean_orphans()

    return synced_paths, total_chunks, orphan_files, orphan_chunks

def main():
    parser = argparse.ArgumentParser(description="Central Brain - Unified Local Agent Memory CLI")
    sub = parser.add_subparsers(dest="command")

    # query
    q_p = sub.add_parser("query", aliases=["search"], help="Query the central brain")
    q_p.add_argument("text", type=str, help="Search query")
    q_p.add_argument("-k", "--top-k", type=int, default=5, help="Number of results")

    # remember
    r_p = sub.add_parser("remember", help="Save a memory or fact")
    r_p.add_argument("fact", type=str, help="Fact or decision to remember")
    r_p.add_argument("-e", "--entity", type=str, default="General", help="Entity or topic name")
    r_p.add_argument("-c", "--category", type=str, default="Knowledge", help="Category (Knowledge/Fix/Rule/Project)")
    r_p.add_argument("-s", "--source", type=str, default="CLI", help="Source agent/user")

    # forget
    f_p = sub.add_parser("forget", help="Remove wrong or outdated memory from the central brain")
    f_p.add_argument("target", type=str, help="Search term/phrase of the fact to remove")
    f_p.add_argument("-e", "--entity", type=str, default=None, help="Specific entity/topic filter")

    # correct
    c_p = sub.add_parser("correct", help="Correct/supersede a memory with a new finding")
    c_p.add_argument("entity", type=str, help="Entity or topic name")
    c_p.add_argument("new_fact", type=str, help="The new, corrected fact or solution")
    c_p.add_argument("-o", "--old", type=str, default=None, help="Old keyword or fact to replace")
    c_p.add_argument("-c", "--category", type=str, default="Fix", help="Category (Fix/Rule/Knowledge/Project)")
    c_p.add_argument("-s", "--source", type=str, default="CLI", help="Source agent/user")

    # ingest
    i_p = sub.add_parser("ingest", help="Ingest markdown file or directory into vector index")
    i_p.add_argument("path", type=str, help="File or directory path to ingest")

    # graph
    g_p = sub.add_parser("graph", help="Extract AST code knowledge graph using Graphify and index into Central Brain")
    g_p.add_argument("path", nargs="?", default=".", help="Project directory to graph (default: current dir)")
    g_p.add_argument("--deep", action="store_true", help="Run full semantic multimodal extraction via local LLM")

    # god-nodes
    gn_p = sub.add_parser("god-nodes", help="List architectural code hubs and most connected modules")
    gn_p.add_argument("path", nargs="?", default=None, help="Optional project directory or path to graph.json")
    gn_p.add_argument("-n", "--top", type=int, default=10, help="Number of nodes to show (default: 10)")

    # callers
    cl_p = sub.add_parser("callers", help="Find callers, callees, and dependencies for a symbol")
    cl_p.add_argument("symbol", type=str, help="Function, class, or module name to query")
    cl_p.add_argument("path", nargs="?", default=None, help="Optional project directory or graph.json path")

    # sync
    sub.add_parser("sync", help="Sync all registered knowledge bases & files")

    # prune
    sub.add_parser("prune", help="Clean deleted files, deduplicate facts, and reclaim disk space")

    # status
    sub.add_parser("status", help="Display Central Brain metrics")

    # mcp
    sub.add_parser("mcp", help="Run MCP stdio server")

    args = parser.parse_args()
    ensure_dirs()

    if args.command in ["query", "search"]:
        res = search_brain(args.text, args.top_k)
        print(f"\n🧠 CENTRAL BRAIN SEARCH RESULTS for: '{args.text}'\n" + "="*60)

        if res["facts"]:
            print("\n📌 RELEVANT FACTS:")
            for f in res["facts"]:
                print(f"  • [{f['category']}] ({f['entity']}): {f['fact']} ({f['timestamp']})")

        if res["chunks"]:
            print("\n📄 RELEVANT KNOWLEDGE CHUNKS:")
            for idx, c in enumerate(res["chunks"], 1):
                path_name = Path(c['file_path']).name
                print(f"\n--- Result #{idx} [Score: {c['score']}] | File: {path_name} ({c['header']}) ---")
                print(c['content'].strip()[:400] + ("..." if len(c['content']) > 400 else ""))
        else:
            print("\nNo matching chunks found.")
        print()

    elif args.command == "remember":
        remember(args.fact, args.entity, args.category, args.source)
        print(f"✅ Saved memory to Central Brain: [{args.category}] ({args.entity}): {args.fact}")

    elif args.command == "forget":
        cnt = forget(args.target, args.entity)
        print(f"🗑️ Central Brain: Purged {cnt} matching facts matching '{args.target}' (Entity: {args.entity or 'Any'}).")

    elif args.command == "correct":
        correct(args.entity, args.new_fact, args.old, args.category, args.source)
        print(f"✨ Central Brain: Successfully corrected memory for [{args.category}] ({args.entity}) -> {args.new_fact}")

    elif args.command == "graph":
        print(f"🕸️ Extracting code knowledge graph for '{args.path}'...")
        ok, msg = run_graphify(args.path, code_only=not args.deep)
        print(msg)

    elif args.command == "god-nodes":
        res = get_god_nodes(args.path, args.top)
        print(f"\n🏛️ ARCHITECTURAL GOD NODES (Most Connected):\n{res}\n")

    elif args.command == "callers":
        res = query_graph_connections(args.symbol, args.path)
        print(f"\n{res}\n")

    elif args.command == "ingest":
        p = Path(args.path).resolve()
        if p.is_file():
            cnt = ingest_file(p)
            print(f"✅ Ingested file: {p.name} ({cnt} chunks indexed)")
        elif p.is_dir():
            cnt = ingest_directory(p)
            print(f"✅ Ingested directory: {p} ({cnt} total chunks indexed)")
        else:
            print(f"❌ Error: Path '{args.path}' does not exist.")

    elif args.command == "sync":
        paths, chunks, del_files, del_chunks = sync_brain()
        msg = f"🔄 Central Brain Sync Complete: Processed {paths} sources ({chunks} total chunks active)."
        if del_files > 0:
            msg += f" Purged {del_files} deleted files ({del_chunks} orphan chunks removed)."
        print(msg)

    elif args.command == "prune":
        del_files, del_chunks = prune_brain()
        print(f"🧹 Central Brain Prune Complete: Cleaned {del_files} deleted files ({del_chunks} chunks removed). Fact table deduplicated and database vacuumed.")

    elif args.command == "status":
        st = get_status()
        print("\n🧠 CENTRAL BRAIN SYSTEM STATUS")
        print("="*40)
        for k, v in st.items():
            print(f"  • {k.replace('_', ' ').title()}: {v}")
        print()

    elif args.command == "mcp":
        run_mcp_server()

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
