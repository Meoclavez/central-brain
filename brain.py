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
    chunks = chunk_markdown(content, str_path)
    ingested_count = 0

    with conn:
        # Clear existing chunks for this file
        conn.execute("DELETE FROM chunks WHERE file_path = ?", (str_path,))

        for header, chunk_text in chunks:
            chunk_hash = hashlib.sha256(f"{str_path}:{header}:{chunk_text}".encode('utf-8')).hexdigest()
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

    return synced_paths, total_chunks

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

    # ingest
    i_p = sub.add_parser("ingest", help="Ingest markdown file or directory into vector index")
    i_p.add_argument("path", type=str, help="File or directory path to ingest")

    # sync
    sub.add_parser("sync", help="Sync all registered knowledge bases & files")

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
        paths, chunks = sync_brain()
        print(f"🔄 Central Brain Sync Complete: Processed {paths} source paths ({chunks} total chunks indexed).")

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
