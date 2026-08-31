#!/usr/bin/env python3
"""
Central Brain - Unified Local Memory & Spec-Driven State Engine for Multi-Platform AI Agents
Features:
  - Fast batch embeddings via Ollama /api/embed (with safe truncation & retry)
  - Code-fence-safe hierarchical Markdown chunking with breadcrumbs
  - Recency-weighted hybrid search (Dense Vectors + SQLite FTS5 BM25 + Exponential Decay)
  - Multi-field precision filtering (by entity, category, source, time range, file path)
  - Spec-driven project state tracking (.planning/ & STATE.md)
  - Automated transactional SQLite & vault backups (brain backup / restore)
  - Compiled memory digest export (brain export)
  - Universal JSON output mode (--json) for seamless agent scriptability
  - JSON-RPC 2.0 stdio MCP Server (brain mcp)
"""

import sys
import os
import sqlite3
import json
import hashlib
import time
import math
import struct
import tarfile
import shutil
import argparse
import urllib.request
import urllib.error
import re
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
BACKUP_DIR = BRAIN_DIR / "backups"

OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
DEFAULT_EMBED_MODEL = "mxbai-embed-large"
EMBED_BATCH_SIZE = 32

def ensure_dirs():
    for d in [BRAIN_DIR, KNOWLEDGE_DIR, PROJECTS_DIR, EPISODES_DIR, DB_DIR, BACKUP_DIR]:
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
    conn.execute("PRAGMA synchronous=NORMAL;")
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
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_file_path ON chunks(file_path);")
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                file_path, header, content, content='chunks', content_rowid='id'
            )
        """)
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
        conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_entity_cat ON facts(entity, category, timestamp);")
    return conn

def encode_vector_blob(vec: list[float]) -> bytes:
    """Packs float vector into compact binary IEEE 754 float32 blob."""
    if not vec:
        return b""
    return struct.pack(f"{len(vec)}f", *vec)

def decode_vector_blob(blob: bytes) -> list[float]:
    """Unpacks binary blob into float vector (supports backward-compatibility with JSON strings)."""
    if not blob:
        return []
    if isinstance(blob, str):
        try:
            return json.loads(blob)
        except Exception:
            return []
    if len(blob) % 4 == 0 and not (blob.startswith(b'[') or blob.startswith(b'{')):
        count = len(blob) // 4
        return list(struct.unpack(f"{count}f", blob))
    try:
        return json.loads(blob.decode('utf-8'))
    except Exception:
        return []

def get_embeddings_batch(texts: list[str], model: str = DEFAULT_EMBED_MODEL, max_retries: int = 3) -> list[list[float] | None]:
    """High-performance batch embedding via Ollama /api/embed endpoint with automatic truncation."""
    if not texts:
        return []
    cleaned_texts = [t.strip() if t and t.strip() else " " for t in texts]

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                OLLAMA_EMBED_URL,
                data=json.dumps({"model": model, "input": cleaned_texts, "truncate": True}).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                embeddings = data.get("embeddings")
                if embeddings and len(embeddings) == len(texts):
                    return embeddings
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.2 * (attempt + 1))
            else:
                print(f"[Brain Warning] Ollama batch embedding failed ({e}). Falling back to keyword search.", file=sys.stderr)
    return [None] * len(texts)

def get_embedding(text: str, model: str = DEFAULT_EMBED_MODEL) -> list[float] | None:
    """Fetches single text embedding via batch endpoint."""
    if not text or not text.strip():
        return None
    res = get_embeddings_batch([text], model=model)
    return res[0] if res else None

def cosine_similarity(v1, v2):
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0

def split_oversized_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    paragraphs = text.split("\n\n")
    chunks = []
    current_para = []
    current_len = 0

    for p in paragraphs:
        p_len = len(p)
        if current_len + p_len + 2 > max_chars and current_para:
            chunk_body = "\n\n".join(current_para).strip()
            chunks.append(chunk_body)
            overlap_prefix = chunk_body[-overlap_chars:].strip() if len(chunk_body) > overlap_chars else ""
            current_para = [overlap_prefix, p] if overlap_prefix else [p]
            current_len = sum(len(x) + 2 for x in current_para)
        else:
            current_para.append(p)
            current_len += p_len + 2

    if current_para:
        chunk_body = "\n\n".join(current_para).strip()
        if chunk_body and (not chunks or chunk_body != chunks[-1]):
            chunks.append(chunk_body)

    return chunks if chunks else [text[:max_chars]]

def chunk_markdown(content: str, file_path: str, max_chunk_chars: int = 2400, overlap_chars: int = 200) -> list[tuple[str, str]]:
    """
    Advanced semantic Markdown chunker:
    - Protects code blocks from false '#' header splits.
    - Preserves hierarchical breadcrumbs ('Architecture > Database > WAL').
    - Recursively splits oversized sections at paragraph/list boundaries with overlap.
    - Filters trivial stubs and merges header contexts.
    """
    if not content or not content.strip():
        return []

    lines = content.splitlines()
    header_stack = {}
    sections = []
    current_lines = []
    in_code_block = False
    fence_char = None

    def get_current_breadcrumb():
        if not header_stack:
            return Path(file_path).name if file_path else "General"
        levels = sorted(header_stack.keys())
        return " > ".join(header_stack[lvl] for lvl in levels if header_stack[lvl])

    for line in lines:
        stripped = line.strip()

        # Track fenced code block state
        if stripped.startswith("```") or stripped.startswith("~~~"):
            fence = stripped[:3]
            if not in_code_block:
                in_code_block = True
                fence_char = fence
            elif fence == fence_char:
                in_code_block = False
                fence_char = None
            current_lines.append(line)
            continue

        if in_code_block:
            current_lines.append(line)
            continue

        # Detect markdown heading outside code blocks
        if stripped.startswith("#") and len(stripped) > 1:
            header_level = len(stripped) - len(stripped.lstrip("#"))
            if 1 <= header_level <= 6 and (len(stripped) == header_level or stripped[header_level] == " "):
                non_empty_content = "\n".join(current_lines).strip()
                if non_empty_content:
                    sections.append((get_current_breadcrumb(), non_empty_content))
                current_lines = []

                header_title = stripped[header_level:].strip()
                header_stack = {lvl: txt for lvl, txt in header_stack.items() if lvl < header_level}
                header_stack[header_level] = header_title
                continue

        current_lines.append(line)

    non_empty_content = "\n".join(current_lines).strip()
    if non_empty_content:
        sections.append((get_current_breadcrumb(), non_empty_content))

    final_chunks = []
    for breadcrumb, sec_text in sections:
        if len(sec_text) <= max_chunk_chars:
            if len(sec_text) > 15:
                final_chunks.append((breadcrumb, sec_text))
        else:
            sub_chunks = split_oversized_text(sec_text, max_chunk_chars, overlap_chars)
            for idx, sub_text in enumerate(sub_chunks, 1):
                sub_header = f"{breadcrumb} (Part {idx})" if len(sub_chunks) > 1 else breadcrumb
                final_chunks.append((sub_header, sub_text))

    return final_chunks

def ingest_file(file_path: Path):
    file_path = file_path.resolve()
    if not file_path.exists() or file_path.suffix.lower() not in ['.md', '.txt', '.json', '.conf', '.sh']:
        return 0

    try:
        content = file_path.read_text(encoding='utf-8', errors='ignore')
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return 0

    conn = get_db()
    str_path = str(file_path)
    file_content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()

    existing = conn.execute("SELECT COUNT(*), hash FROM chunks WHERE file_path = ?", (str_path,)).fetchall()
    if existing and existing[0][0] > 0:
        first_hash = existing[0][1] or ""
        if first_hash.startswith(f"{file_content_hash}:"):
            return existing[0][0]

    chunks = chunk_markdown(content, str_path)
    if not chunks:
        return 0

    # Batch embedding calculation
    prepared_inputs = [f"{header}\n{text}" for header, text in chunks]
    all_vectors = []
    for i in range(0, len(prepared_inputs), EMBED_BATCH_SIZE):
        batch = prepared_inputs[i:i + EMBED_BATCH_SIZE]
        vecs = get_embeddings_batch(batch)
        all_vectors.extend(vecs)

    ingested_count = 0
    with conn:
        conn.execute("DELETE FROM chunks WHERE file_path = ?", (str_path,))
        for idx, ((header, chunk_text), vec) in enumerate(zip(chunks, all_vectors)):
            chunk_hash = f"{file_content_hash}:{idx}:{hashlib.sha256(chunk_text.encode('utf-8')).hexdigest()[:16]}"
            vec_blob = encode_vector_blob(vec) if vec else None
            conn.execute(
                "INSERT INTO chunks (file_path, header, content, embedding, hash, updated_at) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
                (str_path, header, chunk_text, vec_blob, chunk_hash)
            )
            ingested_count += 1

    return ingested_count

def ingest_directory(dir_path: Path):
    total = 0
    dir_path = dir_path.resolve()
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith(('.md', '.txt', '.conf', '.sh')):
                fp = Path(root) / file
                total += ingest_file(fp)
    return total

def sync_facts_json():
    """Syncs the SQLite facts table into ~/.central_brain/facts.json for version control tracking."""
    conn = get_db()
    rows = conn.execute("SELECT id, entity, category, fact, source, timestamp FROM facts ORDER BY id ASC").fetchall()
    facts_list = [dict(r) for r in rows]
    with open(FACTS_PATH, "w", encoding="utf-8") as f:
        json.dump(facts_list, f, indent=2)

def remember(fact: str, entity: str = "General", category: str = "Knowledge", source: str = "CLI"):
    """Saves a structured fact to SQLite, syncs facts.json, and appends to today's episode file."""
    conn = get_db()
    with conn:
        conn.execute(
            "INSERT INTO facts (entity, category, fact, source) VALUES (?, ?, ?, ?)",
            (entity, category, fact, source)
        )
    sync_facts_json()

    today_str = datetime.now().strftime("%Y-%m-%d")
    today_file = EPISODES_DIR / f"{today_str}.md"
    mode = "a" if today_file.exists() else "w"
    with open(today_file, mode, encoding="utf-8") as f:
        if mode == "w":
            f.write(f"# Agent Episode Log - {today_str}\n\n")
        time_str = datetime.now().strftime("%H:%M:%S")
        f.write(f"- [{time_str}] **[{category}]** ({entity}): {fact} (via {source})\n")

    ingest_file(today_file)
    return True

def forget(target: str, entity: str = None):
    """Deletes matching facts, syncs facts.json, and records an invalidation log in today's episode file."""
    conn = get_db()
    deleted_count = 0
    with conn:
        if entity and entity != "General":
            cur = conn.execute("DELETE FROM facts WHERE entity = ? AND (fact LIKE ? OR ? = '')", (entity, f"%{target}%", target))
            deleted_count = cur.rowcount
        else:
            cur = conn.execute("DELETE FROM facts WHERE fact LIKE ? OR entity LIKE ?", (f"%{target}%", f"%{target}%"))
            deleted_count = cur.rowcount

    sync_facts_json()

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

def correct(entity: str, new_fact: str, old_fact_search: str = None, category: str = "Fix", source: str = "CLI"):
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

    sync_facts_json()

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

def calculate_recency_and_category_boost(doc: dict) -> float:
    """Calculates temporal recency decay and category multipliers."""
    boost = 1.0
    now = datetime.now()
    fp = doc.get("file_path", "")
    content = doc.get("content", "")
    header = doc.get("header", "")

    doc_date = None
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", fp)
    if date_match:
        try:
            doc_date = datetime.strptime(date_match.group(1), "%Y-%m-%d")
        except Exception:
            pass

    if not doc_date and doc.get("updated_at"):
        try:
            doc_date = datetime.strptime(str(doc["updated_at"]).split(".")[0], "%Y-%m-%d %H:%M:%S")
        except Exception:
            pass

    if doc_date:
        age_days = max(0.0, (now - doc_date).total_seconds() / 86400.0)
        recency_factor = 0.30 * math.exp(- (math.log(2) / 14.0) * age_days)
        boost += recency_factor

    if "[Fix]" in content or "Fix" in header or "Fix" in fp:
        boost *= 1.20
    elif "[Rule]" in content or "project_map" in fp or "STATE.md" in fp:
        boost *= 1.15

    return boost

def search_brain(query: str, top_k: int = 5, entity: str = None, category: str = None,
                 source: str = None, since: str = None, until: str = None, path_filter: str = None):
    """
    Recency-weighted Hybrid Search across Vectors, FTS5 Keywords, and Structured Facts.
    Supports multi-field precision filtering.
    """
    conn = get_db()
    query_vec = get_embedding(query) if query else None

    # 1. Dense Vector Search
    vector_results = []
    if query_vec:
        sql = "SELECT id, file_path, header, content, embedding, updated_at FROM chunks WHERE embedding IS NOT NULL"
        params = []
        if path_filter:
            sql += " AND file_path LIKE ?"
            params.append(f"%{path_filter}%")
        rows = conn.execute(sql, params).fetchall()
        for r in rows:
            vec = decode_vector_blob(r['embedding'])
            sim = cosine_similarity(query_vec, vec)
            vector_results.append((sim, dict(r)))
        vector_results.sort(key=lambda x: x[0], reverse=True)

    # 2. FTS5 Keyword Search
    fts_results = {}
    if query:
        try:
            clean_q = "".join([c if c.isalnum() or c.isspace() else " " for c in query]).strip()
            if clean_q:
                sql = "SELECT rowid as id, file_path, header, content, rank, updated_at FROM chunks_fts WHERE chunks_fts MATCH ?"
                params = [clean_q]
                if path_filter:
                    sql += " AND file_path LIKE ?"
                    params.append(f"%{path_filter}%")
                sql += " ORDER BY rank LIMIT 25"
                fts_rows = conn.execute(sql, params).fetchall()
                for rank_idx, r in enumerate(fts_rows):
                    fts_results[r['id']] = 1.0 / (rank_idx + 1)
        except Exception:
            pass

    # 3. Recency-Weighted Hybrid Scoring
    hybrid_scores = {}
    doc_map = {}

    for sim, doc in vector_results[:40]:
        doc_id = doc['id']
        doc_map[doc_id] = doc
        hybrid_scores[doc_id] = hybrid_scores.get(doc_id, 0.0) + (0.70 * sim)

    for doc_id, kw_score in fts_results.items():
        if doc_id not in doc_map:
            r = conn.execute("SELECT id, file_path, header, content, updated_at FROM chunks WHERE id = ?", (doc_id,)).fetchone()
            if r:
                doc_map[doc_id] = dict(r)
        hybrid_scores[doc_id] = hybrid_scores.get(doc_id, 0.0) + (0.30 * kw_score)

    final_ranked = []
    for doc_id, base_score in hybrid_scores.items():
        if doc_id in doc_map:
            doc = doc_map[doc_id]
            multiplier = calculate_recency_and_category_boost(doc)
            final_score = base_score * multiplier
            final_ranked.append((final_score, doc))

    final_ranked.sort(key=lambda x: x[0], reverse=True)
    top_docs = final_ranked[:top_k]

    # 4. Structured Facts Search with Precision Filtering
    fact_conditions = []
    fact_params = []
    if query:
        fact_conditions.append("(fact LIKE ? OR entity LIKE ?)")
        fact_params.extend([f"%{query}%", f"%{query}%"])
    if entity:
        fact_conditions.append("entity = ? COLLATE NOCASE")
        fact_params.append(entity)
    if category:
        fact_conditions.append("category = ? COLLATE NOCASE")
        fact_params.append(category)
    if source:
        fact_conditions.append("source = ? COLLATE NOCASE")
        fact_params.append(source)
    if since:
        fact_conditions.append("timestamp >= ?")
        fact_params.append(since)
    if until:
        fact_conditions.append("timestamp <= ?")
        fact_params.append(until)

    where_sql = " AND ".join(fact_conditions) if fact_conditions else "1=1"
    facts_rows = conn.execute(
        f"SELECT id, entity, category, fact, source, timestamp FROM facts WHERE {where_sql} ORDER BY id DESC LIMIT ?",
        (*fact_params, top_k)
    ).fetchall()

    return {
        "chunks": [{"score": round(score, 4), **{k: v for k, v in doc.items() if k != 'embedding'}} for score, doc in top_docs],
        "facts": [dict(f) for f in facts_rows]
    }

def init_project(project_name: str, target_dir: Path = None, description: str = "") -> tuple[bool, str]:
    """Scaffolds a clean .planning/ spec-driven structure and registers it with Central Brain."""
    if not target_dir:
        target_dir = Path.cwd()
    else:
        target_dir = Path(target_dir).resolve()

    target_dir.mkdir(parents=True, exist_ok=True)
    planning_dir = target_dir / ".planning"
    planning_dir.mkdir(parents=True, exist_ok=True)
    (planning_dir / "phases").mkdir(parents=True, exist_ok=True)

    today_str = datetime.now().strftime("%Y-%m-%d")

    project_md = planning_dir / "PROJECT.md"
    if not project_md.exists():
        project_md.write_text(f"""# Project: {project_name}

**Created:** {today_str}  
**Description:** {description or 'Add project description here'}

## 🎯 Goals & Scope
- [ ] Goal 1: Core functionality
- [ ] Goal 2: Integration & testing

## 🛠️ Technology Stack
- Language/Framework: 
- Storage/Database: 
- Key Dependencies: 

## 🏗️ Architecture
- Components: 
- Interfaces: 
""", encoding="utf-8")

    roadmap_md = planning_dir / "ROADMAP.md"
    if not roadmap_md.exists():
        roadmap_md.write_text(f"""# Project Roadmap: {project_name}

## 📌 Milestones
- [ ] **Phase 1: Architecture & Foundations** (Current)
- [ ] **Phase 2: Core Feature Implementation**
- [ ] **Phase 3: Verification, Testing & Polish**
""", encoding="utf-8")

    state_md = planning_dir / "STATE.md"
    if not state_md.exists():
        state_md.write_text(f"""# Project State: {project_name}

**Updated:** {today_str}  
**Active Phase:** Phase 1: Architecture & Foundations  
**Status:** In Progress

## 🧭 Recent Decisions
- Initialized spec-driven planning structure (.planning/) on {today_str}.

## 🚧 Blockers / Risks
- None currently identified.

## 📋 Next Actions
1. Define core requirements in .planning/PROJECT.md.
2. Outline detailed implementation steps.
""", encoding="utf-8")

    # Register in sources.json
    sources_file = BRAIN_DIR / "sources.json"
    sources = []
    if sources_file.exists():
        try:
            with open(sources_file, "r", encoding="utf-8") as f:
                sources = json.load(f)
        except Exception:
            sources = []

    str_plan = str(planning_dir)
    if str_plan not in sources:
        sources.append(str_plan)
        with open(sources_file, "w", encoding="utf-8") as f:
            json.dump(sources, f, indent=2)

    ingest_directory(planning_dir)
    remember(f"Initialized project '{project_name}' with .planning/ state tracking in {target_dir}", entity=project_name, category="Project", source="CLI")

    return True, f"Initialized .planning/ structure in {target_dir} and registered with Central Brain."

def get_project_state(target_dir: Path = None) -> dict:
    """Reads and parses .planning/STATE.md or PROJECT.md from a project directory."""
    if not target_dir:
        target_dir = Path.cwd()
    else:
        target_dir = Path(target_dir).resolve()

    planning_dir = None
    curr = target_dir
    while curr != curr.parent:
        if (curr / ".planning").is_dir():
            planning_dir = curr / ".planning"
            break
        curr = curr.parent

    if not planning_dir or not planning_dir.exists():
        map_file = target_dir / ".agents" / "project_map.md"
        if map_file.exists():
            return {
                "project_path": str(target_dir),
                "type": "project_map",
                "content": map_file.read_text(encoding="utf-8", errors="ignore")
            }
        return {"error": f"No .planning/ or .agents/project_map.md found in {target_dir} or parent directories."}

    res = {
        "project_path": str(planning_dir.parent),
        "planning_dir": str(planning_dir),
        "files": [f.name for f in planning_dir.glob("*.md")]
    }

    state_file = planning_dir / "STATE.md"
    if state_file.exists():
        res["state"] = state_file.read_text(encoding="utf-8", errors="ignore")
    project_file = planning_dir / "PROJECT.md"
    if project_file.exists():
        res["project"] = project_file.read_text(encoding="utf-8", errors="ignore")
    roadmap_file = planning_dir / "ROADMAP.md"
    if roadmap_file.exists():
        res["roadmap"] = roadmap_file.read_text(encoding="utf-8", errors="ignore")

    return res

def clean_orphans():
    """Finds indexed files that no longer exist on disk and purges their chunks & FTS entries."""
    conn = get_db()
    indexed_files = [r[0] for r in conn.execute("SELECT DISTINCT file_path FROM chunks").fetchall()]
    orphan_files = 0
    orphan_chunks = 0

    with conn:
        for fp_str in indexed_files:
            p = Path(fp_str)
            if not p.exists():
                count = conn.execute("SELECT COUNT(*) FROM chunks WHERE file_path = ?", (fp_str,)).fetchone()[0]
                conn.execute("DELETE FROM chunks WHERE file_path = ?", (fp_str,))
                orphan_files += 1
                orphan_chunks += count

    return orphan_files, orphan_chunks

def prune_brain():
    """Cleans orphan files, deduplicates facts, and vacuums the SQLite database."""
    orphans_files, orphan_chunks = clean_orphans()
    conn = get_db()

    with conn:
        conn.execute("""
            DELETE FROM facts WHERE id NOT IN (
                SELECT MAX(id) FROM facts GROUP BY entity, category, fact
            )
        """)

    sync_facts_json()

    # Reclaim disk space via VACUUM
    prev_iso = conn.isolation_level
    conn.isolation_level = None
    conn.execute("VACUUM;")
    conn.isolation_level = prev_iso

    return orphans_files, orphan_chunks

def backup_brain(output_path: Path = None, include_vault: bool = True) -> tuple[bool, str, dict]:
    """Creates a transactional SQLite snapshot and packages the Central Brain vault."""
    ensure_dirs()
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not output_path:
        output_path = BACKUP_DIR / f"brain_backup_{timestamp_str}.tar.gz"
    else:
        output_path = Path(output_path).resolve()

    sync_facts_json()
    temp_snapshot_dir = BACKUP_DIR / f"temp_{timestamp_str}"
    temp_snapshot_dir.mkdir(parents=True, exist_ok=True)
    temp_db = temp_snapshot_dir / "brain.db"

    try:
        # 1. Transactional SQLite online backup
        src_conn = get_db()
        dst_conn = sqlite3.connect(temp_db)
        with dst_conn:
            src_conn.backup(dst_conn, pages=250)
        dst_conn.close()

        # 2. Collect vault assets
        st = get_status()
        manifest = {
            "backup_version": "2.0",
            "created_at": datetime.now().isoformat(),
            "metrics": st,
            "included_vault": include_vault
        }
        with open(temp_snapshot_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        if FACTS_PATH.exists():
            shutil.copy2(FACTS_PATH, temp_snapshot_dir / "facts.json")
        sources_file = BRAIN_DIR / "sources.json"
        if sources_file.exists():
            shutil.copy2(sources_file, temp_snapshot_dir / "sources.json")

        if include_vault:
            for vdir in ["knowledge", "projects", "episodes"]:
                src_v = BRAIN_DIR / vdir
                if src_v.exists():
                    shutil.copytree(src_v, temp_snapshot_dir / vdir, dirs_exist_ok=True)

        # 3. Create compressed tarball
        with tarfile.open(output_path, "w:gz") as tar:
            for item in temp_snapshot_dir.iterdir():
                tar.add(item, arcname=item.name)

        return True, f"Backup successfully created at {output_path}", manifest
    except Exception as e:
        return False, f"Backup failed: {e}", {}
    finally:
        shutil.rmtree(temp_snapshot_dir, ignore_errors=True)

def restore_brain(backup_path: Path, force: bool = False) -> tuple[bool, str]:
    """Restores Central Brain database and vault from a backup archive."""
    backup_path = Path(backup_path).resolve()
    if not backup_path.exists():
        return False, f"Backup file {backup_path} does not exist."

    # Create safety backup of current state
    if not force:
        safety_ok, safety_msg, _ = backup_brain(include_vault=True)
        if not safety_ok:
            return False, f"Could not create pre-restore safety snapshot: {safety_msg}"

    extract_tmp = BACKUP_DIR / f"restore_tmp_{int(time.time())}"
    extract_tmp.mkdir(parents=True, exist_ok=True)

    try:
        with tarfile.open(backup_path, "r:gz") as tar:
            tar.extractall(extract_tmp)

        restored_db = extract_tmp / "brain.db"
        if restored_db.exists():
            chk_conn = sqlite3.connect(restored_db)
            integ = chk_conn.execute("PRAGMA integrity_check;").fetchone()[0]
            chk_conn.close()
            if integ != "ok":
                return False, f"Restored database failed integrity check: {integ}"

            shutil.copy2(restored_db, DB_PATH)

        if (extract_tmp / "facts.json").exists():
            shutil.copy2(extract_tmp / "facts.json", FACTS_PATH)
        if (extract_tmp / "sources.json").exists():
            shutil.copy2(extract_tmp / "sources.json", BRAIN_DIR / "sources.json")

        for vdir in ["knowledge", "projects", "episodes"]:
            src_v = extract_tmp / vdir
            if src_v.exists():
                shutil.copytree(src_v, BRAIN_DIR / vdir, dirs_exist_ok=True)

        return True, f"Central Brain restored successfully from {backup_path}"
    except Exception as e:
        return False, f"Restore failed: {e}"
    finally:
        shutil.rmtree(extract_tmp, ignore_errors=True)

def export_brain(output_file: Path = None, fmt: str = "markdown", category: str = None, entity: str = None, days: int = 30) -> str:
    """Compiles permanent rules, structured facts, and recent episodes into a single digest."""
    conn = get_db()
    facts_cond = []
    params = []
    if category:
        facts_cond.append("category = ? COLLATE NOCASE")
        params.append(category)
    if entity:
        facts_cond.append("entity = ? COLLATE NOCASE")
        params.append(entity)

    where_clause = " WHERE " + " AND ".join(facts_cond) if facts_cond else ""
    facts_rows = conn.execute(f"SELECT entity, category, fact, source, timestamp FROM facts{where_clause} ORDER BY category, entity", params).fetchall()

    if fmt == "json":
        data = {
            "generated_at": datetime.now().isoformat(),
            "facts": [dict(r) for r in facts_rows],
            "status": get_status()
        }
        out_str = json.dumps(data, indent=2)
    else:
        lines = [
            "# 🧠 Central Brain — Compiled Knowledge & System Memory Digest",
            f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n",
            "## 📌 Permanent Rules & System Fixes"
        ]

        fixes = [r for r in facts_rows if r['category'] in ['Fix', 'Rule']]
        other_facts = [r for r in facts_rows if r['category'] not in ['Fix', 'Rule']]

        if fixes:
            for f in fixes:
                lines.append(f"- **[{f['category']}]** ({f['entity']}): {f['fact']}")
        else:
            lines.append("*(No permanent rules/fixes registered yet)*")

        lines.append("\n## 📚 Entity Knowledge & Findings")
        if other_facts:
            current_ent = None
            for f in other_facts:
                if f['entity'] != current_ent:
                    current_ent = f['entity']
                    lines.append(f"\n### {current_ent}")
                lines.append(f"- [{f['category']}] {f['fact']} *(via {f['source']}, {f['timestamp'].split()[0]})*")
        else:
            lines.append("*(No additional entity facts registered)*")

        lines.append(f"\n## 🕒 Recent Episode History (Last {days} Days)")
        ep_files = sorted(list(EPISODES_DIR.glob("*.md")), reverse=True)[:days]
        if ep_files:
            for ep in ep_files:
                ep_text = ep.read_text(encoding='utf-8', errors='ignore').strip()
                lines.append(f"\n### Episode: {ep.stem}")
                for ep_line in ep_text.splitlines():
                    if ep_line.startswith("- ["):
                        lines.append(f"  {ep_line}")
        else:
            lines.append("*(No recent episode logs)*")

        out_str = "\n".join(lines) + "\n"

    if output_file:
        output_file = Path(output_file).resolve()
        output_file.write_text(out_str, encoding="utf-8")

    return out_str

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

    # Also auto-discover any .planning folders under ~/Projects
    projects_base = Path.home() / "Projects"
    if projects_base.exists():
        for pl_dir in projects_base.glob("*/.planning"):
            pl_str = str(pl_dir)
            if pl_str not in sources:
                sources.append(pl_str)

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

    orphan_files, orphan_chunks = clean_orphans()
    sync_facts_json()

    return synced_paths, total_chunks, orphan_files, orphan_chunks

def get_status():
    conn = get_db()
    total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    total_files = conn.execute("SELECT COUNT(DISTINCT file_path) FROM chunks").fetchone()[0]
    total_facts = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]

    ollama_ok = False
    try:
        vec = get_embedding("test")
        if vec:
            ollama_ok = True
    except Exception:
        pass

    db_size = DB_PATH.stat().st_size if DB_PATH.exists() else 0
    return {
        "brain_directory": str(BRAIN_DIR),
        "total_indexed_files": total_files,
        "total_chunks": total_chunks,
        "total_facts": total_facts,
        "ollama_embedding_status": f"Connected ({DEFAULT_EMBED_MODEL} via /api/embed)" if ollama_ok else "Unavailable / Fallback to FTS",
        "database_size_bytes": db_size,
        "database_size_mb": round(db_size / (1024 * 1024), 2)
    }

def run_mcp_server():
    """Runs a standard Model Context Protocol (MCP) JSON-RPC stdio server."""
    sys.stderr.write("Starting Central Brain MCP Server (stdio)...\n")
    sys.stderr.flush()

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            req = json.loads(line.strip())
            method = req.get("method")
            req_id = req.get("id")

            if method == "initialize":
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "central-brain", "version": "2.1.0"}
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
                                "description": "Search Central Brain across all projects, configurations, and past learned solutions using recency-weighted hybrid search.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "query": {"type": "string", "description": "Search query or problem description"},
                                        "top_k": {"type": "integer", "default": 5},
                                        "entity": {"type": "string", "description": "Optional entity filter"},
                                        "category": {"type": "string", "description": "Optional category filter (Fix/Rule/Knowledge/Project)"}
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
                                "name": "brain_state",
                                "description": "Get current spec-driven project state, active phase, decisions, and blockers for a project.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "path": {"type": "string", "description": "Optional project path"}
                                    }
                                }
                            },
                            {
                                "name": "brain_init_project",
                                "description": "Scaffold spec-driven .planning/ structure (PROJECT.md, ROADMAP.md, STATE.md) for a project.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string", "description": "Project name"},
                                        "path": {"type": "string", "description": "Optional target directory"},
                                        "description": {"type": "string", "description": "Brief description"}
                                    },
                                    "required": ["name"]
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
                                "name": "brain_export",
                                "description": "Export a compiled markdown or JSON digest of all permanent system rules, entity knowledge, and recent episodes.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "days": {"type": "integer", "default": 30},
                                        "category": {"type": "string", "description": "Optional category filter"}
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
                    res = search_brain(args.get("query"), args.get("top_k", 5), entity=args.get("entity"), category=args.get("category"))
                    resp = {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}}
                elif name == "brain_remember":
                    remember(args.get("fact"), args.get("entity", "General"), args.get("category", "Knowledge"), source="MCP")
                    resp = {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": "Fact saved to Central Brain successfully."}]}}
                elif name == "brain_state":
                    res = get_project_state(args.get("path"))
                    resp = {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}}
                elif name == "brain_init_project":
                    ok, msg = init_project(args.get("name"), args.get("path"), args.get("description", ""))
                    resp = {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": msg}]}}
                elif name == "brain_forget":
                    cnt = forget(args.get("target"), args.get("entity"))
                    resp = {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": f"Purged {cnt} matching facts from Central Brain."}]}}
                elif name == "brain_correct":
                    correct(args.get("entity"), args.get("new_fact"), args.get("old_fact_search"), args.get("category", "Fix"), source="MCP")
                    resp = {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": f"Successfully corrected memory for entity '{args.get('entity')}'."}]}}
                elif name == "brain_export":
                    digest = export_brain(days=args.get("days", 30), category=args.get("category"))
                    resp = {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": digest}]}}
                elif name == "brain_status":
                    res = get_status()
                    resp = {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}}
                else:
                    resp = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Tool '{name}' not found"}}
            else:
                resp = {"jsonrpc": "2.0", "id": req_id, "result": {}}

            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        except Exception as e:
            err_resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32603, "message": str(e)}}
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser(description="Central Brain - Unified Local Agent Memory & State CLI")
    parser.add_argument("--json", action="store_true", help="Output all results in structured JSON format")
    sub = parser.add_subparsers(dest="command")

    # query
    q_p = sub.add_parser("query", aliases=["search"], help="Query the central brain")
    q_p.add_argument("text", type=str, help="Search query")
    q_p.add_argument("-k", "--top-k", type=int, default=5, help="Number of results")
    q_p.add_argument("-e", "--entity", type=str, default=None, help="Filter by entity")
    q_p.add_argument("-c", "--category", type=str, default=None, help="Filter by category")
    q_p.add_argument("-s", "--source", type=str, default=None, help="Filter by source")
    q_p.add_argument("--since", type=str, default=None, help="Filter since date (YYYY-MM-DD)")
    q_p.add_argument("--until", type=str, default=None, help="Filter until date (YYYY-MM-DD)")
    q_p.add_argument("-p", "--path", type=str, default=None, help="Filter by file path pattern")
    q_p.add_argument("--json", action="store_true", help="Output in JSON format")

    # state / plan
    st_cmd = sub.add_parser("state", aliases=["plan"], help="Inspect spec-driven project state (.planning/STATE.md)")
    st_cmd.add_argument("path", nargs="?", default=None, help="Optional project directory")
    st_cmd.add_argument("--json", action="store_true", help="Output in JSON format")

    # init-project
    ip_cmd = sub.add_parser("init-project", help="Scaffold spec-driven .planning/ structure (PROJECT.md, ROADMAP.md, STATE.md)")
    ip_cmd.add_argument("name", type=str, help="Project name")
    ip_cmd.add_argument("path", nargs="?", default=None, help="Target project directory (default: current dir)")
    ip_cmd.add_argument("-d", "--description", type=str, default="", help="Project description")
    ip_cmd.add_argument("--json", action="store_true", help="Output in JSON format")

    # remember
    r_p = sub.add_parser("remember", help="Save a memory or fact")
    r_p.add_argument("fact", type=str, help="Fact or decision to remember")
    r_p.add_argument("-e", "--entity", type=str, default="General", help="Entity or topic name")
    r_p.add_argument("-c", "--category", type=str, default="Knowledge", help="Category (Knowledge/Fix/Rule/Project)")
    r_p.add_argument("-s", "--source", type=str, default="CLI", help="Source agent/user")
    r_p.add_argument("--json", action="store_true", help="Output in JSON format")

    # forget
    f_p = sub.add_parser("forget", help="Remove wrong or outdated memory from the central brain")
    f_p.add_argument("target", type=str, help="Search term/phrase of the fact to remove")
    f_p.add_argument("-e", "--entity", type=str, default=None, help="Specific entity/topic filter")
    f_p.add_argument("--json", action="store_true", help="Output in JSON format")

    # correct
    c_p = sub.add_parser("correct", help="Correct/supersede a memory with a new finding")
    c_p.add_argument("entity", type=str, help="Entity or topic name")
    c_p.add_argument("new_fact", type=str, help="The new, corrected fact or solution")
    c_p.add_argument("-o", "--old", type=str, default=None, help="Old keyword or fact to replace")
    c_p.add_argument("-c", "--category", type=str, default="Fix", help="Category (Fix/Rule/Knowledge/Project)")
    c_p.add_argument("-s", "--source", type=str, default="CLI", help="Source agent/user")
    c_p.add_argument("--json", action="store_true", help="Output in JSON format")

    # ingest
    i_p = sub.add_parser("ingest", help="Ingest markdown file or directory into vector index")
    i_p.add_argument("path", type=str, help="File or directory path to ingest")
    i_p.add_argument("--json", action="store_true", help="Output in JSON format")

    # sync
    s_p = sub.add_parser("sync", help="Sync all registered knowledge bases & files")
    s_p.add_argument("--json", action="store_true", help="Output in JSON format")

    # prune
    p_p = sub.add_parser("prune", help="Clean deleted files, deduplicate facts, and reclaim disk space")
    p_p.add_argument("--json", action="store_true", help="Output in JSON format")

    # backup
    bk_p = sub.add_parser("backup", help="Create a transactional snapshot and backup of Central Brain")
    bk_p.add_argument("output", nargs="?", default=None, help="Destination archive path (.tar.gz)")
    bk_p.add_argument("--no-vault", action="store_true", help="Backup SQLite DB only, omit markdown vaults")
    bk_p.add_argument("--json", action="store_true", help="Output in JSON format")

    # restore
    rst_p = sub.add_parser("restore", help="Restore Central Brain from a backup archive")
    rst_p.add_argument("archive", type=str, help="Backup archive file (.tar.gz)")
    rst_p.add_argument("--force", action="store_true", help="Skip pre-restore safety snapshot")
    rst_p.add_argument("--json", action="store_true", help="Output in JSON format")

    # export
    exp_p = sub.add_parser("export", help="Export compiled memory digest (MEMORY.md or JSON)")
    exp_p.add_argument("output", nargs="?", default=None, help="Output destination file")
    exp_p.add_argument("--format", choices=["markdown", "json"], default="markdown", help="Digest format")
    exp_p.add_argument("-c", "--category", type=str, default=None, help="Filter by category")
    exp_p.add_argument("-e", "--entity", type=str, default=None, help="Filter by entity")
    exp_p.add_argument("-d", "--days", type=int, default=30, help="Days of episode history to include")
    exp_p.add_argument("--json", action="store_true", help="Output in JSON format")

    # status
    st_p = sub.add_parser("status", help="Display Central Brain metrics")
    st_p.add_argument("--json", action="store_true", help="Output in JSON format")

    # mcp
    sub.add_parser("mcp", help="Run MCP stdio server")

    args = parser.parse_args()
    ensure_dirs()
    is_json = getattr(args, "json", False) or parser.get_default("json")

    if args.command in ["query", "search"]:
        res = search_brain(
            args.text, top_k=args.top_k, entity=args.entity, category=args.category,
            source=args.source, since=args.since, until=args.until, path_filter=args.path
        )
        if is_json:
            print(json.dumps({"status": "success", "command": "query", "data": res, "timestamp": datetime.now().isoformat()}, indent=2))
        else:
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

    elif args.command in ["state", "plan"]:
        res = get_project_state(args.path)
        if is_json:
            print(json.dumps({"status": "success" if "error" not in res else "error", "command": "state", "data": res}, indent=2))
        else:
            if "error" in res:
                print(f"❌ {res['error']}")
            elif res.get("type") == "project_map":
                print(f"\n🗺️ PROJECT MAP ({res['project_path']}):\n" + "="*50)
                print(res["content"])
            else:
                print(f"\n🧭 PROJECT STATE ({res['project_path']}):\n" + "="*50)
                if "state" in res:
                    print(res["state"])
                elif "project" in res:
                    print(res["project"])

    elif args.command == "init-project":
        ok, msg = init_project(args.name, args.path, args.description)
        if is_json:
            print(json.dumps({"status": "success" if ok else "error", "command": "init-project", "data": {"message": msg, "project": args.name}}, indent=2))
        else:
            print(f"{'✅' if ok else '❌'} {msg}")

    elif args.command == "remember":
        remember(args.fact, args.entity, args.category, args.source)
        if is_json:
            print(json.dumps({"status": "success", "command": "remember", "data": {"entity": args.entity, "category": args.category, "fact": args.fact, "source": args.source}}, indent=2))
        else:
            print(f"✅ Saved memory to Central Brain: [{args.category}] ({args.entity}): {args.fact}")

    elif args.command == "forget":
        cnt = forget(args.target, args.entity)
        if is_json:
            print(json.dumps({"status": "success", "command": "forget", "data": {"purged_count": cnt, "target": args.target, "entity": args.entity}}, indent=2))
        else:
            print(f"🗑️ Central Brain: Purged {cnt} matching facts matching '{args.target}' (Entity: {args.entity or 'Any'}).")

    elif args.command == "correct":
        correct(args.entity, args.new_fact, args.old, args.category, args.source)
        if is_json:
            print(json.dumps({"status": "success", "command": "correct", "data": {"entity": args.entity, "category": args.category, "new_fact": args.new_fact, "source": args.source}}, indent=2))
        else:
            print(f"✨ Central Brain: Successfully corrected memory for [{args.category}] ({args.entity}) -> {args.new_fact}")

    elif args.command == "ingest":
        p = Path(args.path).resolve()
        cnt = ingest_file(p) if p.is_file() else (ingest_directory(p) if p.is_dir() else 0)
        if is_json:
            print(json.dumps({"status": "success" if cnt > 0 else "error", "command": "ingest", "data": {"path": str(p), "indexed_chunks": cnt}}, indent=2))
        else:
            if p.exists():
                print(f"✅ Ingested: {p} ({cnt} chunks indexed)")
            else:
                print(f"❌ Error: Path '{args.path}' does not exist.")

    elif args.command == "sync":
        paths, chunks, del_files, del_chunks = sync_brain()
        if is_json:
            print(json.dumps({"status": "success", "command": "sync", "data": {"synced_paths": paths, "total_chunks": chunks, "purged_files": del_files, "purged_chunks": del_chunks}}, indent=2))
        else:
            msg = f"🔄 Central Brain Sync Complete: Processed {paths} sources ({chunks} total chunks active)."
            if del_files > 0:
                msg += f" Purged {del_files} deleted files ({del_chunks} orphan chunks removed)."
            print(msg)

    elif args.command == "prune":
        del_files, del_chunks = prune_brain()
        if is_json:
            print(json.dumps({"status": "success", "command": "prune", "data": {"purged_files": del_files, "purged_chunks": del_chunks}}, indent=2))
        else:
            print(f"🧹 Central Brain Prune Complete: Cleaned {del_files} deleted files ({del_chunks} chunks removed). Fact table deduplicated and database vacuumed.")

    elif args.command == "backup":
        ok, msg, manifest = backup_brain(args.output, include_vault=not args.no_vault)
        if is_json:
            print(json.dumps({"status": "success" if ok else "error", "command": "backup", "data": {"message": msg, "manifest": manifest}}, indent=2))
        else:
            print(f"{'📦' if ok else '❌'} {msg}")

    elif args.command == "restore":
        ok, msg = restore_brain(args.archive, force=args.force)
        if is_json:
            print(json.dumps({"status": "success" if ok else "error", "command": "restore", "data": {"message": msg}}, indent=2))
        else:
            print(f"{'✅' if ok else '❌'} {msg}")

    elif args.command == "export":
        res = export_brain(args.output, fmt=args.format, category=args.category, entity=args.entity, days=args.days)
        if is_json:
            print(json.dumps({"status": "success", "command": "export", "data": {"digest": res if not args.output else f"Saved to {args.output}"}}, indent=2))
        else:
            if not args.output:
                print(res)
            else:
                print(f"📄 Central Brain digest exported to: {args.output}")

    elif args.command == "status":
        st = get_status()
        if is_json:
            print(json.dumps({"status": "success", "command": "status", "data": st, "timestamp": datetime.now().isoformat()}, indent=2))
        else:
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
