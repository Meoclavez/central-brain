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
import subprocess
import platform
import difflib
from pathlib import Path
from datetime import datetime

# Base Directory Setup
BRAIN_DIR = Path(os.getenv("CENTRAL_BRAIN_DIR", os.getenv("BRAIN_DIR", Path.home() / ".central_brain"))).resolve()
KNOWLEDGE_DIR = BRAIN_DIR / "knowledge"
PROJECTS_DIR = BRAIN_DIR / "projects"
EPISODES_DIR = BRAIN_DIR / "episodes"
DB_DIR = BRAIN_DIR / "db"
DB_PATH = DB_DIR / "brain.db"
FACTS_PATH = BRAIN_DIR / "facts.json"
SOURCES_PATH = BRAIN_DIR / "sources.json"
SYSTEM_PROMPT_PATH = BRAIN_DIR / "SYSTEM_PROMPT.md"
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
    if not SOURCES_PATH.exists():
        default_sources = [
            str(KNOWLEDGE_DIR),
            str(PROJECTS_DIR),
            str(EPISODES_DIR),
            str(Path.home() / "Documents" / "Configs"),
            str(Path.home() / ".agents" / "project_map.md"),
            str(Path.home() / "AGENTS.md")
        ]
        with open(SOURCES_PATH, "w", encoding="utf-8") as f:
            json.dump(default_sources, f, indent=2)

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

        # Self-healing rehydration: If facts table is empty but facts.json has entries, restore them!
        try:
            cur = conn.execute("SELECT COUNT(*) FROM facts")
            row = cur.fetchone()
            if row and row[0] == 0 and FACTS_PATH.exists() and FACTS_PATH.stat().st_size > 10:
                with open(FACTS_PATH, "r", encoding="utf-8") as f:
                    cached_facts = json.load(f)
                if cached_facts and isinstance(cached_facts, list):
                    for item in cached_facts:
                        conn.execute(
                            "INSERT OR IGNORE INTO facts (id, entity, category, fact, source, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                            (item.get("id"), item.get("entity", "General"), item.get("category", "Knowledge"), item.get("fact", ""), item.get("source", "Restored"), item.get("timestamp", datetime.now().isoformat()))
                        )
        except Exception:
            pass
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
            null_embeds = conn.execute("SELECT COUNT(*) FROM chunks WHERE file_path = ? AND embedding IS NULL", (str_path,)).fetchone()[0]
            if null_embeds == 0:
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

def backfill_missing_embeddings(batch_size: int = EMBED_BATCH_SIZE) -> int:
    """Finds chunks in SQLite with NULL embeddings and backfills them via Ollama."""
    conn = get_db()
    rows = conn.execute("SELECT id, header, content FROM chunks WHERE embedding IS NULL").fetchall()
    if not rows:
        return 0

    # Verify Ollama is reachable before attempting
    test_vec = get_embedding("test")
    if not test_vec:
        return 0

    total_backfilled = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        prepared_inputs = [f"{r['header'] or ''}\n{r['content']}" for r in batch]
        vecs = get_embeddings_batch(prepared_inputs)
        with conn:
            for r, vec in zip(batch, vecs):
                if vec:
                    conn.execute("UPDATE chunks SET embedding = ? WHERE id = ?", (encode_vector_blob(vec), r["id"]))
                    total_backfilled += 1
    return total_backfilled

def sync_facts_json():
    """Syncs the SQLite facts table into ~/.central_brain/facts.json for version control tracking."""
    conn = get_db()
    rows = conn.execute("SELECT id, entity, category, fact, source, timestamp FROM facts ORDER BY id ASC").fetchall()
    # Safeguard against accidental data loss:
    # If SQLite has 0 rows, but facts.json already has >0 facts, do NOT overwrite with an empty list!
    if not rows and FACTS_PATH.exists() and FACTS_PATH.stat().st_size > 10:
        try:
            with open(FACTS_PATH, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached and len(cached) > 0:
                return
        except Exception:
            pass
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

def forget(target: str = None, entity: str = None, fact_id: int = None):
    """Deletes matching facts or a specific fact by ID, syncs facts.json, and records an invalidation log."""
    conn = get_db()
    deleted_count = 0
    purged_info = ""

    # Auto-detect numeric ID in target (e.g. '42' or '#42')
    if fact_id is None and target:
        m = re.match(r"^#?(\d+)$", str(target).strip())
        if m:
            fact_id = int(m.group(1))

    with conn:
        if fact_id is not None:
            row = conn.execute("SELECT id, entity, category, fact FROM facts WHERE id = ?", (fact_id,)).fetchone()
            if row:
                conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
                deleted_count = 1
                purged_info = f"fact #{fact_id} [{row['category']}] ({row['entity']}): '{row['fact']}'"
            else:
                return 0
        elif entity and entity != "General":
            cur = conn.execute("DELETE FROM facts WHERE entity = ? AND (fact LIKE ? OR ? = '')", (entity, f"%{target}%", target or ""))
            deleted_count = cur.rowcount
            purged_info = f"facts under entity '{entity}' matching '{target}'"
        else:
            cur = conn.execute("DELETE FROM facts WHERE fact LIKE ? OR entity LIKE ?", (f"%{target}%", f"%{target}%"))
            deleted_count = cur.rowcount
            purged_info = f"facts matching '{target}'"

    sync_facts_json()

    today_str = datetime.now().strftime("%Y-%m-%d")
    today_file = EPISODES_DIR / f"{today_str}.md"
    mode = "a" if today_file.exists() else "w"
    with open(today_file, mode, encoding="utf-8") as f:
        if mode == "w":
            f.write(f"# Agent Episode Log - {today_str}\n\n")
        time_str = datetime.now().strftime("%H:%M:%S")
        f.write(f"- [{time_str}] **[Invalidated/Forgotten]** Purged {purged_info}\n")

    ingest_file(today_file)
    return deleted_count

def correct(entity: str = None, new_fact: str = None, old_fact_search: str = None, category: str = "Fix", source: str = "CLI", fact_id: int = None):
    """Corrects/supersedes an existing memory with a new finding by ID or entity search."""
    conn = get_db()

    # Auto-detect numeric ID in entity (e.g. `brain correct 42 "new fact"`)
    if fact_id is None and entity:
        m = re.match(r"^#?(\d+)$", str(entity).strip())
        if m:
            fact_id = int(m.group(1))

    old_fact = None
    use_entity = entity
    use_cat = category

    with conn:
        if fact_id is not None:
            row = conn.execute("SELECT id, entity, category, fact FROM facts WHERE id = ?", (fact_id,)).fetchone()
            if not row:
                return False
            old_fact = row["fact"]
            use_entity = entity if (entity and not re.match(r"^#?(\d+)$", str(entity).strip())) else row["entity"]
            use_cat = category if category else row["category"]
            conn.execute(
                "UPDATE facts SET entity = ?, category = ?, fact = ?, source = ?, timestamp = CURRENT_TIMESTAMP WHERE id = ?",
                (use_entity, use_cat, new_fact, source, fact_id)
            )
        elif old_fact_search:
            conn.execute("DELETE FROM facts WHERE entity = ? AND fact LIKE ?", (entity, f"%{old_fact_search}%"))
            conn.execute(
                "INSERT INTO facts (entity, category, fact, source) VALUES (?, ?, ?, ?)",
                (entity, category, new_fact, source)
            )
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
        if fact_id is not None:
            f.write(f"- [{time_str}] **[Correction]** ({use_entity}): Corrected fact #{fact_id}: '{new_fact}' (Supersedes: '{old_fact}')\n")
        else:
            old_info = f" (Supersedes: '{old_fact_search}')" if old_fact_search else ""
            f.write(f"- [{time_str}] **[Correction]** ({entity}): {new_fact}{old_info}\n")

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
    sources = []
    if SOURCES_PATH.exists():
        try:
            with open(SOURCES_PATH, "r", encoding="utf-8") as f:
                sources = json.load(f)
        except Exception:
            sources = []

    str_plan = str(planning_dir)
    if str_plan not in sources:
        sources.append(str_plan)
        with open(SOURCES_PATH, "w", encoding="utf-8") as f:
            json.dump(sources, f, indent=2)

    ingest_directory(planning_dir)
    remember(f"Initialized project '{project_name}' with .planning/ state tracking in {target_dir}", entity=project_name, category="Project", source="CLI")

    return True, f"Initialized .planning/ structure in {target_dir} and registered with Central Brain."

def find_project_root(start_dir: Path) -> Path:
    """Finds the root of the project by ascending until a project marker (.git, package.json, etc.) or home is reached."""
    curr = start_dir.resolve()
    if curr == Path.home() or curr == curr.parent:
        return curr

    project_markers = [".git", ".hg", ".svn", "package.json", "Cargo.toml", "pyproject.toml", "go.mod", "CMakeLists.txt"]

    while curr != curr.parent and curr != Path.home():
        if any((curr / marker).exists() for marker in project_markers):
            return curr
        curr = curr.parent

    return start_dir.resolve()

def extract_project_doc_summary(content: str, max_overview_lines: int = 12) -> tuple[str, list[str]]:
    """Extracts a clean, token-efficient summary from a project README or documentation file.
    Returns (summary_text, all_section_titles)."""
    clean = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL)
    clean = re.sub(r"<[^>]+>", "", clean)

    _, all_secs = extract_markdown_section(content, "__all__")

    clean_lines = [l.strip() for l in clean.splitlines() if l.strip()]
    overview_lines = []
    for l in clean_lines:
        if l.startswith("## ") and len(overview_lines) > 2:
            break
        if not l.startswith("---") and not l.startswith("!["):
            overview_lines.append(l)
        if len(overview_lines) >= max_overview_lines:
            break

    overview_text = "\n".join(overview_lines)

    tech_stack, _ = extract_markdown_section(content, "Tech Stack")
    if not tech_stack:
        tech_stack, _ = extract_markdown_section(content, "Stack")

    features, _ = extract_markdown_section(content, "Features")

    summary_parts = [overview_text]
    if tech_stack:
        summary_parts.append(tech_stack)
    elif features:
        feat_lines = features.splitlines()[:12]
        summary_parts.append("\n".join(feat_lines))

    return "\n\n".join(summary_parts).strip(), all_secs

def get_project_state(target_dir: Path = None) -> dict:
    """Reads and parses .planning/STATE.md, PROJECT.md, or .agents/project_map.md with project boundary isolation."""
    if not target_dir:
        target_dir = Path.cwd()
    else:
        target_dir = Path(target_dir).resolve()

    project_root = find_project_root(target_dir)
    is_home_root = (project_root == Path.home())

    # 1. Search upwards for .planning directory up to project_root
    planning_dir = None
    curr = target_dir
    while True:
        if (curr / ".planning").is_dir():
            planning_dir = curr / ".planning"
            break
        if curr == project_root or curr == curr.parent or curr == Path.home():
            break
        curr = curr.parent

    if planning_dir and planning_dir.exists():
        state_file = planning_dir / "STATE.md"
        project_file = planning_dir / "PROJECT.md"
        roadmap_file = planning_dir / "ROADMAP.md"

        resolved_file = state_file if state_file.exists() else (project_file if project_file.exists() else roadmap_file)
        file_type = "planning_state" if (state_file.exists()) else "planning_project"

        res = {
            "project_path": str(planning_dir.parent),
            "planning_dir": str(planning_dir),
            "resolved_file": str(resolved_file) if resolved_file and resolved_file.exists() else None,
            "file_type": file_type,
            "type": "planning",
            "files": [f.name for f in planning_dir.glob("*.md")]
        }
        if state_file.exists():
            res["state"] = state_file.read_text(encoding="utf-8", errors="ignore")
            res["content"] = res["state"]
        if project_file.exists():
            res["project"] = project_file.read_text(encoding="utf-8", errors="ignore")
            if "content" not in res:
                res["content"] = res["project"]
        if roadmap_file.exists():
            res["roadmap"] = roadmap_file.read_text(encoding="utf-8", errors="ignore")

        return res

    # 2. Search upwards for .agents/project_map.md up to project_root
    curr = target_dir
    map_file = None
    while True:
        candidate = curr / ".agents" / "project_map.md"
        if candidate.is_file():
            map_file = candidate
            break
        if curr == project_root or curr == curr.parent or curr == Path.home():
            break
        curr = curr.parent

    # 3. Only fall back to ~/.agents/project_map.md if we are actually targeting home
    if not map_file and is_home_root:
        home_map = Path.home() / ".agents" / "project_map.md"
        if home_map.is_file():
            map_file = home_map

    if map_file and map_file.exists():
        content = map_file.read_text(encoding="utf-8", errors="ignore")
        return {
            "project_path": str(map_file.parent.parent if map_file.parent.name == ".agents" else map_file.parent),
            "resolved_file": str(map_file),
            "file_type": "project_map",
            "type": "project_map",
            "content": content
        }

    # 4. Fallback for unmapped projects: check for project documentation (README.md, AGENTS.md, etc.)
    doc_candidates = [
        ("README.md", "readme"),
        ("readme.md", "readme"),
        ("AGENTS.md", "agents_rules"),
        ("CLAUDE.md", "agent_rules")
    ]
    resolved_doc = None
    doc_type = None
    for filename, dtype in doc_candidates:
        cand = project_root / filename
        if cand.is_file():
            resolved_doc = cand
            doc_type = dtype
            break
        if target_dir != project_root:
            cand_sub = target_dir / filename
            if cand_sub.is_file():
                resolved_doc = cand_sub
                doc_type = dtype
                break

    # Count indexed chunks in Central Brain DB for this project
    conn = get_db()
    indexed_count = 0
    try:
        row = conn.execute("SELECT COUNT(*) FROM chunks WHERE file_path LIKE ?", (f"{project_root}%",)).fetchone()
        if row:
            indexed_count = row[0]
    except Exception:
        pass

    if resolved_doc and resolved_doc.exists():
        doc_content = resolved_doc.read_text(encoding="utf-8", errors="ignore")
        return {
            "project_path": str(project_root),
            "resolved_file": str(resolved_doc),
            "file_type": doc_type,
            "type": "project_doc",
            "content": doc_content,
            "indexed_chunks": indexed_count,
            "is_fallback": True
        }

    return {
        "error": f"No .planning/, .agents/project_map.md, or project documentation found in {target_dir} or project root ({project_root}).",
        "project_path": str(project_root),
        "indexed_chunks": indexed_count,
        "suggestion": f"Run 'brain map init {project_root}' or 'brain init-project \"{project_root.name}\"' to initialize state tracking."
    }

def extract_markdown_section(content: str, target_section: str) -> tuple[str | None, list[str]]:
    """Extracts a specific markdown section (by title substring match) from a markdown document.
    Returns (section_content, list_of_all_section_titles)."""
    lines = content.splitlines()
    in_code_block = False
    sections = []

    current_sec = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if not in_code_block and stripped.startswith("#"):
            m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
            if m:
                level = len(m.group(1))
                title = m.group(2).strip()
                if current_sec:
                    current_sec["end"] = idx
                    sections.append(current_sec)
                current_sec = {"level": level, "title": title, "start": idx, "end": len(lines)}

    if current_sec:
        current_sec["end"] = len(lines)
        sections.append(current_sec)

    all_titles = [s["title"] for s in sections]
    target_clean = re.sub(r"[^\w\s]", "", target_section).strip().lower()

    matched_sec = None
    for s in sections:
        sec_clean = re.sub(r"[^\w\s]", "", s["title"]).strip().lower()
        if target_clean in sec_clean or sec_clean in target_clean:
            matched_sec = s
            break

    if not matched_sec:
        return None, all_titles

    start_idx = matched_sec["start"]
    end_idx = len(lines)
    for s in sections:
        if s["start"] > start_idx and s["level"] <= matched_sec["level"]:
            end_idx = s["start"]
            break

    section_lines = lines[start_idx:end_idx]
    return "\n".join(section_lines).strip(), all_titles

def apply_token_budget(text: str, max_tokens: int = None) -> str:
    """Clamps output text to an approximate token budget (1 token ≈ 4 characters)."""
    if not max_tokens or max_tokens <= 0:
        return text
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text

    slice_point = text.rfind("\n", 0, max_chars)
    if slice_point == -1 or slice_point < max_chars // 2:
        slice_point = max_chars

    return text[:slice_point] + f"\n\n[... output truncated: reached limit of --max-tokens {max_tokens} ...]"

def atomic_write_file(path: Path, content: str):
    """Atomically writes content to a file, creating a .bak backup."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        bak_file = path.with_suffix(path.suffix + ".bak")
        try:
            shutil.copy2(path, bak_file)
        except Exception:
            pass

    tmp_file = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        with open(tmp_file, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_file, path)
    finally:
        if tmp_file.exists():
            try:
                tmp_file.unlink()
            except Exception:
                pass

def map_add_entry(target_file: Path, section_name: str, entry: str) -> tuple[bool, str]:
    """Appends an entry/bullet into a specific section of a markdown map or state file."""
    target_file = Path(target_file).resolve()
    if not target_file.exists():
        return False, f"File {target_file} does not exist."

    content = target_file.read_text(encoding="utf-8", errors="ignore")
    lines = content.splitlines()

    in_code_block = False
    sections = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if not in_code_block and stripped.startswith("#"):
            m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
            if m:
                sections.append({"level": len(m.group(1)), "title": m.group(2).strip(), "idx": idx})

    target_clean = re.sub(r"[^\w\s]", "", section_name).strip().lower()
    matched = None
    for s in sections:
        sc = re.sub(r"[^\w\s]", "", s["title"]).strip().lower()
        if target_clean in sc or sc in target_clean:
            matched = s
            break

    if not matched:
        available = [s["title"] for s in sections]
        return False, f"Section '{section_name}' not found in {target_file}. Available: {', '.join(available)}"

    insert_idx = len(lines)
    for s in sections:
        if s["idx"] > matched["idx"] and s["level"] <= matched["level"]:
            insert_idx = s["idx"]
            break

    clean_entry = entry.strip()
    if not clean_entry.startswith("-") and not clean_entry.startswith("*") and not clean_entry.startswith("1."):
        clean_entry = f"- {clean_entry}"

    lines.insert(insert_idx, clean_entry)
    new_content = "\n".join(lines) + "\n"
    atomic_write_file(target_file, new_content)
    ingest_file(target_file)
    return True, f"Added entry to '{matched['title']}' in {target_file}"

def map_set_section(target_file: Path, section_name: str, new_body: str) -> tuple[bool, str]:
    """Replaces the body of a markdown section with new content."""
    target_file = Path(target_file).resolve()
    if not target_file.exists():
        return False, f"File {target_file} does not exist."

    content = target_file.read_text(encoding="utf-8", errors="ignore")
    lines = content.splitlines()

    in_code_block = False
    sections = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if not in_code_block and stripped.startswith("#"):
            m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
            if m:
                sections.append({"level": len(m.group(1)), "title": m.group(2).strip(), "idx": idx})

    target_clean = re.sub(r"[^\w\s]", "", section_name).strip().lower()
    matched = None
    for s in sections:
        sc = re.sub(r"[^\w\s]", "", s["title"]).strip().lower()
        if target_clean in sc or sc in target_clean:
            matched = s
            break

    if not matched:
        available = [s["title"] for s in sections]
        return False, f"Section '{section_name}' not found in {target_file}. Available: {', '.join(available)}"

    end_idx = len(lines)
    for s in sections:
        if s["idx"] > matched["idx"] and s["level"] <= matched["level"]:
            end_idx = s["idx"]
            break

    new_lines = lines[:matched["idx"] + 1] + [""] + new_body.strip().splitlines() + [""] + lines[end_idx:]
    new_content = "\n".join(new_lines) + "\n"
    atomic_write_file(target_file, new_content)
    ingest_file(target_file)
    return True, f"Updated section '{matched['title']}' in {target_file}"

def init_project_map(target_dir: Path = None) -> tuple[bool, str]:
    """Scaffolds a new .agents/project_map.md in the project directory."""
    if not target_dir:
        target_dir = Path.cwd()
    else:
        target_dir = Path(target_dir).resolve()

    agents_dir = target_dir / ".agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    map_file = agents_dir / "project_map.md"

    if map_file.exists():
        return False, f"Project map already exists at {map_file}"

    today_str = datetime.now().strftime("%Y-%m-%d")
    project_name = target_dir.name
    content = f"""# Project Architecture & Component Map: {project_name}

**Created:** {today_str}  
**Root Directory:** `{target_dir}`

## 🧭 System Overview
- Brief high-level summary of {project_name} modules and goals.

## 📦 Active Components & Services
- Component 1: Initial core service.

## ⚙️ Configuration & Key Paths
- Key configuration files and their roles.

## 📝 Recent Architectural Decisions
- Initial project map established on {today_str}.
"""
    map_file.write_text(content, encoding="utf-8")
    ingest_file(map_file)
    remember(f"Scaffolded .agents/project_map.md in {target_dir}", entity=project_name, category="Project", source="CLI")
    return True, f"Initialized .agents/project_map.md in {target_dir}"

def state_update(target_dir: Path = None, phase: str = None, status: str = None) -> tuple[bool, str]:
    """Updates active phase and/or status in .planning/STATE.md, auto-updating the timestamp."""
    st = get_project_state(target_dir)
    if "error" in st:
        return False, st["error"]
    resolved = st.get("resolved_file")
    if not resolved or not resolved.endswith("STATE.md"):
        return False, f"State file not found or target is not STATE.md ({resolved})"

    state_path = Path(resolved)
    content = state_path.read_text(encoding="utf-8", errors="ignore")
    today_str = datetime.now().strftime("%Y-%m-%d")

    content = re.sub(r"\*\*Updated:\*\*.*", f"**Updated:** {today_str}", content)
    if phase:
        content = re.sub(r"\*\*Active Phase:\*\*.*", f"**Active Phase:** {phase}", content)
    if status:
        content = re.sub(r"\*\*Status:\*\*.*", f"**Status:** {status}", content)

    atomic_write_file(state_path, content)
    ingest_file(state_path)
    return True, f"Updated state in {state_path} (Phase: {phase or 'Unchanged'}, Status: {status or 'Unchanged'})"

def state_add_entry(target_dir: Path = None, entry_type: str = "action", text: str = "") -> tuple[bool, str]:
    """Adds a decision, blocker, or action to .planning/STATE.md."""
    st = get_project_state(target_dir)
    if "error" in st:
        return False, st["error"]
    resolved = st.get("resolved_file")
    if not resolved or not resolved.endswith("STATE.md"):
        return False, f"State file not found or target is not STATE.md ({resolved})"

    state_path = Path(resolved)
    type_to_section = {
        "action": "Next Actions",
        "decision": "Recent Decisions",
        "blocker": "Blockers / Risks"
    }
    sec_name = type_to_section.get(entry_type.lower(), entry_type)
    ok, msg = map_add_entry(state_path, sec_name, text)
    if ok:
        today_str = datetime.now().strftime("%Y-%m-%d")
        c = state_path.read_text(encoding="utf-8", errors="ignore")
        c = re.sub(r"\*\*Updated:\*\*.*", f"**Updated:** {today_str}", c)
        atomic_write_file(state_path, c)
        ingest_file(state_path)
    return ok, msg

def get_git_info(directory: Path) -> dict:
    """Safely retrieves git repository status for a directory without external dependencies."""
    if not directory or not directory.exists():
        return {"is_git_repo": False}
    try:
        is_git = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "--is-inside-work-tree"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=2
        )
        if is_git.returncode != 0 or is_git.stdout.strip() != "true":
            return {"is_git_repo": False}

        root_proc = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "--show-toplevel"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=2
        )
        git_root = root_proc.stdout.strip() if root_proc.returncode == 0 else str(directory)

        branch_proc = subprocess.run(
            ["git", "-C", str(directory), "branch", "--show-current"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=2
        )
        branch = branch_proc.stdout.strip() if branch_proc.returncode == 0 else ""
        if not branch:
            head_proc = subprocess.run(
                ["git", "-C", str(directory), "rev-parse", "--short", "HEAD"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=2
            )
            branch = f"HEAD ({head_proc.stdout.strip()})" if head_proc.returncode == 0 else "unknown"

        status_proc = subprocess.run(
            ["git", "-C", str(directory), "status", "--porcelain"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=3
        )
        status_lines = [l for l in status_proc.stdout.splitlines() if l.strip()] if status_proc.returncode == 0 else []

        remote_proc = subprocess.run(
            ["git", "-C", str(directory), "remote", "get-url", "origin"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=2
        )
        remote_url = remote_proc.stdout.strip() if remote_proc.returncode == 0 else None

        log_proc = subprocess.run(
            ["git", "-C", str(directory), "log", "-1", "--format=%h %s (%cr)"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=2
        )
        last_commit = log_proc.stdout.strip() if log_proc.returncode == 0 else None

        return {
            "is_git_repo": True,
            "git_root": git_root,
            "branch": branch,
            "clean": len(status_lines) == 0,
            "uncommitted_changes": len(status_lines),
            "remote_url": remote_url,
            "last_commit": last_commit
        }
    except Exception:
        return {"is_git_repo": False}

def detect_project_tech_stack(project_root: Path) -> list[str]:
    """Detects tech stack, frameworks, and build tools in a project directory."""
    if not project_root or not project_root.is_dir():
        return []
    stack = []

    # 1. Node / Frontend / JS / TS
    pkg_json = project_root / "package.json"
    if pkg_json.exists():
        try:
            with open(pkg_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
            if "next" in deps:
                stack.append("Next.js")
            elif "react" in deps:
                stack.append("React")
            if "vue" in deps:
                stack.append("Vue")
            if "svelte" in deps:
                stack.append("Svelte")
            if "astro" in deps:
                stack.append("Astro")
            if "vite" in deps:
                stack.append("Vite")
            if "tailwindcss" in deps:
                stack.append("TailwindCSS")
            if "electron" in deps:
                stack.append("Electron")
            if "typescript" in deps or (project_root / "tsconfig.json").exists():
                stack.append("TypeScript")
            elif not any("React" in s or "Next" in s or "Vue" in s for s in stack):
                stack.append("Node.js")
        except Exception:
            stack.append("Node.js")
    elif (project_root / "tsconfig.json").exists():
        stack.append("TypeScript")

    # 2. Python
    pyproject = project_root / "pyproject.toml"
    reqs = project_root / "requirements.txt"
    if pyproject.exists() or reqs.exists() or (project_root / "setup.py").exists() or list(project_root.glob("*.py")):
        py_label = "Python"
        found_fw = []
        text_to_check = ""
        if pyproject.exists():
            try:
                text_to_check += pyproject.read_text(encoding="utf-8", errors="ignore").lower()
            except Exception:
                pass
        if reqs.exists():
            try:
                text_to_check += reqs.read_text(encoding="utf-8", errors="ignore").lower()
            except Exception:
                pass
        if "fastapi" in text_to_check:
            found_fw.append("FastAPI")
        if "django" in text_to_check:
            found_fw.append("Django")
        if "flask" in text_to_check:
            found_fw.append("Flask")
        if "torch" in text_to_check or "pytorch" in text_to_check:
            found_fw.append("PyTorch")

        if found_fw:
            stack.append(f"{py_label} ({', '.join(found_fw)})")
        else:
            stack.append(py_label)

    # 3. Rust
    if (project_root / "Cargo.toml").exists():
        if (project_root / "src-tauri").exists():
            stack.append("Rust (Tauri)")
        else:
            stack.append("Rust")

    # 4. Go
    if (project_root / "go.mod").exists():
        stack.append("Go")

    # 5. Flutter / Dart
    if (project_root / "pubspec.yaml").exists():
        stack.append("Flutter / Dart")

    # 6. C / C++ / Systems
    if (project_root / "CMakeLists.txt").exists():
        stack.append("C++ (CMake)")
    elif (project_root / "Makefile").exists() and not any(x in stack for x in ["Node.js", "Python", "Rust", "Go"]):
        stack.append("C / Make")

    # 7. Containerization & Arch Linux
    if (project_root / "Dockerfile").exists() or (project_root / "compose.yaml").exists() or (project_root / "docker-compose.yml").exists():
        stack.append("Docker")
    if (project_root / "PKGBUILD").exists():
        stack.append("Arch PKGBUILD")

    return stack

def check_ollama_status() -> dict:
    """Checks Ollama daemon connectivity, latency, and embed model availability."""
    start_time = time.time()
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            latency_ms = round((time.time() - start_time) * 1000, 1)
            data = json.loads(resp.read().decode("utf-8"))
            models = [m.get("name", "") for m in data.get("models", [])]
            has_model = any(DEFAULT_EMBED_MODEL in m for m in models)
            return {
                "online": True,
                "latency_ms": latency_ms,
                "models_count": len(models),
                "embed_model": DEFAULT_EMBED_MODEL,
                "embed_model_available": has_model,
                "installed_models": models[:5]
            }
    except Exception as e:
        return {
            "online": False,
            "latency_ms": None,
            "embed_model": DEFAULT_EMBED_MODEL,
            "embed_model_available": False,
            "error": str(e)
        }

def get_paths_info(target_dir: Path = None) -> dict:
    """Returns a comprehensive, introspectable map of Central Brain paths, storage, active workspace state, git, and toolchain."""
    ensure_dirs()
    if not target_dir:
        target_dir = Path.cwd()
    else:
        target_dir = Path(target_dir).resolve()

    db_size = DB_PATH.stat().st_size if DB_PATH.exists() else 0
    wal_path = DB_PATH.with_suffix(".db-wal")
    wal_size = wal_path.stat().st_size if wal_path.exists() else 0
    facts_size = FACTS_PATH.stat().st_size if FACTS_PATH.exists() else 0
    prompt_size = SYSTEM_PROMPT_PATH.stat().st_size if SYSTEM_PROMPT_PATH.exists() else 0

    sources_count = 0
    registered_sources = []
    if SOURCES_PATH.exists():
        try:
            with open(SOURCES_PATH, "r", encoding="utf-8") as f:
                registered_sources = json.load(f)
                sources_count = len(registered_sources)
        except Exception:
            pass

    facts_count = 0
    total_chunks = 0
    if DB_PATH.exists():
        try:
            conn = get_db()
            facts_count = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        except Exception:
            pass

    backups = list(BACKUP_DIR.glob("*.tar.gz")) if BACKUP_DIR.exists() else []
    latest_backup = None
    if backups:
        newest = max(backups, key=lambda p: p.stat().st_mtime)
        latest_backup = {
            "filename": newest.name,
            "size_mb": round(newest.stat().st_size / (1024 * 1024), 2),
            "created_at": datetime.fromtimestamp(newest.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        }

    # Workspace Context
    project_state = get_project_state(target_dir)
    project_root = Path(project_state.get("project_path")) if project_state.get("project_path") else (target_dir if (target_dir / ".git").exists() else None)
    eval_dir = project_root if project_root else target_dir

    git_info = get_git_info(eval_dir)
    tech_stack = detect_project_tech_stack(eval_dir)

    # Check registration in sources.json
    is_registered_source = False
    try:
        check_paths = [eval_dir, eval_dir.resolve()]
        for s in registered_sources:
            sp = Path(s).resolve()
            if any(cp == sp or cp.is_relative_to(sp) for cp in check_paths if sp.exists()):
                is_registered_source = True
                break
    except Exception:
        pass

    # Chunks and facts for this workspace
    workspace_chunks = 0
    workspace_null_embeds = 0
    workspace_facts = 0
    if DB_PATH.exists():
        try:
            conn = get_db()
            prefix = str(eval_dir.resolve()) + "%"
            r1 = conn.execute("SELECT COUNT(*) FROM chunks WHERE file_path LIKE ?", (prefix,)).fetchone()
            workspace_chunks = r1[0] if r1 else 0
            r2 = conn.execute("SELECT COUNT(*) FROM chunks WHERE file_path LIKE ? AND embedding IS NULL", (prefix,)).fetchone()
            workspace_null_embeds = r2[0] if r2 else 0

            proj_name = eval_dir.name
            r3 = conn.execute("SELECT COUNT(*) FROM facts WHERE entity LIKE ? OR source LIKE ? OR fact LIKE ?",
                              (f"%{proj_name}%", f"%{proj_name}%", f"%{proj_name}%")).fetchone()
            workspace_facts = r3[0] if r3 else 0
        except Exception:
            pass

    # Toolchain
    ollama_status = check_ollama_status()
    toolchain_info = {
        "ollama": ollama_status,
        "sqlite_version": sqlite3.sqlite_version,
        "python_version": platform.python_version(),
        "os": "Arch Linux" if Path("/etc/arch-release").exists() else platform.system(),
        "kernel": platform.release(),
        "hostname": platform.node()
    }

    # Actionable Recommendations
    recommendations = []
    if not is_registered_source and project_root and project_root != Path.home():
        recommendations.append(f"Workspace is not registered in Central Brain sources. To index, add its path to sources.json or run `brain ingest {project_root}`.")
    if not project_state.get("resolved_file"):
        recommendations.append(f"No project map or state file found. Run `brain map init {eval_dir}` or `brain init-project <name>`.")
    elif project_state.get("file_type") == "documentation_fallback":
        recommendations.append(f"Using {Path(project_state.get('resolved_file')).name} as fallback. Run `brain map init {eval_dir}` to create an official spec-driven map.")
    if workspace_null_embeds > 0:
        recommendations.append(f"{workspace_null_embeds} chunks in this workspace lack vector embeddings. Run `brain doctor --fix` to backfill.")
    if not ollama_status.get("online"):
        recommendations.append("Ollama daemon is offline. Start it ('systemctl --user start ollama' or 'ollama serve') for dense vector search.")
    elif not ollama_status.get("embed_model_available"):
        recommendations.append(f"Embedding model '{DEFAULT_EMBED_MODEL}' is missing. Run `ollama pull {DEFAULT_EMBED_MODEL}`.")
    if git_info.get("is_git_repo") and not git_info.get("clean"):
        recommendations.append(f"Workspace has {git_info.get('uncommitted_changes')} uncommitted git changes. Review before major refactoring.")

    storage_info = {
        "brain_dir": str(BRAIN_DIR),
        "db_path": str(DB_PATH),
        "db_size_bytes": db_size,
        "db_size_mb": round(db_size / (1024 * 1024), 2),
        "wal_size_bytes": wal_size,
        "wal_size_mb": round(wal_size / (1024 * 1024), 2),
        "facts_path": str(FACTS_PATH),
        "facts_count": facts_count,
        "sources_path": str(SOURCES_PATH),
        "sources_count": sources_count,
        "system_prompt_path": str(SYSTEM_PROMPT_PATH),
        "backups_count": len(backups),
        "latest_backup": latest_backup
    }

    return {
        # Backward compatibility with existing callers
        "brain_dir": {"path": str(BRAIN_DIR), "exists": BRAIN_DIR.exists()},
        "db_path": {"path": str(DB_PATH), "exists": DB_PATH.exists(), "size_bytes": db_size, "size_mb": round(db_size / (1024 * 1024), 2)},
        "facts_path": {"path": str(FACTS_PATH), "exists": FACTS_PATH.exists(), "size_bytes": facts_size, "fact_count": facts_count},
        "sources_path": {"path": str(SOURCES_PATH), "exists": SOURCES_PATH.exists(), "sources_count": sources_count},
        "system_prompt_path": {"path": str(SYSTEM_PROMPT_PATH), "exists": SYSTEM_PROMPT_PATH.exists(), "size_bytes": prompt_size},
        "knowledge_dir": {"path": str(KNOWLEDGE_DIR), "exists": KNOWLEDGE_DIR.exists(), "files_count": len(list(KNOWLEDGE_DIR.glob("**/*.md"))) if KNOWLEDGE_DIR.exists() else 0},
        "projects_dir": {"path": str(PROJECTS_DIR), "exists": PROJECTS_DIR.exists(), "files_count": len(list(PROJECTS_DIR.glob("**/*.md"))) if PROJECTS_DIR.exists() else 0},
        "episodes_dir": {"path": str(EPISODES_DIR), "exists": EPISODES_DIR.exists(), "files_count": len(list(EPISODES_DIR.glob("*.md"))) if EPISODES_DIR.exists() else 0},
        "backups_dir": {"path": str(BACKUP_DIR), "exists": BACKUP_DIR.exists(), "backups_count": len(backups)},
        "workspace": {
            "target_dir": str(target_dir),
            "project_path": project_state.get("project_path"),
            "project_root": str(eval_dir) if eval_dir else None,
            "resolved_file": project_state.get("resolved_file"),
            "file_type": project_state.get("file_type"),
            "has_planning": bool(project_state.get("planning_dir"))
        },
        # Enhanced Introspection Map
        "storage": storage_info,
        "git": git_info,
        "tech_stack": tech_stack,
        "central_brain_status": {
            "is_registered_source": is_registered_source,
            "workspace_chunks_count": workspace_chunks,
            "workspace_null_embeddings": workspace_null_embeds,
            "workspace_facts_count": workspace_facts,
            "total_system_chunks": total_chunks,
            "total_system_facts": facts_count
        },
        "toolchain": toolchain_info,
        "recommendations": recommendations
    }

def generate_role_context(role: str = "general", target_dir: Path = None, max_tokens: int = 800) -> str:
    """Generates a compact, role-tailored system prompt snippet for subagent context injection."""
    role_norm = (role or "general").strip().lower()
    st = get_project_state(target_dir)

    lines = [
        f"# AGENT CONTEXT INJECTION: {role_norm.upper()}",
        "",
        "## 1. System Platform & Hardware Facts",
        "- **OS:** Arch Linux (Rolling release).",
        "- **Platform:** ASUS TUF Gaming A15 (FA506NFR) - AMD CPU + NVIDIA GPU.",
        "- **Wi-Fi & Bluetooth:** MediaTek MT7921 802.11ax PCIe (`14c3:7961`, `mt7921e`) and USB Bluetooth (`13d3:3563`, driver `btusb` / `btmtk`). (Decommissioned Realtek RTL8852BE).",
        "- **ACPI Sleep:** S0 (s2idle), S4, S5. (ACPI DSDT does NOT support S3 deep sleep).",
        ""
    ]

    lines.append("## 2. Active Project Context")
    if "error" not in st:
        proj_path = st.get("project_path", "")
        res_file = st.get("resolved_file", "")
        f_type = st.get("file_type", "")
        lines.append(f"- **Project Root:** {proj_path}")
        lines.append(f"- **State Source:** {res_file} ({f_type})")
        content = st.get("content", "")
        phase_m = re.search(r"\*\*Active Phase:\*\*\s*([^\n]+)", content)
        status_m = re.search(r"\*\*Status:\*\*\s*([^\n]+)", content)
        if phase_m:
            lines.append(f"- **Active Phase:** {phase_m.group(1).strip()}")
        if status_m:
            lines.append(f"- **Status:** {status_m.group(1).strip()}")
    else:
        lines.append("- **Status:** No active .planning/ or project_map.md detected.")
    lines.append("")

    conn = get_db()
    facts = []
    if role_norm in ["hardware", "system", "kernel", "audio", "wifi", "bluetooth"]:
        facts = conn.execute("""
            SELECT id, entity, category, fact FROM facts 
            WHERE category IN ('Fix', 'Rule') 
              AND (entity LIKE '%Bluetooth%' OR entity LIKE '%Wi-Fi%' OR entity LIKE '%Audio%' 
                   OR entity LIKE '%PipeWire%' OR entity LIKE '%udev%' OR entity LIKE '%Hardware%'
                   OR fact LIKE '%driver%' OR fact LIKE '%kernel%' OR fact LIKE '%udev%')
            ORDER BY id DESC LIMIT 6
        """).fetchall()
    elif role_norm in ["frontend", "web", "ui", "browser"]:
        facts = conn.execute("""
            SELECT id, entity, category, fact FROM facts 
            WHERE category IN ('Fix', 'Rule', 'Knowledge')
              AND (entity LIKE '%Web%' OR entity LIKE '%Frontend%' OR fact LIKE '%Chrome%' 
                   OR fact LIKE '%CSS%' OR fact LIKE '%DOM%' OR fact LIKE '%UI%')
            ORDER BY id DESC LIMIT 6
        """).fetchall()
    elif role_norm in ["security", "audit"]:
        facts = conn.execute("""
            SELECT id, entity, category, fact FROM facts 
            WHERE category IN ('Rule', 'Fix') 
              AND (entity LIKE '%Security%' OR fact LIKE '%permission%' OR fact LIKE '%credential%'
                   OR fact LIKE '%secret%' OR fact LIKE '%D-Bus%' OR fact LIKE '%policy%')
            ORDER BY id DESC LIMIT 6
        """).fetchall()
    else:
        facts = conn.execute("""
            SELECT id, entity, category, fact FROM facts 
            WHERE category IN ('Fix', 'Rule')
            ORDER BY id DESC LIMIT 5
        """).fetchall()

    lines.append(f"## 3. Verified Rules & Fixes ({role_norm})")
    if facts:
        for f in facts:
            lines.append(f"- [#{f['id']}] **[{f['category']}]** ({f['entity']}): {f['fact']}")
    else:
        lines.append("- No specific rules found. Query dynamically via `brain query`.")
    lines.append("")

    lines.extend([
        "## 4. Execution Directives",
        "- **Spec-Driven Loop:** DISCUSS -> PLAN -> EXECUTE -> VERIFY -> SHIP & REMEMBER.",
        "- **Quality Gates:** Empirically verify all code and commands before completing tasks.",
        "- **Memory Persistence:** When resolving issues, persist verified findings via:",
        '  `brain remember "<fact>" --entity "<Topic>" --category "<Fix|Rule>"`',
        ""
    ])

    raw_text = "\n".join(lines)
    return apply_token_budget(raw_text, max_tokens)

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
        if SOURCES_PATH.exists():
            shutil.copy2(SOURCES_PATH, temp_snapshot_dir / "sources.json")

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
            shutil.copy2(extract_tmp / "sources.json", SOURCES_PATH)

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
    default_sources = [
        str(KNOWLEDGE_DIR),
        str(PROJECTS_DIR),
        str(EPISODES_DIR),
        str(Path.home() / "Documents" / "Configs"),
        str(Path.home() / ".agents" / "project_map.md"),
        str(Path.home() / "AGENTS.md")
    ]

    if not SOURCES_PATH.exists():
        with open(SOURCES_PATH, "w", encoding="utf-8") as f:
            json.dump(default_sources, f, indent=2)
        sources = default_sources
    else:
        try:
            with open(SOURCES_PATH, "r", encoding="utf-8") as f:
                sources = json.load(f)
        except Exception:
            sources = default_sources

    # Also auto-discover any .planning folders and project maps under ~/Projects
    sources_modified = False
    projects_base = Path.home() / "Projects"
    if projects_base.exists():
        for pl_dir in projects_base.glob("*/.planning"):
            pl_str = str(pl_dir)
            if pl_str not in sources:
                sources.append(pl_str)
                sources_modified = True
        for map_f in projects_base.glob("*/.agents/project_map.md"):
            map_str = str(map_f)
            if map_str not in sources:
                sources.append(map_str)
                sources_modified = True
        for map_f2 in projects_base.glob("*/*/.agents/project_map.md"):
            map_str2 = str(map_f2)
            if map_str2 not in sources:
                sources.append(map_str2)
                sources_modified = True

    if sources_modified:
        try:
            with open(SOURCES_PATH, "w", encoding="utf-8") as f:
                json.dump(sources, f, indent=2)
        except Exception:
            pass

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
    backfilled_count = backfill_missing_embeddings()

    return synced_paths, total_chunks, orphan_files, orphan_chunks, backfilled_count

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

def run_doctor(fix: bool = False) -> tuple[bool, dict]:
    """Runs a 7-point health check across SQLite, Ollama, vector completeness, registries, and backups.
    If fix is True, automatically repairs recoverable defects.
    Returns (all_passed, results_dict).
    """
    ensure_dirs()
    conn = get_db()
    results = {
        "sqlite_integrity": {"passed": False, "detail": ""},
        "ollama_embed": {"passed": False, "detail": ""},
        "facts_sync": {"passed": False, "detail": ""},
        "vector_completeness": {"passed": False, "detail": ""},
        "sources_health": {"passed": False, "detail": ""},
        "fts5_index": {"passed": False, "detail": ""},
        "backup_freshness": {"passed": False, "detail": ""}
    }
    fixes_applied = []

    # 1. SQLite DB Integrity
    try:
        cur = conn.execute("PRAGMA integrity_check;")
        row = cur.fetchone()
        if row and row[0] == "ok":
            db_size_mb = round(DB_PATH.stat().st_size / (1024 * 1024), 2) if DB_PATH.exists() else 0
            results["sqlite_integrity"]["passed"] = True
            results["sqlite_integrity"]["detail"] = f"Integrity check returned ok ({db_size_mb} MB)"
        else:
            err_msg = row[0] if row else "Unknown integrity error"
            results["sqlite_integrity"]["passed"] = False
            results["sqlite_integrity"]["detail"] = f"Integrity check failed: {err_msg}"
            if fix:
                conn.execute("VACUUM;")
                fixes_applied.append("Ran VACUUM on database.")
    except Exception as e:
        results["sqlite_integrity"]["passed"] = False
        results["sqlite_integrity"]["detail"] = f"Integrity check exception: {e}"

    # 2. Ollama Embeddings Engine & Model
    ollama_info = check_ollama_status()
    if ollama_info.get("online"):
        if ollama_info.get("embed_model_available"):
            results["ollama_embed"]["passed"] = True
            results["ollama_embed"]["detail"] = f"ONLINE ({ollama_info.get('latency_ms')} ms), '{DEFAULT_EMBED_MODEL}' ready"
        else:
            results["ollama_embed"]["passed"] = False
            results["ollama_embed"]["detail"] = f"ONLINE ({ollama_info.get('latency_ms')} ms), but '{DEFAULT_EMBED_MODEL}' is missing"
    else:
        results["ollama_embed"]["passed"] = False
        results["ollama_embed"]["detail"] = f"OFFLINE (Cannot reach {OLLAMA_EMBED_URL})"

    # 3. Facts Table & facts.json Synchronization
    try:
        db_facts_count = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        json_facts_count = 0
        cached = []
        if FACTS_PATH.exists():
            with open(FACTS_PATH, "r", encoding="utf-8") as f:
                cached = json.load(f)
                json_facts_count = len(cached) if isinstance(cached, list) else 0

        if db_facts_count == json_facts_count and db_facts_count > 0:
            results["facts_sync"]["passed"] = True
            results["facts_sync"]["detail"] = f"In sync ({db_facts_count} SQLite rows == {json_facts_count} facts.json entries)"
        elif db_facts_count == 0 and json_facts_count == 0:
            results["facts_sync"]["passed"] = True
            results["facts_sync"]["detail"] = "Empty (0 facts in SQLite or facts.json)"
        else:
            results["facts_sync"]["passed"] = False
            results["facts_sync"]["detail"] = f"Mismatch: {db_facts_count} SQLite rows vs {json_facts_count} facts.json entries"
            if fix:
                if db_facts_count == 0 and json_facts_count > 0:
                    for item in cached:
                        conn.execute(
                            "INSERT OR IGNORE INTO facts (id, entity, category, fact, source, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                            (item.get("id"), item.get("entity", "General"), item.get("category", "Knowledge"), item.get("fact", ""), item.get("source", "Restored"), item.get("timestamp", datetime.now().isoformat()))
                        )
                    fixes_applied.append(f"Rehydrated {json_facts_count} facts from facts.json into SQLite.")
                else:
                    sync_facts_json()
                    fixes_applied.append("Synchronized SQLite facts into facts.json.")
                results["facts_sync"]["passed"] = True
    except Exception as e:
        results["facts_sync"]["passed"] = False
        results["facts_sync"]["detail"] = f"Sync check exception: {e}"

    # 4. Vector Completeness
    try:
        total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        null_chunks = conn.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NULL").fetchone()[0]
        if null_chunks == 0:
            results["vector_completeness"]["passed"] = True
            results["vector_completeness"]["detail"] = f"Complete ({total_chunks}/{total_chunks} chunks have embeddings)"
        else:
            results["vector_completeness"]["passed"] = False
            results["vector_completeness"]["detail"] = f"{null_chunks} chunks missing embeddings ({total_chunks - null_chunks}/{total_chunks} embedded)"
            if fix:
                if ollama_info.get("online") and ollama_info.get("embed_model_available"):
                    backfilled = backfill_missing_embeddings()
                    if backfilled > 0:
                        fixes_applied.append(f"Backfilled {backfilled} missing chunk embeddings via Ollama.")
                        results["vector_completeness"]["passed"] = True
                    else:
                        fixes_applied.append("Attempted embedding backfill but 0 vectors were returned.")
                else:
                    fixes_applied.append("Cannot backfill embeddings: Ollama daemon or model is offline.")
    except Exception as e:
        results["vector_completeness"]["passed"] = False
        results["vector_completeness"]["detail"] = f"Vector check exception: {e}"

    # 5. Sources Registry Health
    dead_sources = []
    total_sources = 0
    srcs = []
    if SOURCES_PATH.exists():
        try:
            with open(SOURCES_PATH, "r", encoding="utf-8") as f:
                srcs = json.load(f)
            total_sources = len(srcs)
            for s in srcs:
                if not Path(s).exists():
                    dead_sources.append(s)
        except Exception:
            pass

    if not dead_sources:
        results["sources_health"]["passed"] = True
        results["sources_health"]["detail"] = f"All {total_sources} registered source paths exist on disk"
    else:
        results["sources_health"]["passed"] = False
        results["sources_health"]["detail"] = f"{len(dead_sources)} dead source path(s) found"
        if fix:
            try:
                valid_srcs = [s for s in srcs if s not in dead_sources]
                with open(SOURCES_PATH, "w", encoding="utf-8") as f:
                    json.dump(valid_srcs, f, indent=2)
                clean_orphans()
                fixes_applied.append(f"Pruned {len(dead_sources)} dead sources and cleaned orphan database chunks.")
                results["sources_health"]["passed"] = True
            except Exception as e:
                fixes_applied.append(f"Failed pruning dead sources: {e}")

    # 6. FTS5 Index Consistency
    try:
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('integrity-check');")
        results["fts5_index"]["passed"] = True
        results["fts5_index"]["detail"] = "chunks_fts virtual table consistent"
    except Exception as e:
        results["fts5_index"]["passed"] = False
        results["fts5_index"]["detail"] = f"FTS5 integrity check failed: {e}"
        if fix:
            try:
                conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild');")
                fixes_applied.append("Rebuilt chunks_fts full-text index.")
                results["fts5_index"]["passed"] = True
            except Exception as e2:
                fixes_applied.append(f"FTS5 rebuild failed: {e2}")

    # 7. Backup Freshness (< 7 days)
    backups = list(BACKUP_DIR.glob("*.tar.gz")) if BACKUP_DIR.exists() else []
    if backups:
        newest = max(backups, key=lambda p: p.stat().st_mtime)
        age_days = round((time.time() - newest.stat().st_mtime) / (24 * 3600), 1)
        if age_days <= 7.0:
            results["backup_freshness"]["passed"] = True
            results["backup_freshness"]["detail"] = f"Recent backup found: {newest.name} ({age_days} days ago)"
        else:
            results["backup_freshness"]["passed"] = False
            results["backup_freshness"]["detail"] = f"Latest backup is stale: {newest.name} ({age_days} days ago)"
            if fix:
                ok, msg, _ = backup_brain()
                if ok:
                    fixes_applied.append("Created fresh transactional backup archive.")
                    results["backup_freshness"]["passed"] = True
    else:
        results["backup_freshness"]["passed"] = False
        results["backup_freshness"]["detail"] = "No backups found in backups directory"
        if fix:
            ok, msg, _ = backup_brain()
            if ok:
                fixes_applied.append("Created initial backup archive.")
                results["backup_freshness"]["passed"] = True

    all_passed = all(v["passed"] for v in results.values())
    results["_meta"] = {
        "all_passed": all_passed,
        "fixes_applied": fixes_applied,
        "timestamp": datetime.now().isoformat()
    }
    return all_passed, results

def list_brain_items(kind: str = "facts", category: str = None, entity: str = None, limit: int = 25) -> tuple[str, list]:
    """Lists facts, sources, discovered projects, or backups."""
    ensure_dirs()
    kind = (kind or "facts").lower().strip()

    if kind in ["facts", "fact"]:
        conn = get_db()
        q = "SELECT id, entity, category, fact, timestamp FROM facts"
        params = []
        wheres = []
        if category:
            wheres.append("LOWER(category) = LOWER(?)")
            params.append(category)
        if entity:
            wheres.append("LOWER(entity) = LOWER(?)")
            params.append(entity)
        if wheres:
            q += " WHERE " + " AND ".join(wheres)
        q += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(q, params).fetchall()
        items = [dict(r) for r in rows]
        return "facts", items

    elif kind in ["sources", "source"]:
        items = []
        if SOURCES_PATH.exists():
            try:
                with open(SOURCES_PATH, "r", encoding="utf-8") as f:
                    src_list = json.load(f)
                conn = get_db()
                for s in src_list:
                    p = Path(s)
                    exists = p.exists()
                    chunks_cnt = 0
                    if exists:
                        prefix = str(p.resolve()) + ("%" if p.is_dir() else "")
                        row = conn.execute("SELECT COUNT(*) FROM chunks WHERE file_path LIKE ?", (prefix,)).fetchone()
                        chunks_cnt = row[0] if row else 0
                    items.append({
                        "path": s,
                        "exists": exists,
                        "type": "directory" if p.is_dir() else ("file" if p.is_file() else "missing"),
                        "indexed_chunks": chunks_cnt
                    })
            except Exception:
                pass
        return "sources", items

    elif kind in ["projects", "project"]:
        items = []
        projects_base = Path.home() / "Projects"
        search_dirs = [projects_base]
        if PROJECTS_DIR.exists() and PROJECTS_DIR != projects_base:
            search_dirs.append(PROJECTS_DIR)

        visited = set()
        for base in search_dirs:
            if not base.exists():
                continue
            for item in sorted(base.iterdir()):
                if item.is_dir() and not item.name.startswith("."):
                    res_path = str(item.resolve())
                    if res_path in visited:
                        continue
                    visited.add(res_path)
                    st = get_project_state(item)
                    stack = detect_project_tech_stack(item)
                    git = get_git_info(item)
                    items.append({
                        "name": item.name,
                        "path": res_path,
                        "resolved_file": st.get("resolved_file"),
                        "file_type": st.get("file_type"),
                        "tech_stack": stack,
                        "git_branch": git.get("branch") if git.get("is_git_repo") else None,
                        "git_clean": git.get("clean") if git.get("is_git_repo") else None
                    })
        return "projects", items

    elif kind in ["backups", "backup"]:
        items = []
        if BACKUP_DIR.exists():
            for bk in sorted(BACKUP_DIR.glob("*.tar.gz"), key=lambda p: p.stat().st_mtime, reverse=True):
                sz_mb = round(bk.stat().st_size / (1024 * 1024), 2)
                mtime_str = datetime.fromtimestamp(bk.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                items.append({
                    "filename": bk.name,
                    "path": str(bk),
                    "size_mb": sz_mb,
                    "created_at": mtime_str
                })
        return "backups", items

    else:
        return "error", [{"error": f"Unknown list type '{kind}'. Choose from 'facts', 'sources', 'projects', 'backups'."}]

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
                        "serverInfo": {"name": "central-brain", "version": "2.2.0"}
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
                                        "category": {"type": "string", "description": "Optional category filter (Fix/Rule/Knowledge/Project)"},
                                        "compact": {"type": "boolean", "default": False, "description": "Token-efficient compact output format"}
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
                                        "path": {"type": "string", "description": "Optional project path"},
                                        "section": {"type": "string", "description": "Optional section name filter"}
                                    }
                                }
                            },
                            {
                                "name": "brain_info",
                                "description": "Inspect all Central Brain system paths, files, databases, and active workspace resolution.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "path": {"type": "string", "description": "Optional workspace path"}
                                    }
                                }
                            },
                            {
                                "name": "brain_inject",
                                "description": "Generate a compact (<800 token) role-tailored system prompt snippet with verified hardware facts, rules, and project context for subagent initialization.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "role": {"type": "string", "enum": ["general", "system", "hardware", "frontend", "backend", "security"], "default": "general"},
                                        "path": {"type": "string", "description": "Optional project path"},
                                        "tokens": {"type": "integer", "default": 800}
                                    }
                                }
                            },
                            {
                                "name": "brain_map_add",
                                "description": "Append an entry to a specific section of the active project map (.agents/project_map.md or .planning/STATE.md).",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "section": {"type": "string", "description": "Section title"},
                                        "entry": {"type": "string", "description": "Entry or bullet to append"},
                                        "path": {"type": "string", "description": "Optional project path"}
                                    },
                                    "required": ["section", "entry"]
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
                                "description": "Remove wrong or outdated facts from the Central Brain by search term or exact ID.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "target": {"type": "string", "description": "Search term, keyword, or fact ID"},
                                        "id": {"type": "integer", "description": "Exact fact ID for precision deletion"},
                                        "entity": {"type": "string", "description": "Optional entity name"}
                                    }
                                }
                            },
                            {
                                "name": "brain_correct",
                                "description": "Correct/supersede an existing memory or fact with a new finding by ID or entity.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "new_fact": {"type": "string", "description": "The new corrected fact or solution"},
                                        "id": {"type": "integer", "description": "Exact fact ID to update in-place"},
                                        "entity": {"type": "string", "description": "Entity or topic to correct"},
                                        "old_fact_search": {"type": "string", "description": "Optional keyword of the old fact to replace"},
                                        "category": {"type": "string", "default": "Fix"}
                                    },
                                    "required": ["new_fact"]
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
                            },
                            {
                                "name": "brain_doctor",
                                "description": "Run 7-point health check across SQLite, Ollama, vector completeness, registries, and backups, with optional auto-repair.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "fix": {"type": "boolean", "default": False, "description": "Automatically repair detected defects"}
                                    }
                                }
                            },
                            {
                                "name": "brain_list",
                                "description": "List facts, registered sources, discovered projects, or backups.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "kind": {"type": "string", "enum": ["facts", "sources", "projects", "backups"], "default": "facts"},
                                        "category": {"type": "string", "description": "Filter facts by category"},
                                        "entity": {"type": "string", "description": "Filter facts by entity"},
                                        "limit": {"type": "integer", "default": 25}
                                    }
                                }
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
                    if args.get("compact"):
                        compact_facts = [{"id": f["id"], "category": f["category"], "entity": f["entity"], "fact": f["fact"]} for f in res.get("facts", [])]
                        compact_chunks = [{"file": Path(c["file_path"]).name, "header": c.get("header"), "snippet": c.get("content", "")[:120].strip()} for c in res.get("chunks", [])]
                        res = {"facts": compact_facts, "chunks": compact_chunks}
                    resp = {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}}
                elif name == "brain_remember":
                    remember(args.get("fact"), args.get("entity", "General"), args.get("category", "Knowledge"), source="MCP")
                    resp = {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": "Fact saved to Central Brain successfully."}]}}
                elif name == "brain_state":
                    res = get_project_state(args.get("path"))
                    if args.get("section") and "content" in res:
                        sec_content, all_secs = extract_markdown_section(res["content"], args.get("section"))
                        if sec_content:
                            res["section"] = args.get("section")
                            res["content"] = sec_content
                        else:
                            res["error"] = f"Section '{args.get('section')}' not found. Available: {', '.join(all_secs)}"
                    resp = {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}}
                elif name == "brain_info":
                    res = get_paths_info(args.get("path"))
                    resp = {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}}
                elif name == "brain_inject":
                    ctx = generate_role_context(args.get("role", "general"), args.get("path"), args.get("tokens", 800))
                    resp = {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": ctx}]}}
                elif name == "brain_map_add":
                    st = get_project_state(args.get("path"))
                    target_file = st.get("resolved_file")
                    if target_file:
                        ok, msg = map_add_entry(Path(target_file), args.get("section"), args.get("entry"))
                    else:
                        ok, msg = False, "No active project map or state file found."
                    resp = {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": msg}]}}
                elif name == "brain_init_project":
                    ok, msg = init_project(args.get("name"), args.get("path"), args.get("description", ""))
                    resp = {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": msg}]}}
                elif name == "brain_forget":
                    cnt = forget(args.get("target"), args.get("entity"), fact_id=args.get("id"))
                    resp = {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": f"Purged {cnt} matching fact(s) from Central Brain."}]}}
                elif name == "brain_correct":
                    ok = correct(args.get("entity"), args.get("new_fact"), args.get("old_fact_search"), args.get("category", "Fix"), source="MCP", fact_id=args.get("id"))
                    resp = {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": "Successfully corrected memory in Central Brain." if ok else "Failed to correct memory: fact not found."}]}}
                elif name == "brain_export":
                    digest = export_brain(days=args.get("days", 30), category=args.get("category"))
                    resp = {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": digest}]}}
                elif name == "brain_status":
                    res = get_status()
                    resp = {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}}
                elif name == "brain_doctor":
                    passed, res = run_doctor(fix=args.get("fix", False))
                    resp = {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}}
                elif name == "brain_list":
                    kind, items = list_brain_items(args.get("kind", "facts"), args.get("category"), args.get("entity"), args.get("limit", 25))
                    resp = {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": json.dumps({"kind": kind, "items": items}, indent=2)}]}}
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

class BrainArgumentParser(argparse.ArgumentParser):
    """ArgumentParser with fuzzy command suggestions and rich formatted help."""
    def error(self, message):
        if "invalid choice: " in message:
            match = re.search(r"invalid choice: '([^']+)'", message)
            if match:
                bad_choice = match.group(1)
                subparsers_actions = [
                    action for action in self._actions 
                    if isinstance(action, argparse._SubParsersAction)
                ]
                valid_choices = []
                for subaction in subparsers_actions:
                    valid_choices.extend(subaction.choices.keys())

                matches = difflib.get_close_matches(bad_choice, valid_choices, n=3, cutoff=0.5)
                sys.stderr.write(f"\n❌ Error: Unknown command '{bad_choice}'.\n")
                if matches:
                    sys.stderr.write(f"💡 Did you mean: {', '.join([repr(m) for m in matches])}?\n")
                sys.stderr.write("\nRun 'brain --help' to see all available commands.\n\n")
                sys.exit(2)
        super().error(message)

def main():
    epilog_text = """
Examples:
  # Introspection & Diagnostics
  brain info                      Inspect paths, active workspace, Git status, and toolchain
  brain info /path/to/project     Inspect a specific project workspace
  brain doctor                    Run 7-point health check across SQLite, Ollama, and registries
  brain doctor --fix              Automatically self-heal missing vectors, dead sources, and FTS5

  # Dynamic Memory & Search
  brain query "mt7921 bluetooth"  Query verified facts & indexed documents
  brain query "audio" --compact   Token-efficient 1-line facts with [#id]
  brain remember "rule" -c Rule   Save verified solution or rule
  brain list facts                List recent facts with IDs and categories
  brain correct --id 42 "new"     In-place deterministic update by ID
  brain forget --id 42            Deterministic deletion by ID

  # Project State & Architecture Map
  brain state                     Show resolved project state or documentation summary
  brain state -s "Decisions"      Inspect a specific section without context blowout
  brain state --add action "task" Add action item to active STATE.md
  brain map show                  Display active project map (.agents/project_map.md)
  brain map add "Rules" "- rule"  Append rule to project map section
  brain inject hardware           Generate verified context prompt for subagents (<800 tokens)
"""
    parser = BrainArgumentParser(
        description="Central Brain - Unified Local Agent Memory & State CLI",
        epilog=epilog_text,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--json", action="store_true", help="Output all results in structured JSON format")
    sub = parser.add_subparsers(dest="command")

    # query / search
    q_p = sub.add_parser("query", aliases=["search"], help="Query the central brain")
    q_p.add_argument("text", type=str, help="Search query")
    q_p.add_argument("-k", "--top-k", type=int, default=5, help="Number of results")
    q_p.add_argument("-e", "--entity", type=str, default=None, help="Filter by entity")
    q_p.add_argument("-c", "--category", type=str, default=None, help="Filter by category")
    q_p.add_argument("-s", "--source", type=str, default=None, help="Filter by source")
    q_p.add_argument("--since", type=str, default=None, help="Filter since date (YYYY-MM-DD)")
    q_p.add_argument("--until", type=str, default=None, help="Filter until date (YYYY-MM-DD)")
    q_p.add_argument("-p", "--path", type=str, default=None, help="Filter by file path pattern")
    q_p.add_argument("--compact", "--terse", action="store_true", dest="compact", help="Token-efficient single-line output mode")
    q_p.add_argument("--max-tokens", type=int, default=None, help="Limit output to approximate token budget")
    q_p.add_argument("--json", action="store_true", help="Output in JSON format")

    # state / plan
    st_cmd = sub.add_parser("state", aliases=["plan"], help="Inspect or mutate spec-driven project state (.planning/STATE.md)")
    st_cmd.add_argument("path", nargs="?", default=None, help="Optional project directory")
    st_cmd.add_argument("-s", "--section", type=str, default=None, help="Filter output to a specific section name")
    st_cmd.add_argument("-c", "--compact", "--summary", action="store_true", dest="compact", help="Token-efficient summary view with section directory")
    st_cmd.add_argument("--outline", action="store_true", help="Display only the section outline of the document")
    st_cmd.add_argument("--full", action="store_true", help="Display full un-truncated content")
    st_cmd.add_argument("--max-tokens", type=int, default=None, help="Limit output to approximate token budget")
    st_cmd.add_argument("--add", nargs=2, metavar=("TYPE", "TEXT"), help="Add an entry: --add <action|decision|blocker> '<text>'")
    st_cmd.add_argument("--phase", type=str, default=None, help="Update active phase in STATE.md")
    st_cmd.add_argument("--status", type=str, default=None, help="Update status in STATE.md")
    st_cmd.add_argument("--json", action="store_true", help="Output in JSON format")

    # map
    map_p = sub.add_parser("map", help="Inspect and mutate spec-driven project map (.agents/project_map.md)")
    map_sub = map_p.add_subparsers(dest="map_action")

    map_show = map_sub.add_parser("show", help="Display the active project map")
    map_show.add_argument("path", nargs="?", default=None, help="Optional project directory")
    map_show.add_argument("-s", "--section", type=str, default=None, help="Filter output to a specific section name")
    map_show.add_argument("-c", "--compact", "--summary", action="store_true", dest="compact", help="Token-efficient summary view with section directory")
    map_show.add_argument("--outline", action="store_true", help="Display only the section outline of the document")
    map_show.add_argument("--full", action="store_true", help="Display full un-truncated content")
    map_show.add_argument("--max-tokens", type=int, default=None, help="Limit output to approximate token budget")
    map_show.add_argument("--json", action="store_true", help="Output in JSON format")

    map_ls = map_sub.add_parser("list-sections", help="List all markdown section headers in the active project map")
    map_ls.add_argument("path", nargs="?", default=None, help="Optional project directory")
    map_ls.add_argument("--json", action="store_true", help="Output in JSON format")

    map_add = map_sub.add_parser("add", help="Append an entry to a specific section in the project map")
    map_add.add_argument("section", type=str, help="Target section header name (e.g. 'Active System Rules')")
    map_add.add_argument("entry", type=str, help="Bullet point or text entry to append")
    map_add.add_argument("path", nargs="?", default=None, help="Optional project directory")
    map_add.add_argument("--json", action="store_true", help="Output in JSON format")

    map_set = map_sub.add_parser("set-section", help="Replace the body of a specific section in the project map")
    map_set.add_argument("section", type=str, help="Target section header name")
    map_set.add_argument("content", type=str, help="New body content for the section")
    map_set.add_argument("path", nargs="?", default=None, help="Optional project directory")
    map_set.add_argument("--json", action="store_true", help="Output in JSON format")

    map_init_cmd = map_sub.add_parser("init", help="Scaffold a new .agents/project_map.md in the project directory")
    map_init_cmd.add_argument("path", nargs="?", default=None, help="Target project directory (default: current dir)")
    map_init_cmd.add_argument("--json", action="store_true", help="Output in JSON format")

    # info / paths
    info_p = sub.add_parser("info", aliases=["paths"], help="Display Central Brain paths, configuration introspection, and active workspace map")
    info_p.add_argument("path", nargs="?", default=None, help="Optional workspace directory to inspect")
    info_p.add_argument("--paths", action="store_true", help="Display paths and storage locations")
    info_p.add_argument("--json", action="store_true", help="Output in JSON format")

    # doctor / repair
    doc_p = sub.add_parser("doctor", aliases=["repair"], help="Run 7-point health check and self-heal Central Brain systems")
    doc_p.add_argument("--fix", action="store_true", help="Automatically repair detected issues (backfill embeddings, sync facts, prune dead sources, rebuild FTS5)")
    doc_p.add_argument("--json", action="store_true", help="Output diagnostic results in JSON format")

    # list / ls
    list_p = sub.add_parser("list", aliases=["ls"], help="List facts, registered sources, discovered projects, or backups")
    list_p.add_argument("kind", nargs="?", default="facts", choices=["facts", "sources", "projects", "backups"], help="Type of items to list (default: facts)")
    list_p.add_argument("-c", "--category", type=str, default=None, help="Filter facts by category")
    list_p.add_argument("-e", "--entity", type=str, default=None, help="Filter facts by entity")
    list_p.add_argument("-l", "--limit", type=int, default=25, help="Maximum number of items to display (default: 25)")
    list_p.add_argument("--json", action="store_true", help="Output in JSON format")

    # inject
    inj_p = sub.add_parser("inject", help="Generate a compact, role-tailored system prompt snippet for subagent context injection")
    inj_p.add_argument("role", nargs="?", default="general", choices=["general", "system", "hardware", "frontend", "web", "backend", "api", "security", "audit"], help="Target subagent role")
    inj_p.add_argument("-p", "--path", type=str, default=None, help="Optional project directory")
    inj_p.add_argument("-t", "--tokens", type=int, default=800, help="Maximum token budget for injected context (default: 800)")
    inj_p.add_argument("--json", action="store_true", help="Output in JSON format")

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
    f_p.add_argument("target", type=str, nargs="?", default=None, help="Search term/phrase of the fact to remove OR fact ID")
    f_p.add_argument("--id", type=int, default=None, help="Deterministic deletion by exact fact ID")
    f_p.add_argument("-e", "--entity", type=str, default=None, help="Specific entity/topic filter")
    f_p.add_argument("--json", action="store_true", help="Output in JSON format")

    # correct
    c_p = sub.add_parser("correct", help="Correct/supersede a memory with a new finding")
    c_p.add_argument("entity", type=str, nargs="?", default=None, help="Entity or topic name OR fact ID")
    c_p.add_argument("new_fact", type=str, nargs="?", default=None, help="The new, corrected fact or solution")
    c_p.add_argument("--id", type=int, default=None, help="Deterministic in-place update by exact fact ID")
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
        is_compact = getattr(args, "compact", False)
        max_tokens = getattr(args, "max_tokens", None)

        if is_json:
            out_data = res
            if is_compact:
                out_data = {
                    "facts": [{"id": f["id"], "category": f["category"], "entity": f["entity"], "fact": f["fact"]} for f in res.get("facts", [])],
                    "chunks": [{"file": Path(c["file_path"]).name, "header": c.get("header"), "snippet": c.get("content", "")[:120].strip()} for c in res.get("chunks", [])]
                }
            json_str = json.dumps({"status": "success", "command": "query", "data": out_data, "timestamp": datetime.now().isoformat()}, indent=2)
            print(apply_token_budget(json_str, max_tokens))
        else:
            if is_compact:
                lines = []
                if res["facts"]:
                    for f in res["facts"]:
                        lines.append(f"[#{f['id']}] [{f['category']}] ({f['entity']}): {f['fact']}")
                if res["chunks"]:
                    for c in res["chunks"]:
                        fname = Path(c['file_path']).name
                        snippet = c['content'].replace("\n", " ").strip()[:100]
                        lines.append(f"[Chunk] {fname} > {c['header']}: {snippet}...")
                if not lines:
                    lines.append("No matching facts or chunks found.")
                print(apply_token_budget("\n".join(lines), max_tokens))
            else:
                out_lines = [
                    f"\n🧠 CENTRAL BRAIN SEARCH RESULTS for: '{args.text}'",
                    "="*60
                ]
                if res["facts"]:
                    out_lines.append("\n📌 RELEVANT FACTS:")
                    for f in res["facts"]:
                        out_lines.append(f"  • [#{f['id']}] [{f['category']}] ({f['entity']}): {f['fact']} ({f['timestamp']})")
                if res["chunks"]:
                    out_lines.append("\n📄 RELEVANT KNOWLEDGE CHUNKS:")
                    for idx, c in enumerate(res["chunks"], 1):
                        path_name = Path(c['file_path']).name
                        out_lines.append(f"\n--- Result #{idx} [Score: {c['score']}] | File: {path_name} ({c['header']}) ---")
                        out_lines.append(c['content'].strip()[:400] + ("..." if len(c['content']) > 400 else ""))
                else:
                    out_lines.append("\nNo matching chunks found.")
                out_lines.append("")
                print(apply_token_budget("\n".join(out_lines), max_tokens))

    elif args.command in ["state", "plan"]:
        # Handle state mutations if flags passed
        if getattr(args, "add", None):
            entry_type, text = args.add
            ok, msg = state_add_entry(args.path, entry_type, text)
            if is_json:
                print(json.dumps({"status": "success" if ok else "error", "command": "state", "action": "add", "data": {"message": msg}}, indent=2))
            else:
                print(f"{'✅' if ok else '❌'} {msg}")
            return

        if getattr(args, "phase", None) or getattr(args, "status", None):
            ok, msg = state_update(args.path, phase=args.phase, status=args.status)
            if is_json:
                print(json.dumps({"status": "success" if ok else "error", "command": "state", "action": "update", "data": {"message": msg}}, indent=2))
            else:
                print(f"{'✅' if ok else '❌'} {msg}")
            return

        res = get_project_state(args.path)
        sec_filter = getattr(args, "section", None)
        is_compact = getattr(args, "compact", False)
        is_outline = getattr(args, "outline", False)
        is_full = getattr(args, "full", False)
        max_tokens = getattr(args, "max_tokens", None)

        if is_json:
            if sec_filter and "content" in res:
                sec_text, all_secs = extract_markdown_section(res["content"], sec_filter)
                if sec_text:
                    res["section"] = sec_filter
                    res["content"] = sec_text
                else:
                    res["error"] = f"Section '{sec_filter}' not found. Available: {', '.join(all_secs)}"
            elif is_outline and "content" in res:
                _, all_secs = extract_markdown_section(res["content"], "__all__")
                res["outline"] = all_secs
            json_str = json.dumps({"status": "success" if "error" not in res else "error", "command": "state", "data": res}, indent=2)
            print(apply_token_budget(json_str, max_tokens))
        else:
            if "error" in res:
                print(f"❌ {res['error']}")
                if "suggestion" in res:
                    print(f"💡 {res['suggestion']}")
            else:
                p_type = res.get("type")
                is_fallback = res.get("is_fallback", False)
                title_kind = "PROJECT MAP" if p_type == "project_map" else ("PROJECT STATE" if p_type == "planning" else "PROJECT OVERVIEW")
                resolved_info = f"📍 Resolved File: {res.get('resolved_file')} ({res.get('file_type', 'unknown')})"
                
                header_lines = [f"\n🧭 {title_kind} ({res.get('project_path')}):", resolved_info]
                if is_fallback:
                    idx_cnt = res.get('indexed_chunks', 0)
                    idx_status = f"{idx_cnt} chunks" if idx_cnt > 0 else "0 chunks (Unindexed)"
                    header_lines.append(f"📊 Central Brain Index: {idx_status}")
                    header_lines.append("💡 Notice: No autonomous .agents/project_map.md or .planning/ tracking found.")
                    header_lines.append(f"   Run 'brain map init {res.get('project_path')}' to scaffold an autonomous project map.")
                header_text = "\n".join(header_lines) + "\n" + "="*60

                body = res.get("content") or res.get("state") or res.get("project") or ""

                if is_outline:
                    _, all_secs = extract_markdown_section(body, "__all__")
                    full_out = f"{header_text}\n\n📋 Markdown Sections in {Path(res.get('resolved_file', '')).name}:\n" + "\n".join([f"  • {s}" for s in all_secs]) + "\n"
                elif sec_filter:
                    sec_text, all_secs = extract_markdown_section(body, sec_filter)
                    if sec_text:
                        full_out = f"{header_text}\n\n{sec_text}\n"
                    else:
                        full_out = f"{header_text}\n\n❌ Section '{sec_filter}' not found. Available sections:\n" + "\n".join([f"  • {s}" for s in all_secs]) + "\n"
                elif is_compact:
                    summary_text, all_secs = extract_project_doc_summary(body, max_overview_lines=8)
                    full_out = f"{header_text}\n\n{summary_text}\n\n📋 Available Sections ({len(all_secs)} sections):\n" + "\n".join([f"  • {s}" for s in all_secs]) + f"\n\n💡 Tip: To inspect a section without context blowout, run:\n   brain state {res.get('project_path')} -s \"<SectionName>\"\n"
                elif is_fallback and not is_full:
                    # Fallback documentation view (e.g. README): optimize instead of dumping entire file
                    summary_text, all_secs = extract_project_doc_summary(body, max_overview_lines=12)
                    sec_list = "\n".join([f"  • {s}" for s in all_secs[:10]])
                    if len(all_secs) > 10:
                        sec_list += f"\n  ... and {len(all_secs) - 10} more sections"
                    full_out = f"{header_text}\n\n{summary_text}\n\n📋 Available Sections in {Path(res.get('resolved_file', '')).name}:\n{sec_list}\n\n💡 Tip: View a specific section with: brain state {res.get('project_path')} -s \"<Section>\" (or pass --full to view entire file)\n"
                else:
                    full_out = f"{header_text}\n\n{body}\n"

                print(apply_token_budget(full_out, max_tokens))

    elif args.command == "map":
        action = getattr(args, "map_action", None)
        target_path = getattr(args, "path", None)
        sec_filter = getattr(args, "section", None)
        max_tokens = getattr(args, "max_tokens", None)

        st = get_project_state(target_path)
        resolved_file = st.get("resolved_file")

        if action == "list-sections":
            if not resolved_file or "content" not in st:
                err_msg = f"No project map or state file found in {target_path or Path.cwd()}"
                if is_json:
                    print(json.dumps({"status": "error", "command": "map", "error": err_msg}, indent=2))
                else:
                    print(f"❌ {err_msg}")
                return

            _, all_secs = extract_markdown_section(st["content"], "__none__")
            if is_json:
                print(json.dumps({"status": "success", "command": "map", "action": "list-sections", "file": resolved_file, "sections": all_secs}, indent=2))
            else:
                print(f"\n📋 Markdown Sections in {resolved_file}:")
                for s in all_secs:
                    print(f"  • {s}")
                print()

        elif action == "add":
            if not resolved_file or st.get("file_type") != "project_map":
                err_msg = f"No active project map (.agents/project_map.md) found in {target_path or Path.cwd()}. Run 'brain map init' first."
                if is_json:
                    print(json.dumps({"status": "error", "command": "map", "error": err_msg}, indent=2))
                else:
                    print(f"❌ {err_msg}")
                return

            ok, msg = map_add_entry(Path(resolved_file), args.section, args.entry)
            if is_json:
                print(json.dumps({"status": "success" if ok else "error", "command": "map", "action": "add", "data": {"message": msg, "file": resolved_file}}, indent=2))
            else:
                print(f"{'✅' if ok else '❌'} {msg}")

        elif action == "set-section":
            if not resolved_file or st.get("file_type") != "project_map":
                err_msg = f"No active project map (.agents/project_map.md) found in {target_path or Path.cwd()}. Run 'brain map init' first."
                if is_json:
                    print(json.dumps({"status": "error", "command": "map", "error": err_msg}, indent=2))
                else:
                    print(f"❌ {err_msg}")
                return

            ok, msg = map_set_section(Path(resolved_file), args.section, args.content)
            if is_json:
                print(json.dumps({"status": "success" if ok else "error", "command": "map", "action": "set-section", "data": {"message": msg, "file": resolved_file}}, indent=2))
            else:
                print(f"{'✅' if ok else '❌'} {msg}")

        elif action == "init":
            ok, msg = init_project_map(target_path)
            if is_json:
                print(json.dumps({"status": "success" if ok else "error", "command": "map", "action": "init", "data": {"message": msg}}, indent=2))
            else:
                print(f"{'✅' if ok else '❌'} {msg}")

        else:
            # Default map view
            is_compact = getattr(args, "compact", False)
            is_outline = getattr(args, "outline", False)
            is_full = getattr(args, "full", False)
            if is_json:
                if sec_filter and "content" in st:
                    sec_text, all_secs = extract_markdown_section(st["content"], sec_filter)
                    if sec_text:
                        st["section"] = sec_filter
                        st["content"] = sec_text
                    else:
                        st["error"] = f"Section '{sec_filter}' not found. Available: {', '.join(all_secs)}"
                elif is_outline and "content" in st:
                    _, all_secs = extract_markdown_section(st["content"], "__all__")
                    st["outline"] = all_secs
                json_str = json.dumps({"status": "success" if "error" not in st else "error", "command": "map", "data": st}, indent=2)
                print(apply_token_budget(json_str, max_tokens))
            else:
                if "error" in st:
                    print(f"❌ {st['error']}")
                    if "suggestion" in st:
                        print(f"💡 {st['suggestion']}")
                else:
                    header = f"\n🗺️ PROJECT MAP ({st.get('project_path')}):\n📍 Resolved File: {st.get('resolved_file')} ({st.get('file_type')})\n" + "="*60
                    body = st.get("content", "")
                    if is_outline:
                        _, all_secs = extract_markdown_section(body, "__all__")
                        full_out = f"{header}\n\n📋 Markdown Sections in {Path(st.get('resolved_file', '')).name}:\n" + "\n".join([f"  • {s}" for s in all_secs]) + "\n"
                    elif sec_filter:
                        sec_text, all_secs = extract_markdown_section(body, sec_filter)
                        if sec_text:
                            full_out = f"{header}\n\n{sec_text}\n"
                        else:
                            full_out = f"{header}\n\n❌ Section '{sec_filter}' not found. Available sections:\n" + "\n".join([f"  • {s}" for s in all_secs]) + "\n"
                    elif is_compact:
                        summary_text, all_secs = extract_project_doc_summary(body, max_overview_lines=8)
                        full_out = f"{header}\n\n{summary_text}\n\n📋 Available Sections ({len(all_secs)} sections):\n" + "\n".join([f"  • {s}" for s in all_secs]) + f"\n\n💡 Tip: To inspect a section without context blowout, run:\n   brain map show {st.get('project_path')} -s \"<SectionName>\"\n"
                    else:
                        full_out = f"{header}\n\n{body}\n"
                    print(apply_token_budget(full_out, max_tokens))

    elif args.command in ["info", "paths"]:
        paths_info = get_paths_info(args.path)
        if is_json:
            print(json.dumps({"status": "success", "command": "info", "data": paths_info, "timestamp": datetime.now().isoformat()}, indent=2))
        else:
            print("\n🧠 CENTRAL BRAIN & WORKSPACE INTROSPECTION")
            print("="*68)
            print("📁 Storage & Registries:")
            bd = paths_info["brain_dir"]
            print(f"  • Base Directory:       {bd['path']:<38} [{'EXISTS' if bd['exists'] else 'MISSING'}]")
            db = paths_info["db_path"]
            wal_mb = paths_info.get("storage", {}).get("wal_size_mb", 0)
            print(f"  • SQLite Database:      {db['path']:<38} [{'EXISTS' if db['exists'] else 'MISSING'}] ({db.get('size_mb', 0)} MB, WAL: {wal_mb} MB)")
            fp = paths_info["facts_path"]
            print(f"  • Facts Registry:       {fp['path']:<38} [{'EXISTS' if fp['exists'] else 'MISSING'}] ({fp.get('fact_count', 0)} facts)")
            sp = paths_info["sources_path"]
            print(f"  • Sources Registry:     {sp['path']:<38} [{'EXISTS' if sp['exists'] else 'MISSING'}] ({sp.get('sources_count', 0)} sources)")
            bk = paths_info["backups_dir"]
            print(f"  • Backups Directory:    {bk['path']:<38} [{'EXISTS' if bk['exists'] else 'MISSING'}] ({bk.get('backups_count', 0)} archives)")

            print("\n📍 Active Workspace Context:")
            ws = paths_info["workspace"]
            print(f"  • Target Directory:     {ws.get('target_dir')}")
            print(f"  • Project Root:         {ws.get('project_root') or 'None detected'}")
            tech = paths_info.get("tech_stack", [])
            print(f"  • Tech Stack:           {', '.join(tech) if tech else 'None detected'}")
            git = paths_info.get("git", {})
            if git.get("is_git_repo"):
                git_status_str = f"{git.get('branch')} ({'clean' if git.get('clean') else f'{git.get('uncommitted_changes')} uncommitted changes'})"
                print(f"  • Git Repository:       {git_status_str}")
                if git.get("last_commit"):
                    print(f"    Last Commit:          {git.get('last_commit')}")
                if git.get("remote_url"):
                    print(f"    Remote URL:           {git.get('remote_url')}")
            else:
                print(f"  • Git Repository:       Not a git repository")
            print(f"  • State File:           {ws.get('resolved_file') or 'None detected'} ({ws.get('file_type') or 'N/A'})")
            cb = paths_info.get("central_brain_status", {})
            reg_status = "Registered in sources" if cb.get("is_registered_source") else "Not registered in sources"
            print(f"  • Central Brain Status: {reg_status} ({cb.get('workspace_chunks_count', 0)} chunks indexed, {cb.get('workspace_null_embeddings', 0)} missing embeddings)")

            print("\n⚡ Toolchain & Services:")
            tc = paths_info.get("toolchain", {})
            ol = tc.get("ollama", {})
            ol_str = f"ONLINE ({ol.get('latency_ms')} ms) | Model: {ol.get('embed_model')} [{'READY' if ol.get('embed_model_available') else 'MISSING'}]" if ol.get("online") else "OFFLINE"
            print(f"  • Ollama Daemon:        {ol_str}")
            print(f"  • Python / SQLite:      {tc.get('python_version')} / {tc.get('sqlite_version')}")
            print(f"  • Host OS / Kernel:     {tc.get('os')} ({tc.get('kernel')}) on {tc.get('hostname')}")

            recs = paths_info.get("recommendations", [])
            if recs:
                print("\n💡 Actionable Recommendations:")
                for r in recs:
                    print(f"  • {r}")
            print("="*68 + "\n")

    elif args.command in ["doctor", "repair"]:
        all_passed, results = run_doctor(fix=getattr(args, "fix", False) or args.command == "repair")
        if is_json:
            print(json.dumps({"status": "success" if all_passed else "warning", "command": "doctor", "data": results}, indent=2))
        else:
            print("\n🩺 CENTRAL BRAIN HEALTH DIAGNOSTICS (7 Quality Gates)")
            print("="*68)
            gate_names = {
                "sqlite_integrity": "SQLite DB Integrity",
                "ollama_embed": "Ollama Embedding Engine",
                "facts_sync": "Facts Registry Sync",
                "vector_completeness": "Vector Embeddings",
                "sources_health": "Sources Registry Health",
                "fts5_index": "Full-Text Search (FTS5)",
                "backup_freshness": "Backup Freshness"
            }
            for k, name in gate_names.items():
                gate = results.get(k, {})
                status_badge = "[PASS]" if gate.get("passed") else "[FAIL]"
                print(f"  {status_badge:<8} {name:<26} {gate.get('detail', '')}")
            print("="*68)
            fixes = results.get("_meta", {}).get("fixes_applied", [])
            if fixes:
                print("🔧 Fixes Applied:")
                for fx in fixes:
                    print(f"  ✨ {fx}")
                print("="*68)

            if all_passed:
                print("✅ All health checks passed! Central Brain is in optimal state.\n")
            else:
                if not getattr(args, "fix", False) and args.command != "repair":
                    print("⚠️  Issues detected. Run `brain doctor --fix` (or `brain repair`) to auto-repair.\n")
                else:
                    print("⚠️  Some issues could not be resolved automatically. Review details above.\n")

    elif args.command in ["list", "ls"]:
        kind, items = list_brain_items(args.kind, args.category, args.entity, args.limit)
        if is_json:
            print(json.dumps({"status": "success", "command": "list", "kind": kind, "data": items, "count": len(items)}, indent=2))
        else:
            print(f"\n📋 CENTRAL BRAIN LIST: {kind.upper()} ({len(items)} items)")
            print("="*68)
            if kind == "facts":
                if not items:
                    print("  No facts found.")
                for item in items:
                    print(f"  [#{item['id']}] [{item['category']}] ({item['entity']}): {item['fact']}")
            elif kind == "sources":
                if not items:
                    print("  No registered sources found.")
                for item in items:
                    stat = "EXISTS" if item["exists"] else "MISSING"
                    print(f"  • {item['path']:<45} [{stat}] ({item['type']}, {item['indexed_chunks']} chunks)")
            elif kind == "projects":
                if not items:
                    print("  No projects found.")
                for item in items:
                    stack_str = ", ".join(item["tech_stack"]) if item["tech_stack"] else "General"
                    git_str = f"git: {item['git_branch']}" if item["git_branch"] else "no git"
                    state_str = Path(item["resolved_file"]).name if item["resolved_file"] else "no state"
                    print(f"  • {item['name']:<20} | {stack_str:<25} | {state_str:<18} | {git_str}")
                    print(f"    Path: {item['path']}")
            elif kind == "backups":
                if not items:
                    print("  No backups found.")
                for item in items:
                    print(f"  • {item['filename']:<35} ({item['size_mb']} MB) - {item['created_at']}")
            print("="*68 + "\n")

    elif args.command == "inject":
        ctx = generate_role_context(args.role, args.path, args.tokens)
        if is_json:
            print(json.dumps({"status": "success", "command": "inject", "role": args.role, "context": ctx}, indent=2))
        else:
            print(ctx)

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
        fact_id = getattr(args, "id", None)
        target = args.target
        if fact_id is None and target and re.match(r"^#?(\d+)$", target.strip()):
            fact_id = int(re.match(r"^#?(\d+)$", target.strip()).group(1))
            target = None

        cnt = forget(target, args.entity, fact_id=fact_id)
        if is_json:
            print(json.dumps({"status": "success", "command": "forget", "data": {"purged_count": cnt, "target": target, "id": fact_id, "entity": args.entity}}, indent=2))
        else:
            if fact_id is not None:
                if cnt > 0:
                    print(f"🗑️ Central Brain: Purged fact #{fact_id}.")
                else:
                    print(f"❌ Central Brain: Fact #{fact_id} not found.")
            else:
                print(f"🗑️ Central Brain: Purged {cnt} matching fact(s) matching '{target}' (Entity: {args.entity or 'Any'}).")

    elif args.command == "correct":
        fact_id = getattr(args, "id", None)
        entity = args.entity
        new_fact = args.new_fact

        # Check for numeric entity argument: e.g. brain correct 42 "new fact"
        if fact_id is None and entity and re.match(r"^#?(\d+)$", entity.strip()):
            fact_id = int(re.match(r"^#?(\d+)$", entity.strip()).group(1))
            entity = None

        # If called as: brain correct --id 42 "new fact" (where new_fact is captured in entity)
        if fact_id is not None and new_fact is None and entity is not None:
            new_fact = entity
            entity = None

        if not new_fact:
            err = "Error: A new corrected fact or solution must be provided."
            if is_json:
                print(json.dumps({"status": "error", "command": "correct", "error": err}, indent=2))
            else:
                print(f"❌ {err}")
            return

        ok = correct(entity, new_fact, args.old, args.category, args.source, fact_id=fact_id)
        if is_json:
            print(json.dumps({"status": "success" if ok else "error", "command": "correct", "data": {"entity": entity, "category": args.category, "new_fact": new_fact, "id": fact_id, "source": args.source}}, indent=2))
        else:
            if ok:
                if fact_id is not None:
                    print(f"✨ Central Brain: Successfully updated fact #{fact_id} -> {new_fact}")
                else:
                    print(f"✨ Central Brain: Successfully corrected memory for [{args.category}] ({entity}) -> {new_fact}")
            else:
                print(f"❌ Central Brain: Could not correct memory (fact not found).")

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
        paths, chunks, del_files, del_chunks, backfilled = sync_brain()
        if is_json:
            print(json.dumps({"status": "success", "command": "sync", "data": {"synced_paths": paths, "total_chunks": chunks, "purged_files": del_files, "purged_chunks": del_chunks, "backfilled_embeddings": backfilled}}, indent=2))
        else:
            msg = f"🔄 Central Brain Sync Complete: Processed {paths} sources ({chunks} total chunks active)."
            if del_files > 0:
                msg += f" Purged {del_files} deleted files ({del_chunks} orphan chunks removed)."
            if backfilled > 0:
                msg += f" Backfilled {backfilled} missing vector embeddings."
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
