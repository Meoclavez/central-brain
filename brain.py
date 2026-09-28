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
from urllib.parse import unquote
import re
import subprocess
import platform
import difflib
from pathlib import Path
from datetime import datetime, timezone

try:  # Optional accelerators; Central Brain stays stdlib-only when they are missing.
    import numpy as np
except Exception:
    np = None
try:
    import yaml
except Exception:
    yaml = None

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
OKF_DIR = BRAIN_DIR / "okf"

BRAIN_VERSION = "2.5.1"
OKF_VERSION = "0.2"
OKF_PRODUCER = f"central-brain/{BRAIN_VERSION}"
OKF_BUILD_REV = "5"  # bump when bundle rendering changes so existing bundles regenerate

OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
DEFAULT_EMBED_MODEL = "mxbai-embed-large"
EMBED_BATCH_SIZE = 32
_OLLAMA_UNREACHABLE = False

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
        migrate_schema_v24(conn)

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

def migrate_schema_v24(conn):
    """Additive v2.4 schema: facts FTS5, fact vectors, OKF concepts, and metadata.
    Only creates new objects, so older brain.py versions keep working on the same DB."""
    has_facts_fts = conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'facts_fts'").fetchone()
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
            entity, category, fact, content='facts', content_rowid='id'
        )
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
            INSERT INTO facts_fts(rowid, entity, category, fact) VALUES (new.id, new.entity, new.category, new.fact);
        END;
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
            INSERT INTO facts_fts(facts_fts, rowid, entity, category, fact) VALUES('delete', old.id, old.entity, old.category, old.fact);
        END;
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
            INSERT INTO facts_fts(facts_fts, rowid, entity, category, fact) VALUES('delete', old.id, old.entity, old.category, old.fact);
            INSERT INTO facts_fts(rowid, entity, category, fact) VALUES (new.id, new.entity, new.category, new.fact);
        END;
    """)
    if not has_facts_fts:
        conn.execute("INSERT INTO facts_fts(facts_fts) VALUES('rebuild');")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS fact_vectors (
            fact_id INTEGER PRIMARY KEY,
            hash TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS concepts (
            key TEXT PRIMARY KEY,
            origin TEXT NOT NULL,
            bundle TEXT NOT NULL,
            concept_id TEXT NOT NULL,
            file_path TEXT,
            type TEXT,
            title TEXT,
            description TEXT,
            tags TEXT,
            status TEXT,
            stale_after TEXT,
            trust TEXT,
            generated_at TEXT,
            resource TEXT,
            links TEXT,
            entities TEXT,
            fact_ids TEXT,
            card TEXT,
            card_hash TEXT,
            embedding BLOB,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_concepts_file ON concepts(file_path);")
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS concepts_fts USING fts5(
            key UNINDEXED, title, description, tags, body
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS okf_meta (
            slug TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE TABLE IF NOT EXISTS brain_meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fact_env (
            fact_id INTEGER PRIMARY KEY,
            kernel TEXT,
            nvidia TEXT,
            source TEXT,
            recorded_at TEXT
        )
    """)

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
    global _OLLAMA_UNREACHABLE
    if not texts:
        return []
    if _OLLAMA_UNREACHABLE:
        return [None] * len(texts)
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
                if isinstance(e, urllib.error.URLError) and not isinstance(e, urllib.error.HTTPError):
                    _OLLAMA_UNREACHABLE = True  # daemon down: skip further attempts in this process
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

def batch_cosine(query_vec: list[float], blobs: list) -> list[float]:
    """Cosine similarity of one query against many stored vector blobs (numpy-accelerated when available)."""
    if not query_vec or not blobs:
        return [0.0] * len(blobs)
    dim = len(query_vec)
    if np is not None:
        q = np.asarray(query_vec, dtype=np.float32)
        qn = float(np.linalg.norm(q)) or 1.0
        out = []
        for blob in blobs:
            if isinstance(blob, (bytes, bytearray, memoryview)) and len(blob) == dim * 4:
                v = np.frombuffer(blob, dtype=np.float32)
            else:
                dec = decode_vector_blob(blob)
                if len(dec) != dim:
                    out.append(0.0)
                    continue
                v = np.asarray(dec, dtype=np.float32)
            vn = float(np.linalg.norm(v))
            out.append(float(q @ v) / (qn * vn) if vn else 0.0)
        return out
    return [cosine_similarity(query_vec, decode_vector_blob(b)) for b in blobs]

FTS_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at", "for", "with", "by", "from", "as",
    "is", "are", "was", "were", "be", "been", "it", "its", "this", "that", "these", "those", "there",
    "i", "me", "my", "we", "our", "you", "your", "do", "does", "did", "how", "what", "which", "who", "whom",
    "why", "when", "where", "can", "could", "should", "would", "will", "shall", "may", "might", "must",
    "not", "no", "yes", "so", "if", "then", "than", "into", "about", "up", "out", "now", "get", "use", "using"
}

def build_fts_query(text: str) -> str | None:
    """Converts free text into an FTS5 OR-query of prefix terms (BM25 ranks docs matching more/rarer terms higher).
    Plain AND matching (the old behaviour) returned nothing for most natural-language questions."""
    if not text:
        return None
    tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
    terms = []
    for t in tokens:
        if len(t) < 2 or t in FTS_STOPWORDS or t in terms:
            continue
        terms.append(t)
    if not terms:
        return None
    return " OR ".join(f'"{t}"*' for t in terms[:16])

def slugify(text: str) -> str:
    """Filesystem- and URL-safe concept slug ('Realtek RTL8852BE Wi-Fi' -> 'realtek-rtl8852be-wi-fi')."""
    s = re.sub(r"[^a-z0-9]+", "-", str(text or "").lower()).strip("-")
    return s[:80] or "general"

def to_iso_utc(ts) -> str | None:
    """Normalizes SQLite CURRENT_TIMESTAMP (UTC) or ISO strings to ISO 8601 with an explicit UTC offset."""
    if not ts:
        return None
    if isinstance(ts, datetime):
        dt = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    raw = str(ts).strip()
    try:
        if re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", raw):
            dt = datetime.strptime(raw[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        else:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.astimezone()  # naive isoformat() values were written in local time
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return None

def parse_iso(ts) -> datetime | None:
    iso = to_iso_utc(ts)
    if not iso:
        return None
    return datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

FRONTMATTER_RE = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL)

def _mini_yaml_scalar(raw: str):
    raw = raw.strip()
    if raw == "":
        return None
    if raw[0] == '"' and raw[-1] == '"' and len(raw) >= 2:
        try:
            return json.loads(raw)
        except Exception:
            return raw[1:-1]
    if raw[0] == "'" and raw[-1] == "'" and len(raw) >= 2:
        return raw[1:-1].replace("''", "'")
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        return [_mini_yaml_scalar(x) for x in _split_flow(inner)] if inner else []
    if raw.startswith("{") and raw.endswith("}"):
        out = {}
        for part in _split_flow(raw[1:-1]):
            if ":" in part:
                k, v = part.split(":", 1)
                out[k.strip()] = _mini_yaml_scalar(v)
        return out
    low = raw.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "~"):
        return None
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)
    if re.fullmatch(r"-?\d+\.\d+", raw):
        return float(raw)
    return raw

def _split_flow(text: str) -> list[str]:
    parts, depth, cur, quote, escaped = [], 0, [], None, False
    for ch in text:
        if quote:
            cur.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\" and quote == '"':
                escaped = True
            elif ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
            continue
        cur.append(ch)
    if "".join(cur).strip():
        parts.append("".join(cur).strip())
    return parts

def _mini_yaml_load(text: str) -> dict:
    """Minimal YAML subset loader (OKF frontmatter shapes) used when PyYAML is unavailable:
    scalars, flow lists/maps, block lists of scalars or maps, and one level of nested maps."""
    root = {}
    lines = [l.rstrip() for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#")]
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^([A-Za-z0-9_\-]+):\s*(.*)$", line)
        if not m or line.startswith(" "):
            i += 1
            continue
        key, rest = m.group(1), m.group(2)
        i += 1
        if rest.strip():
            root[key] = _mini_yaml_scalar(rest)
            continue
        block = []
        while i < len(lines) and (lines[i].startswith(" ") or lines[i].startswith("-")):
            block.append(lines[i])
            i += 1
        if block and block[0].lstrip().startswith("-"):
            items, current = [], None
            for b in block:
                st = b.strip()
                if st.startswith("- ") or st == "-":
                    if current is not None:
                        items.append(current)
                    body = st[1:].strip()
                    km = re.match(r"^([A-Za-z0-9_\-]+):\s*(.*)$", body)
                    if km and not body.startswith(("{", "[", "\"", "'")):
                        current = {km.group(1): _mini_yaml_scalar(km.group(2))}
                    else:
                        current = _mini_yaml_scalar(body)
                else:
                    km = re.match(r"^([A-Za-z0-9_\-]+):\s*(.*)$", st)
                    if km and isinstance(current, dict):
                        current[km.group(1)] = _mini_yaml_scalar(km.group(2))
            if current is not None:
                items.append(current)
            root[key] = items
        else:
            sub = {}
            for b in block:
                km = re.match(r"^\s+([A-Za-z0-9_\-]+):\s*(.*)$", b)
                if km:
                    sub[km.group(1)] = _mini_yaml_scalar(km.group(2))
            root[key] = sub
    return root

def _normalize_yaml_values(obj):
    if isinstance(obj, dict):
        return {str(k): _normalize_yaml_values(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_yaml_values(v) for v in obj]
    if isinstance(obj, datetime):
        return to_iso_utc(obj)
    if hasattr(obj, "isoformat") and not isinstance(obj, str):
        return obj.isoformat()
    return obj

def parse_frontmatter(content: str) -> tuple[dict | None, str]:
    """Splits a markdown document into (frontmatter dict or None, body)."""
    if not content or not content.startswith("---"):
        return None, content
    m = FRONTMATTER_RE.match(content)
    if not m:
        return None, content
    raw = m.group(1)
    meta = None
    if yaml is not None:
        try:
            loaded = yaml.safe_load(raw)
            meta = loaded if isinstance(loaded, dict) else None
        except Exception:
            meta = None
    if meta is None:
        try:
            meta = _mini_yaml_load(raw)
        except Exception:
            meta = None
    if meta is None:
        return None, content
    return _normalize_yaml_values(meta), content[m.end():]

def _yaml_scalar(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v)
    if re.fullmatch(r"[-+]?[0-9][0-9_.:eE+-]*", s):
        return json.dumps(s)
    if s == "" or re.search(r"[:#\[\]{},&*!|>'\"%@`]|^[-?\s]|\s$", s) or s.lower() in ("true", "false", "null", "yes", "no", "~"):
        return json.dumps(s, ensure_ascii=False)
    return s

def dump_frontmatter(meta: dict) -> str:
    """Deterministic YAML emitter for OKF frontmatter (flow style for short maps/lists)."""
    out = ["---"]
    for key, val in meta.items():
        if val is None or val == [] or val == {}:
            continue
        if isinstance(val, dict):
            out.append(f"{key}: {{ " + ", ".join(f"{k}: {_yaml_scalar(v)}" for k, v in val.items()) + " }")
        elif isinstance(val, list) and val and all(isinstance(x, dict) for x in val):
            out.append(f"{key}:")
            for item in val:
                if len(item) <= 2:
                    out.append("  - { " + ", ".join(f"{k}: {_yaml_scalar(v)}" for k, v in item.items()) + " }")
                else:
                    first = True
                    for k, v in item.items():
                        out.append(f"  {'- ' if first else '  '}{k}: {_yaml_scalar(v)}")
                        first = False
        elif isinstance(val, list):
            out.append(f"{key}: [" + ", ".join(_yaml_scalar(x) for x in val) + "]")
        else:
            out.append(f"{key}: {_yaml_scalar(val)}")
    out.append("---")
    return "\n".join(out) + "\n"

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

def is_generated_okf_path(path: Path) -> bool:
    """True for files inside the generated OKF bundle (~/.central_brain/okf), which mirrors the facts store."""
    try:
        Path(path).resolve().relative_to(OKF_DIR.resolve())
        return True
    except ValueError:
        return False

def is_okf_concept(file_path: Path, meta: dict | None) -> bool:
    return bool(meta) and bool(str(meta.get("type") or "").strip()) and file_path.name not in ("index.md", "log.md")

def ingest_file(file_path: Path):
    file_path = file_path.resolve()
    if not file_path.exists() or file_path.suffix.lower() not in ['.md', '.txt', '.json', '.conf', '.sh']:
        return 0
    if is_generated_okf_path(file_path):
        return 0

    try:
        content = file_path.read_text(encoding='utf-8', errors='ignore')
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return 0

    conn = get_db()
    str_path = str(file_path)
    file_content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()

    meta, body = parse_frontmatter(content) if file_path.suffix.lower() == ".md" else (None, content)
    okf_concept = is_okf_concept(file_path, meta)

    existing = conn.execute("SELECT COUNT(*), hash FROM chunks WHERE file_path = ?", (str_path,)).fetchall()
    if existing and existing[0][0] > 0:
        first_hash = existing[0][1] or ""
        if first_hash.startswith(f"{file_content_hash}:"):
            null_embeds = conn.execute("SELECT COUNT(*) FROM chunks WHERE file_path = ? AND embedding IS NULL", (str_path,)).fetchone()[0]
            if null_embeds == 0:
                if okf_concept and not conn.execute("SELECT 1 FROM concepts WHERE key = ?", (f"file:{str_path}",)).fetchone():
                    with conn:
                        register_file_concept(conn, file_path, meta, body)
                return existing[0][0]

    if okf_concept:
        # OKF concept: index the body only (YAML is noise for BM25/vectors) and scope breadcrumbs by title.
        title = str(meta.get("title") or file_path.stem)
        chunks = []
        for idx, (header, text) in enumerate(chunk_markdown(body, str_path)):
            scoped = title if header == file_path.name else f"{title} > {header}"
            if idx == 0 and meta.get("description"):
                text = f"{meta['description']}\n\n{text}"
            chunks.append((scoped, text))
        if not chunks and meta.get("description"):
            chunks = [(title, str(meta["description"]))]
    elif meta and meta.get("okf_version"):
        chunks = chunk_markdown(body, str_path)  # OKF bundle-root index.md: skip the version frontmatter
    else:
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
            # Path is part of the key: identical files in two places (copies, shared requirements.txt)
            # must not collide on the UNIQUE hash column. Prefix stays the file hash for skip checks.
            chunk_hash = f"{file_content_hash}:{idx}:{hashlib.sha256((str_path + chr(0) + chunk_text).encode('utf-8')).hexdigest()[:16]}"
            vec_blob = encode_vector_blob(vec) if vec else None
            conn.execute(
                "INSERT INTO chunks (file_path, header, content, embedding, hash, updated_at) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
                (str_path, header, chunk_text, vec_blob, chunk_hash)
            )
            ingested_count += 1
        if okf_concept:
            register_file_concept(conn, file_path, meta, body)
        else:
            conn.execute("DELETE FROM concepts WHERE key = ?", (f"file:{str_path}",))
            conn.execute("DELETE FROM concepts_fts WHERE key = ?", (f"file:{str_path}",))

    return ingested_count

DIR_INGEST_SUFFIXES = ('.md', '.txt', '.conf', '.sh')
DIR_INGEST_MAX_BYTES = 512 * 1024
# Dependency, build, cache, and VCS folders never hold project knowledge.
DIR_INGEST_SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "env", "__pycache__", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".tox", "dist", "build", "target", ".gradle", ".dart_tool", ".idea",
    ".vscode", "site-packages", "vendor", ".next", ".nuxt", ".cache", "coverage", ".terraform", "Pods",
    "DerivedData", "graphify-out", ".pub-cache", "models", "weights", "checkpoints",
}
DIR_INGEST_KEEP_HIDDEN = {".agents", ".planning"}

def _dir_ingest_allowed(path: Path, root: Path) -> bool:
    try:
        rel_parts = path.relative_to(root).parts[:-1]
    except ValueError:
        return False
    for part in rel_parts:
        if part in DIR_INGEST_SKIP_DIRS or (part.startswith(".") and part not in DIR_INGEST_KEEP_HIDDEN):
            return False
    if not path.name.endswith(DIR_INGEST_SUFFIXES):
        return False
    try:
        return path.is_file() and path.stat().st_size <= DIR_INGEST_MAX_BYTES
    except OSError:
        return False

def iter_source_files(dir_path: Path) -> list[Path]:
    """Knowledge files under a directory source. A folder with its own .git is enumerated with
    `git ls-files --exclude-standard` (honours .gitignore); other folders are walked with
    DIR_INGEST_SKIP_DIRS pruned. .agents/ and .planning/ are always included, even when gitignored."""
    dir_path = Path(dir_path).resolve()
    found = set()
    listed_by_git = False
    if (dir_path / ".git").exists():
        try:
            proc = subprocess.run(["git", "-C", str(dir_path), "ls-files", "-co", "--exclude-standard", "-z"],
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=30)
            if proc.returncode == 0:
                listed_by_git = True
                for rel in proc.stdout.decode("utf-8", "ignore").split("\0"):
                    if rel:
                        found.add((dir_path / rel).resolve())
        except Exception:
            listed_by_git = False
    if not listed_by_git:
        for root, dirs, files in os.walk(dir_path):
            dirs[:] = [d for d in dirs if d not in DIR_INGEST_SKIP_DIRS and (not d.startswith(".") or d in DIR_INGEST_KEEP_HIDDEN)]
            for f in files:
                found.add(Path(root) / f)
    else:
        for keep in DIR_INGEST_KEEP_HIDDEN:
            for root, dirs, files in os.walk(dir_path):
                dirs[:] = [d for d in dirs if d not in DIR_INGEST_SKIP_DIRS and (not d.startswith(".") or d == keep)]
                if keep in Path(root).relative_to(dir_path).parts:
                    for f in files:
                        found.add(Path(root) / f)
    return sorted(p for p in found if _dir_ingest_allowed(p, dir_path))

def ingest_directory(dir_path: Path):
    total = 0
    for fp in iter_source_files(dir_path):
        total += ingest_file(fp)
    return total

def sync_directory_source(dir_path: Path, sources: list[str]) -> tuple[int, int, int]:
    """Ingests a directory source, then drops chunks under it for files the source no longer covers
    (gitignored, excluded folder, oversized). Files registered separately, or under another registered
    directory nested inside this one, are left alone. Returns (chunks, excluded_files, excluded_chunks)."""
    dir_path = Path(dir_path).resolve()
    wanted = iter_source_files(dir_path)
    cnt = sum(ingest_file(fp) for fp in wanted)
    keep = {str(fp) for fp in wanted}
    nested = []
    for s in sources:
        sp = Path(s).expanduser()
        if not sp.exists():
            continue
        sp = sp.resolve()
        if sp.is_file():
            keep.add(str(sp))
        elif sp != dir_path and str(sp).startswith(str(dir_path) + "/"):
            nested.append(str(sp))
    ex_files, ex_chunks = purge_path_chunks(dir_path, keep=keep, keep_prefixes=nested)
    return cnt, ex_files, ex_chunks

def purge_path_chunks(path: Path | str, keep: set[str] = None, keep_prefixes: list[str] = None) -> tuple[int, int]:
    """Deletes chunks (and OKF file concepts) for a file or everything under a directory,
    except paths in `keep` or under `keep_prefixes`. Returns (files_purged, chunks_purged)."""
    p = str(Path(path).expanduser().resolve())
    keep = keep or set()
    prefixes = [x.rstrip("/") + "/" for x in (keep_prefixes or [])]
    conn = get_db()
    rows = conn.execute("SELECT file_path, COUNT(*) FROM chunks WHERE file_path = ? OR file_path LIKE ? GROUP BY file_path",
                        (p, p.rstrip("/") + "/%")).fetchall()
    files = chunks = 0
    with conn:
        for fp, n in rows:
            if fp in keep or any(fp.startswith(pre) for pre in prefixes):
                continue
            conn.execute("DELETE FROM chunks WHERE file_path = ?", (fp,))
            conn.execute("DELETE FROM concepts WHERE key = ?", (f"file:{fp}",))
            conn.execute("DELETE FROM concepts_fts WHERE key = ?", (f"file:{fp}",))
            files += 1
            chunks += n
    return files, chunks

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

ENTITY_SUFFIX_WORDS = {"fix", "fixes", "issue", "issues", "bug", "bugs", "config", "configuration", "setup", "notes"}

def entity_key(name: str) -> str:
    """Slug with trailing generic words removed ('RTL8852BE Bluetooth Fix' ~ 'RTL8852BE Bluetooth')."""
    toks = slugify(name).split("-")
    while len(toks) > 1 and toks[-1] in ENTITY_SUFFIX_WORDS:
        toks.pop()
    return "-".join(toks)

def resolve_entity(conn, entity: str) -> tuple[str, str, list[str]]:
    """Maps a requested entity onto an existing one to stop spelling/variant sprawl.
    Returns (entity_to_use, how, suggestions); how is 'existing', 'normalized', 'snapped', or 'new'.
    Auto-snaps only on slug equality, suffix-word equality, or a near-typo (ratio >= 0.92)."""
    rows = conn.execute("SELECT entity, COUNT(*) FROM facts GROUP BY entity").fetchall()
    counts = {r[0]: r[1] for r in rows if r[0]}
    if entity in counts:
        return entity, "existing", []
    slug, key = slugify(entity), entity_key(entity)
    same_slug = [e for e in counts if slugify(e) == slug]
    if same_slug:
        return max(same_slug, key=lambda e: counts[e]), "normalized", []
    same_key = [e for e in counts if entity_key(e) == key]
    if same_key:
        return max(same_key, key=lambda e: counts[e]), "snapped", []
    scored = []
    for e in counts:
        es = slugify(e)
        ratio = difflib.SequenceMatcher(None, slug, es).ratio()
        if ratio >= 0.92:
            scored.append((2.0 + ratio, e))
            continue
        a, b = slug.split("-"), es.split("-")
        lead = 0
        for x, y in zip(a, b):
            if x != y:
                break
            lead += 1
        shared = {t for t in a if len(t) >= 3 and t not in ENTITY_SUFFIX_WORDS} & {t for t in b if len(t) >= 3}
        if ratio >= 0.72 or lead >= 2 or len(shared) >= 2 or (len(es) >= 5 and (slug.startswith(es + "-") or es.startswith(slug + "-"))):
            scored.append((ratio + 0.1 * lead + 0.1 * len(shared), e))
    scored.sort(key=lambda x: (-x[0], -counts[x[1]]))
    if scored and scored[0][0] >= 2.0:
        return scored[0][1], "snapped", []
    return entity, "new", [e for _, e in scored[:3]]

def current_env() -> dict:
    """Running kernel and NVIDIA driver versions (NVIDIA None when the module is not loaded)."""
    nvidia = None
    try:
        nvidia = Path("/sys/module/nvidia/version").read_text().strip() or None
    except Exception:
        try:
            m = re.search(r"Kernel Module.*?\s(\d+\.\d+(?:\.\d+)?)\s", Path("/proc/driver/nvidia/version").read_text())
            nvidia = m.group(1) if m else None
        except Exception:
            nvidia = None
    return {"kernel": platform.release(), "nvidia": nvidia}

def record_fact_env(conn, fact_id: int, source: str = "live", env: dict = None):
    env = env or current_env()
    conn.execute("INSERT OR REPLACE INTO fact_env (fact_id, kernel, nvidia, source, recorded_at) VALUES (?, ?, ?, ?, ?)",
                 (fact_id, env.get("kernel"), env.get("nvidia"), source, datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")))

PACMAN_LOG = Path("/var/log/pacman.log")

def pacman_version_timeline(log_path: Path = None) -> dict:
    """{package: [(utc_datetime, version), ...]} for kernel and NVIDIA packages from pacman.log."""
    log_path = log_path or PACMAN_LOG
    pkgs = {"linux", "linux-lts", "linux-zen", "linux-hardened", "nvidia", "nvidia-open", "nvidia-dkms", "nvidia-open-dkms", "nvidia-lts", "nvidia-utils"}
    out = {}
    rx = re.compile(r"^\[(\S+)\] \[ALPM\] (installed|upgraded|downgraded|reinstalled) (\S+) \((?:\S+ -> )?(\S+)\)")
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = rx.match(line)
                if not m or m.group(3) not in pkgs:
                    continue
                try:
                    dt = datetime.fromisoformat(m.group(1).replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.astimezone()
                except Exception:
                    continue
                out.setdefault(m.group(3), []).append((dt.astimezone(timezone.utc), m.group(4)))
    except Exception:
        return {}
    return out

def _pkg_to_uname(version: str) -> str:
    """'7.1.8.arch1-1' -> '7.1.8-arch1-1' (uname -r form); other flavours keep their pkgver."""
    return re.sub(r"\.(arch|lts|zen|hardened)(\d)", r"-\1\2", version, count=1)

def backfill_fact_env(conn=None) -> int:
    """Infers kernel/NVIDIA versions for facts recorded before env tagging, from pacman.log upgrade history
    (the package version installed at the fact's timestamp; source='pacman-log')."""
    conn = conn or get_db()
    missing = conn.execute("SELECT f.id, f.timestamp FROM facts f LEFT JOIN fact_env e ON e.fact_id = f.id WHERE e.fact_id IS NULL").fetchall()
    if not missing:
        return 0
    tl = pacman_version_timeline()
    if not tl:
        return 0
    rel = platform.release()
    kpkg = "linux-lts" if "-lts" in rel else "linux-zen" if "-zen" in rel else "linux-hardened" if "-hardened" in rel else "linux"
    npkg = next((p for p in ("nvidia-open", "nvidia", "nvidia-open-dkms", "nvidia-dkms", "nvidia-lts", "nvidia-utils") if p in tl), None)

    def at(pkg, when):
        best = None
        for dt, ver in tl.get(pkg, []):
            if dt <= when:
                best = ver
            else:
                break
        return best

    done = 0
    with conn:
        for fid, ts in missing:
            when = parse_iso(ts)
            if not when:
                continue
            k = at(kpkg, when)
            if not k:
                continue
            n = at(npkg, when) if npkg else None
            record_fact_env(conn, fid, source="pacman-log",
                            env={"kernel": _pkg_to_uname(k), "nvidia": re.sub(r"-\d+$", "", n) if n else None})
            done += 1
    return done

# Kernel-drift relevance: only kernel/driver machinery counts. Deliberately excluded: sysctl keys such as
# `kernel.yama.*`, "module" in the Python sense, user-space audio (pipewire/wireplumber/bluez) and desktop
# words (kde, plasma, wayland, sddm) — those facts do not change meaning with the kernel series.
SYSTEM_STRONG_RE = re.compile(
    r"\b(kernel(?!\.[a-z])|drivers?|modprobe|kernel[- ]modules?|udev(?:adm)?|firmware|dkms|sysfs|acpi|initramfs|"
    r"mkinitcpio|grub|btusb|btmtk|mt7921e?|rtw89\w*|rfkill|asusd|asus-wmi|xhci|pcie|d3cold|s2idle|"
    r"nvidia-open|nvidia-dkms)\b|/sys/\S+|/proc/cmdline|\b[0-9a-f]{4}:[0-9a-f]{4}\b|\b\d+\.\d+\.\d+-arch\d|\blinux[- ]?\d+\.\d+", re.I)
# NVIDIA-drift relevance: the fact is about the NVIDIA driver/stack itself, not a mention of NVIDIA hardware.
NVIDIA_FACT_RE = re.compile(r"nvidia[- ]?(open|driver|module|utils|drm|smi|dkms)|nvidia\b.{0,80}\b(drivers?|kernel|egl|gsp|prime|wayland|x11)\b|"
                            r"\bgsp\b|libEGL|GL\.nvidia", re.I)
SYSTEM_FACT_RE = SYSTEM_STRONG_RE  # kept for the bundle's kernel annotation

def is_system_fact(text: str, in_project: bool = False) -> bool:
    """Kernel-drift eligible: >= 1 kernel/driver signal (>= 2 for facts filed under a project)."""
    strong = {m.group(0).lower() for m in SYSTEM_STRONG_RE.finditer(text or "")}
    return len(strong) >= (2 if in_project else 1)

def _series(version: str | None, parts: int) -> str | None:
    if not version:
        return None
    nums = re.findall(r"\d+", version)
    return ".".join(nums[:parts]) if len(nums) >= parts else None

def env_drift(fact_text: str, env_row, now_env: dict, in_project: bool = False) -> str | None:
    """Warning when a host fact was recorded on another kernel series (major.minor), or an NVIDIA-driver fact
    on another NVIDIA major. The two checks have separate relevance tests (is_system_fact / NVIDIA_FACT_RE)."""
    if not env_row:
        return None
    text = fact_text or ""
    kernel_ok = is_system_fact(text, in_project)
    nv_ok = bool(NVIDIA_FACT_RE.search(text))
    if not (kernel_ok or nv_ok):
        return None
    kernel, nvidia = env_row["kernel"], env_row["nvidia"]
    notes = []
    if kernel_ok and kernel and now_env.get("kernel") and _series(kernel, 2) != _series(now_env["kernel"], 2):
        notes.append(f"kernel {kernel.split('-')[0]}→{now_env['kernel'].split('-')[0]}")
    if nv_ok and nvidia and now_env.get("nvidia") and _series(nvidia, 1) != _series(now_env["nvidia"], 1):
        notes.append(f"NVIDIA {nvidia}→{now_env['nvidia']}")
    return ("recorded on " + ", ".join(notes)) if notes else None

def fact_drift_map(conn, facts: list[dict], now_env: dict = None) -> dict:
    """{fact_id: warning} for the given fact dicts (needs 'id' and 'fact')."""
    if not facts:
        return {}
    now_env = now_env or current_env()
    ids = [f["id"] for f in facts]
    rows = {r["fact_id"]: r for r in conn.execute(
        f"SELECT fact_id, kernel, nvidia FROM fact_env WHERE fact_id IN ({','.join('?' * len(ids))})", ids).fetchall()}
    project_ids = set()
    for r in conn.execute("SELECT fact_ids FROM concepts WHERE origin = 'facts' AND concept_id LIKE 'projects/%'").fetchall():
        project_ids |= set(json.loads(r[0] or "[]"))
    out = {}
    for f in facts:
        w = env_drift(f.get("fact") or f.get("text") or "", rows.get(f["id"]), now_env, in_project=f["id"] in project_ids)
        if w:
            out[f["id"]] = w
    return out

def reverify_facts(fact_ids: list[int]) -> list[int]:
    """Marks facts as re-verified on the running kernel/driver without changing their text."""
    conn = get_db()
    done = []
    with conn:
        for fid in fact_ids:
            if conn.execute("SELECT 1 FROM facts WHERE id = ?", (fid,)).fetchone():
                record_fact_env(conn, fid, source="reverified")
                done.append(fid)
    if done:
        env = current_env()
        append_episode(f"**[Reverified]** Facts {', '.join('#' + str(i) for i in done)} re-verified on kernel {env['kernel']}"
                       + (f", NVIDIA {env['nvidia']}" if env.get("nvidia") else ""))
        okf_refresh_quiet()
    return done

def append_episode(line: str):
    today_str = datetime.now().strftime("%Y-%m-%d")
    today_file = EPISODES_DIR / f"{today_str}.md"
    mode = "a" if today_file.exists() else "w"
    with open(today_file, mode, encoding="utf-8") as f:
        if mode == "w":
            f.write(f"# Agent Episode Log - {today_str}\n\n")
        f.write(f"- [{datetime.now().strftime('%H:%M:%S')}] {line}\n")
    ingest_file(today_file)

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

def remember(fact: str, entity: str = None, category: str = "Knowledge", source: str = "CLI", tags: list[str] = None,
             exact_entity: bool = False) -> dict:
    """Saves a structured fact to SQLite, syncs facts.json, and appends to today's episode file.
    If entity is None or 'General', auto-resolves to current project name if inside a project directory.
    The entity is then mapped onto an existing spelling/variant (see resolve_entity) unless exact_entity.
    Records the running kernel/NVIDIA versions with the fact. Returns a dict describing what was stored."""
    ensure_dirs()
    cat_map = {"fix": "Fix", "rule": "Rule", "knowledge": "Knowledge", "project": "Project"}
    category = cat_map.get(str(category).lower(), str(category).capitalize() if category else "Knowledge")

    # Smart Entity Resolution: If entity is empty or default 'General', detect active project
    if not entity or entity.strip().lower() in ["general", "none", ""]:
        st = get_project_state(Path.cwd())
        if st.get("project_path") and Path(st["project_path"]) != Path.home():
            entity = Path(st["project_path"]).name
        else:
            entity = "General"
    else:
        entity = entity.strip()

    # Append tags to fact if provided
    if tags:
        tag_tokens = []
        for t in tags:
            for part in str(t).split(","):
                part = part.strip().lstrip("#")
                if part and f"#{part}" not in fact:
                    tag_tokens.append(f"#{part}")
        if tag_tokens:
            fact = f"{fact} {' '.join(tag_tokens)}"

    conn = get_db()
    requested = entity
    how, suggestions = "existing", []
    if not exact_entity:
        entity, how, suggestions = resolve_entity(conn, entity)
    with conn:
        cur = conn.execute(
            "INSERT INTO facts (entity, category, fact, source) VALUES (?, ?, ?, ?)",
            (entity, category, fact, source)
        )
        fact_id = cur.lastrowid
        record_fact_env(conn, fact_id)
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
    okf_refresh_quiet()
    deprecated = None
    try:
        row = conn.execute("SELECT concept_id, status FROM concepts WHERE origin = 'facts' AND entities LIKE ?",
                           (f'%{json.dumps(entity)[1:-1]}%',)).fetchone()
        if row and row["status"] == "deprecated":
            deprecated = row["concept_id"]
    except Exception:
        pass
    return {"id": fact_id, "entity": entity, "requested_entity": requested, "resolution": how,
            "suggestions": suggestions, "category": category, "fact": fact, "deprecated_concept": deprecated}

def forget(target: str = None, entity: str = None, fact_id: int = None, category: str = None):
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
        else:
            conds = []
            params = []
            if target:
                conds.append("(fact LIKE ? OR entity LIKE ?)")
                params.extend([f"%{target}%", f"%{target}%"])
            if entity and entity != "General":
                conds.append("entity = ? COLLATE NOCASE")
                params.append(entity)
            if category:
                conds.append("category = ? COLLATE NOCASE")
                params.append(category)
            if not conds:
                return 0
            where_sql = " AND ".join(conds)
            cur = conn.execute(f"DELETE FROM facts WHERE {where_sql}", params)
            deleted_count = cur.rowcount
            purged_info = f"{deleted_count} fact(s) matching '{target or '*'}' (entity: {entity or 'any'}, category: {category or 'any'})"

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
    okf_refresh_quiet()
    return deleted_count

def correct(entity: str = None, new_fact: str = None, old_fact_search: str = None, category: str = "Fix", source: str = "CLI", fact_id: int = None):
    """Corrects/supersedes an existing memory with a new finding by ID or entity search."""
    conn = get_db()
    cat_map = {"fix": "Fix", "rule": "Rule", "knowledge": "Knowledge", "project": "Project"}
    # None keeps an existing fact's category when correcting by ID (previously this silently reset it to Fix).
    category = cat_map.get(str(category).lower(), str(category).capitalize()) if category else None

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
            record_fact_env(conn, fact_id)
        elif old_fact_search:
            category = category or "Fix"
            conn.execute("DELETE FROM facts WHERE entity = ? AND fact LIKE ?", (entity, f"%{old_fact_search}%"))
            cur = conn.execute(
                "INSERT INTO facts (entity, category, fact, source) VALUES (?, ?, ?, ?)",
                (entity, category, new_fact, source)
            )
            record_fact_env(conn, cur.lastrowid)
        else:
            category = category or "Fix"
            conn.execute("DELETE FROM facts WHERE entity = ? AND category = ?", (entity, category))
            cur = conn.execute(
                "INSERT INTO facts (entity, category, fact, source) VALUES (?, ?, ?, ?)",
                (entity, category, new_fact, source)
            )
            record_fact_env(conn, cur.lastrowid)

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
    okf_refresh_quiet()
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

FACT_SIM_FLOOR = 0.52      # cosine floor for a vector-only fact match (mxbai-embed-large; unrelated text sits ~0.35-0.48)
FACT_RELATIVE_FLOOR = 0.55  # drop facts scoring below this fraction of the best fact

def fact_embed_text(entity: str, fact: str) -> str:
    return f"{entity or 'General'}: {fact or ''}"

def ensure_fact_vectors(conn=None, force_check: bool = True) -> int:
    """Embeds facts that have no (or an outdated) vector. Returns the number of facts embedded.
    Hash-keyed, so corrections made by any brain version are re-embedded on the next query or sync."""
    conn = conn or get_db()
    rows = conn.execute("SELECT id, entity, fact FROM facts").fetchall()
    have = {r[0]: r[1] for r in conn.execute("SELECT fact_id, hash FROM fact_vectors").fetchall()}
    live_ids = set()
    todo = []
    for r in rows:
        live_ids.add(r["id"])
        h = hashlib.sha1(fact_embed_text(r["entity"], r["fact"]).encode("utf-8")).hexdigest()
        if have.get(r["id"]) != h:
            todo.append((r["id"], h, fact_embed_text(r["entity"], r["fact"])))
    stale = [fid for fid in have if fid not in live_ids]
    if stale:
        with conn:
            conn.executemany("DELETE FROM fact_vectors WHERE fact_id = ?", [(fid,) for fid in stale])
    embedded = 0
    for i in range(0, len(todo), EMBED_BATCH_SIZE):
        batch = todo[i:i + EMBED_BATCH_SIZE]
        vecs = get_embeddings_batch([t[2] for t in batch])
        if not any(vecs):
            break
        with conn:
            for (fid, h, _), vec in zip(batch, vecs):
                if vec:
                    conn.execute("INSERT OR REPLACE INTO fact_vectors (fact_id, hash, embedding) VALUES (?, ?, ?)",
                                 (fid, h, encode_vector_blob(vec)))
                    embedded += 1
    return embedded

def rank_facts(conn, query: str, query_vec: list[float] | None, conds: list[str], params: list, limit: int) -> list[dict]:
    """Hybrid fact ranking: 0.7 * vector cosine + 0.3 * BM25 reciprocal rank + exact-phrase bonus.
    conds/params are SQL filters over alias 'f' (the facts table)."""
    where_extra = (" AND " + " AND ".join(conds)) if conds else ""
    kw_scores = {}
    fts_q = build_fts_query(query)
    if fts_q:
        try:
            rows = conn.execute(
                f"SELECT f.id FROM facts_fts JOIN facts f ON f.id = facts_fts.rowid "
                f"WHERE facts_fts MATCH ?{where_extra} ORDER BY bm25(facts_fts) LIMIT 50",
                [fts_q, *params]
            ).fetchall()
            for rank_idx, r in enumerate(rows):
                kw_scores[r[0]] = 1.0 / (rank_idx + 1)
        except sqlite3.Error:
            pass

    phrase_ids = {r[0] for r in conn.execute(
        f"SELECT f.id FROM facts f WHERE (f.fact LIKE ? OR f.entity LIKE ?){where_extra}",
        [f"%{query}%", f"%{query}%", *params]
    ).fetchall()}

    vec_scores = {}
    if query_vec:
        ensure_fact_vectors(conn)
        rows = conn.execute(
            f"SELECT f.id, v.embedding FROM facts f JOIN fact_vectors v ON v.fact_id = f.id WHERE 1=1{where_extra}",
            params
        ).fetchall()
        for r, sim in zip(rows, batch_cosine(query_vec, [r[1] for r in rows])):
            vec_scores[r[0]] = sim

    scored = []
    for fid in set(kw_scores) | set(phrase_ids) | set(vec_scores):
        sim = vec_scores.get(fid, 0.0)
        kw = kw_scores.get(fid, 0.0)
        phrase = fid in phrase_ids
        if query_vec and not phrase and sim < FACT_SIM_FLOOR and kw < 0.2:
            continue
        score = 0.7 * sim + 0.3 * kw + (0.15 if phrase else 0.0)
        if not query_vec:
            score = kw + (0.5 if phrase else 0.0)
        scored.append((score, fid))
    if not scored:
        return []
    scored.sort(reverse=True)
    best = scored[0][0]
    keep = [(sc, fid) for sc, fid in scored if sc >= best * FACT_RELATIVE_FLOOR][:limit]
    out = []
    for sc, fid in keep:
        row = conn.execute("SELECT id, entity, category, fact, source, timestamp FROM facts WHERE id = ?", (fid,)).fetchone()
        if row:
            out.append({**dict(row), "score": round(sc, 4)})
    return out

def search_brain(query: str, top_k: int = 5, entity: str = None, category: str = None,
                 source: str = None, since: str = None, until: str = None, path_filter: str = None,
                 facts_only: bool = False, chunks_only: bool = False, query_vec: list[float] = None):
    """
    Recency-weighted Hybrid Search across Vectors, FTS5 Keywords, and Structured Facts.
    Supports multi-field precision filtering and selective retrieval (facts_only / chunks_only).
    Pass query_vec to reuse an embedding computed by the caller (one Ollama call per search).
    """
    conn = get_db()
    cat_map = {"fix": "Fix", "rule": "Rule", "knowledge": "Knowledge", "project": "Project"}
    if category:
        category = cat_map.get(str(category).lower(), category)

    if query and query_vec is None:
        query_vec = get_embedding(query)

    # 1. Dense Vector & FTS5 Search (skipped if facts_only)
    top_docs = []
    if not facts_only:
        vector_results = []
        if query_vec:
            sql = "SELECT id, file_path, header, content, embedding, updated_at FROM chunks WHERE embedding IS NOT NULL"
            params = []
            if path_filter:
                sql += " AND file_path LIKE ?"
                params.append(f"%{path_filter}%")
            rows = conn.execute(sql, params).fetchall()
            sims = batch_cosine(query_vec, [r['embedding'] for r in rows])
            for r, sim in zip(rows, sims):
                d = dict(r)
                d.pop("embedding", None)
                vector_results.append((sim, d))
            vector_results.sort(key=lambda x: x[0], reverse=True)

        # 2. FTS5 Keyword Search (OR of prefix terms, BM25-ranked)
        fts_results = {}
        fts_q = build_fts_query(query)
        if fts_q:
            try:
                sql = "SELECT rowid as id FROM chunks_fts WHERE chunks_fts MATCH ?"
                params = [fts_q]
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

    # 4. Structured Facts Search with Precision Filtering (skipped if chunks_only)
    facts_rows = []
    if not chunks_only:
        fact_conditions = []
        fact_params = []
        if entity:
            fact_conditions.append("f.entity = ? COLLATE NOCASE")
            fact_params.append(entity)
        if category:
            fact_conditions.append("f.category = ? COLLATE NOCASE")
            fact_params.append(category)
        if source:
            fact_conditions.append("f.source = ? COLLATE NOCASE")
            fact_params.append(source)
        if since:
            fact_conditions.append("f.timestamp >= ?")
            fact_params.append(since)
        if until:
            fact_conditions.append("f.timestamp <= ?")
            fact_params.append(until)

        if query:
            facts_rows = rank_facts(conn, query, query_vec, fact_conditions, fact_params, top_k)
        else:
            where_sql = " AND ".join(fact_conditions) if fact_conditions else "1=1"
            facts_rows = [dict(r) for r in conn.execute(
                f"SELECT f.id, f.entity, f.category, f.fact, f.source, f.timestamp FROM facts f WHERE {where_sql} ORDER BY f.id DESC LIMIT ?",
                (*fact_params, top_k)
            ).fetchall()]

    if facts_rows:
        drift = fact_drift_map(conn, facts_rows)
        for f in facts_rows:
            if f["id"] in drift:
                f["env_warning"] = drift[f["id"]]

    return {
        "chunks": [{"score": round(score, 4), **{k: v for k, v in doc.items() if k != 'embedding'}} for score, doc in top_docs],
        "facts": facts_rows
    }

# ============================================================================
# OKF (Open Knowledge Format v0.2) Knowledge Bundle & Quick Map
#   Producer: compiles the facts store + project maps into ~/.central_brain/okf/
#             (index.md progressive disclosure, one concept per entity, log.md).
#   Consumer: any registered source containing OKF concepts (frontmatter with
#             `type:`) is parsed into the concepts table with trust/lifecycle.
#   Quick map: concept-level routing (OKF) fused with chunk/fact retrieval (RAG).
# ============================================================================

FACT_SECTIONS = [("Rule", "Rules"), ("Fix", "Fixes"), ("Knowledge", "Knowledge"), ("Project", "Project Log")]
QM_W_CARD, QM_W_FTS, QM_W_EVIDENCE = 0.35, 0.15, 0.5  # concept score = card cosine + concept BM25 + best fact/chunk evidence
QM_MIN_SCORE = 0.45  # absolute concept floor (measured: real queries top out >= 0.7, off-topic queries <= 0.38)
QUICKMAP_OTHER_FACT_FLOOR = 0.55
QM_DOC_FLOOR = 0.55  # chunk hybrid score floor for evidence/documents (off-topic chunks measured <= 0.50)  # facts outside the chosen concepts must clear this hybrid score to be listed
GENERIC_GROUP_TOKENS = {"the", "new", "fix", "fixes", "general", "system", "project", "universal", "standalone", "multi"}

def okf_trust_tier(verified) -> str:
    """OKF §5.3: no verified => unverified; human:<id> actor => human-reviewed; else machine-confirmed."""
    if not verified:
        return "unverified"
    items = verified if isinstance(verified, list) else [verified]
    actors = [str(v.get("by", "")) for v in items if isinstance(v, dict)]
    if any(a.startswith("human:") for a in actors):
        return "human-reviewed"
    return "machine-confirmed" if actors else "unverified"

def okf_is_stale(stale_after, now: datetime = None) -> bool:
    dt = parse_iso(stale_after)
    return bool(dt) and (now or datetime.now(timezone.utc)) >= dt

def load_sources_list() -> list[str]:
    if SOURCES_PATH.exists():
        try:
            with open(SOURCES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [str(x) for x in data] if isinstance(data, list) else []
        except Exception:
            return []
    return []

def get_okf_meta(conn) -> dict:
    out = {}
    for r in conn.execute("SELECT slug, data FROM okf_meta").fetchall():
        try:
            out[r[0]] = json.loads(r[1])
        except Exception:
            pass
    return out

def set_okf_meta(conn, slug: str, updates: dict) -> dict:
    row = conn.execute("SELECT data FROM okf_meta WHERE slug = ?", (slug,)).fetchone()
    data = json.loads(row[0]) if row else {}
    for k, v in updates.items():
        if v is None:
            continue
        if v == "":
            data.pop(k, None)
        else:
            data[k] = v
    with conn:
        conn.execute("INSERT OR REPLACE INTO okf_meta (slug, data, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
                     (slug, json.dumps(data, sort_keys=True)))
    return data

def discover_projects() -> dict:
    """Projects known to Central Brain: registered project maps / .planning dirs plus first-level
    folders of ~/Projects and ~/Projects-1. Returns {slug: {name, path, map_file, readme}}."""
    candidates = []
    registered = set()
    brain_root = BRAIN_DIR.resolve()
    for src in load_sources_list():
        sp = Path(src)
        if sp.name == "project_map.md" and sp.parent.name == ".agents":
            candidates.append(sp.parent.parent)
        elif sp.name == ".planning":
            candidates.append(sp.parent)
        elif sp.is_dir():
            spr = sp.resolve()
            inside_brain = spr == brain_root or str(spr).startswith(str(brain_root) + "/")
            is_project = any((spr / m).exists() for m in (".git", "README.md", "readme.md", ".agents", "package.json", "pyproject.toml"))
            if not inside_brain and spr != Path.home() and is_project:
                candidates.append(spr)
                registered.add(str(spr))
    for base in (Path.home() / "Projects", Path.home() / "Projects-1"):
        if base.is_dir():
            try:
                candidates.extend(sorted(d for d in base.iterdir() if d.is_dir() and not d.name.startswith(".")))
            except Exception:
                pass
    projects = {}
    for d in candidates:
        if d == Path.home() or not d.is_dir():
            continue
        slug = slugify(d.name)
        if slug in projects:
            continue
        map_file = d / ".agents" / "project_map.md"
        planning_state = d / ".planning" / "STATE.md"
        readme = next((d / n for n in ("README.md", "readme.md") if (d / n).is_file()), None)
        projects[slug] = {
            "name": d.name,
            "path": str(d.resolve()),
            "registered": str(d.resolve()) in registered,
            "map_file": str(map_file) if map_file.is_file() else (str(planning_state) if planning_state.is_file() else None),
            "readme": str(readme) if readme else None,
        }
    return projects

def okf_short_description(text: str, limit: int = 160) -> str:
    """One-line summary: a leading 'Title:' clause if present, else the first sentence, word-truncated."""
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    t = re.sub(r"\s#[A-Za-z][\w-]*", "", t).strip()
    cand = None
    m = re.match(r"^([^:]{12,90}):\s", t)
    if m:
        head = m.group(1).strip()
        balanced = head.count("(") == head.count(")") and head.count("'") % 2 == 0 and head.count('"') % 2 == 0
        if balanced and not re.search(r"https?$", head):
            cand = head
    if cand is None:
        masked = re.sub(r"\b(e\.g|i\.e|etc|vs|approx|incl|No)\.", lambda mm: mm.group(0).replace(".", "\x00"), t)
        m2 = re.search(r"(?<=[A-Za-z0-9\)\]'\"`])[.;]\s", masked[20:])
        cand = t[:20 + m2.start() + 1] if m2 else t
    cand = cand.rstrip(";,: ")
    if len(cand) > limit:
        cand = cand[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "…"
    return cand

def project_doc_description(proj: dict) -> str | None:
    """First prose paragraph of the project map or README (hard-wrapped lines are joined)."""
    for key in ("map_file", "readme"):
        fp = proj.get(key)
        if not fp:
            continue
        try:
            _, body = parse_frontmatter(Path(fp).read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        in_code = False
        para = []
        for line in body.splitlines() + [""]:
            st = line.strip()
            if st.startswith("```"):
                in_code = not in_code
                continue
            skip = (in_code or not st or st.startswith(("#", "|", "<", "![", "---", "[!"))
                    or re.match(r"^[-*]?\s*\*\*[^*]{1,40}:\*\*", st) is not None
                    or re.match(r"^([-*+]|\d+\.)\s", st) is not None)
            if skip:
                text = " ".join(para).strip()
                if len(text) >= 25:
                    return okf_short_description(text, 180)
                para = []
                continue
            para.append(st.lstrip("> ").strip())
    return None

def project_doc_excerpt(proj: dict, limit: int = 1400) -> str:
    """Leading text of the project map or README, used as the concept card when a project has no facts."""
    for key in ("map_file", "readme"):
        fp = proj.get(key)
        if fp:
            try:
                _, body = parse_frontmatter(Path(fp).read_text(encoding="utf-8", errors="ignore"))
                text = re.sub(r"<[^>]+>|!\[[^\]]*\]\([^)]*\)", "", body)
                return re.sub(r"\n{3,}", "\n\n", text).strip()[:limit]
            except Exception:
                continue
    return ""

def okf_entity_group(entity_slug: str, projects: dict, override: str = None) -> tuple[str, str]:
    """Places an entity: ('project', proj_slug) when it is/extends a known project, else ('topic', '')."""
    if override:
        ov = slugify(override)
        return ("project", ov) if ov in projects else ("topic", ov)
    if entity_slug in projects:
        return "project", entity_slug
    best, best_score = None, 0
    etoks = entity_slug.split("-")
    for ps in projects:
        ptoks = ps.split("-")
        if len(ps) >= 6 and entity_slug.startswith(ps + "-"):
            score = 100 + len(ps)
        else:
            common = 0
            for a, b in zip(etoks, ptoks):
                if a != b:
                    break
                common += 1
            score = common if (common >= 2 and len(ptoks) >= 2 and len(etoks[0]) >= 3) else 0
        if score > best_score:
            best, best_score = ps, score
    return ("project", best) if best else ("topic", "")

def okf_signature(facts: list, meta: dict, projects: dict) -> str:
    h = hashlib.sha256()
    h.update(f"{BRAIN_VERSION}|{OKF_BUILD_REV}".encode())
    for f in facts:
        h.update(f"{f['id']}|{f['entity']}|{f['category']}|{f['timestamp']}|{f['fact']}|{f.get('kernel')}|{f.get('nvidia')}\n".encode("utf-8", "ignore"))
    h.update(json.dumps(meta, sort_keys=True).encode())
    for slug in sorted(projects):
        p = projects[slug]
        mt = 0
        for key in ("map_file", "readme"):
            if p.get(key):
                try:
                    mt = max(mt, int(Path(p[key]).stat().st_mtime))
                except Exception:
                    pass
        h.update(f"{slug}|{p['path']}|{p.get('map_file')}|{mt}\n".encode())
    return h.hexdigest()

def get_brain_meta(conn, key: str) -> str | None:
    row = conn.execute("SELECT value FROM brain_meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None

def set_brain_meta(conn, key: str, value: str):
    with conn:
        conn.execute("INSERT OR REPLACE INTO brain_meta (key, value) VALUES (?, ?)", (key, value))

def upsert_concept(conn, row: dict, fts_body: str):
    """Inserts/updates a concept row; clears its embedding when the card text changed."""
    card_hash = hashlib.sha1((row.get("card") or "").encode("utf-8")).hexdigest()
    prev = conn.execute("SELECT card_hash, embedding FROM concepts WHERE key = ?", (row["key"],)).fetchone()
    embedding = prev["embedding"] if (prev and prev["card_hash"] == card_hash) else None
    conn.execute("""
        INSERT OR REPLACE INTO concepts (key, origin, bundle, concept_id, file_path, type, title, description, tags,
            status, stale_after, trust, generated_at, resource, links, entities, fact_ids, card, card_hash, embedding, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (row["key"], row["origin"], row["bundle"], row["concept_id"], row.get("file_path"), row.get("type"),
          row.get("title"), row.get("description"), json.dumps(row.get("tags") or []), row.get("status") or "stable",
          row.get("stale_after"), row.get("trust") or "unverified", row.get("generated_at"), row.get("resource"),
          json.dumps(row.get("links") or []), json.dumps(row.get("entities") or []), json.dumps(row.get("fact_ids") or []),
          row.get("card"), card_hash, embedding))
    conn.execute("DELETE FROM concepts_fts WHERE key = ?", (row["key"],))
    conn.execute("INSERT INTO concepts_fts (key, title, description, tags, body) VALUES (?, ?, ?, ?, ?)",
                 (row["key"], row.get("title") or "", row.get("description") or "", " ".join(row.get("tags") or []), fts_body or ""))

def find_okf_bundle_root(file_path: Path) -> Path:
    """Nearest ancestor whose index.md declares okf_version; falls back to the file's directory."""
    curr = file_path.parent
    for _ in range(8):
        idx = curr / "index.md"
        if idx.is_file():
            try:
                meta, _ = parse_frontmatter(idx.read_text(encoding="utf-8", errors="ignore")[:2000])
                if meta and meta.get("okf_version"):
                    return curr
            except Exception:
                pass
        if curr == curr.parent or curr == Path.home():
            break
        curr = curr.parent
    return file_path.parent

MD_LINK_RE = re.compile(r"\]\(([^)\s]+?\.md)(?:#[^)]*)?\)")

def register_file_concept(conn, file_path: Path, meta: dict, body: str):
    """Records an authored/external OKF concept (consumer side). Caller manages the transaction."""
    bundle = find_okf_bundle_root(file_path)
    try:
        concept_id = str(file_path.relative_to(bundle).with_suffix(""))
    except ValueError:
        concept_id = file_path.stem
    links = []
    for target in MD_LINK_RE.findall(body or ""):
        if re.match(r"^[a-z]+://", target):
            continue
        resolved = (bundle / target.lstrip("/")) if target.startswith("/") else (file_path.parent / target)
        key = f"file:{resolved.resolve()}"
        if key not in links:
            links.append(key)
    gen = meta.get("generated") if isinstance(meta.get("generated"), dict) else {}
    tags = meta.get("tags") if isinstance(meta.get("tags"), list) else ([meta["tags"]] if meta.get("tags") else [])
    title = str(meta.get("title") or file_path.stem.replace("-", " ").replace("_", " "))
    description = str(meta.get("description") or okf_short_description(body, 160))
    card = f"{title}\n{meta.get('type')}\n{description}\ntags: {', '.join(map(str, tags))}\n\n{(body or '')[:1400]}"
    upsert_concept(conn, {
        "key": f"file:{file_path}", "origin": "file", "bundle": str(bundle), "concept_id": concept_id,
        "file_path": str(file_path), "type": str(meta.get("type")), "title": title, "description": description,
        "tags": [str(t) for t in tags], "status": str(meta.get("status") or "stable"),
        "stale_after": to_iso_utc(meta.get("stale_after")), "trust": okf_trust_tier(meta.get("verified")),
        "generated_at": to_iso_utc(gen.get("at") or meta.get("timestamp")), "resource": meta.get("resource"),
        "links": links, "card": card,
    }, fts_body=body or "")

def _fact_date(ts) -> str:
    iso = to_iso_utc(ts)
    return iso[:10] if iso else str(ts or "")[:10]

def okf_render_concept(c: dict) -> str:
    fm = {
        "type": c["type"],
        "title": c["title"],
        "description": c["description"],
        "resource": c.get("resource"),
        "tags": c.get("tags"),
    }
    if c.get("status") and c["status"] != "stable":
        fm["status"] = c["status"]
    fm["stale_after"] = c.get("stale_after")
    fm["generated"] = {"by": OKF_PRODUCER, "at": c.get("generated_at")} if c.get("generated_at") else {"by": OKF_PRODUCER}
    verified = c.get("verified") or []
    if verified:
        fm["verified"] = verified if len(verified) > 1 else verified[0]
    fm["sources"] = c.get("sources")
    fm["entities"] = c.get("entities")
    fm["fact_ids"] = c.get("fact_ids")
    lines = [dump_frontmatter(fm)]
    lines.append(f"# {c['title']}\n")
    if c.get("summary"):
        lines.append(c["summary"].strip() + "\n")
    buckets = {}
    for f in c.get("facts", []):
        sec = next((name for cat, name in FACT_SECTIONS if str(f["category"]).lower() == cat.lower()), "Other Notes")
        buckets.setdefault(sec, []).append(f)
    for sec in [name for _, name in FACT_SECTIONS] + ["Other Notes"]:
        if sec not in buckets:
            continue
        lines.append(f"## {sec}\n")
        for f in buckets[sec]:
            extra = f" [{f['category']}]" if sec == "Other Notes" else ""
            ent = f" ({f['entity']})" if len(c.get("entities") or []) > 1 else ""
            envtag = ""
            if f.get("kernel") and is_system_fact(str(f["fact"]), c["concept_id"].startswith("projects/")):
                envtag = f" (kernel {f['kernel'].split('-')[0]}" + (f", NVIDIA {f['nvidia']}" if f.get("nvidia") and NVIDIA_FACT_RE.search(str(f["fact"])) else "") + ")"
            lines.append(f"- [#{f['id']}] {_fact_date(f['timestamp'])}{extra}{ent}{envtag} · {' '.join(str(f['fact']).split())}")
        lines.append("")
    if c.get("map_outline"):
        lines.append("## Project Map\n")
        lines.append(f"Source: `{c['map_file']}`. Read one section with `brain map show \"{c['project_path']}\" -s \"<Section>\"`.\n")
        lines.extend(f"- {t}" for t in c["map_outline"])
        lines.append("")
    if c.get("children"):
        lines.append("## Concepts In This Project\n")
        lines.extend(f"* [{t}](/{cid}.md) - {d}" for t, cid, d in c["children"])
        lines.append("")
    if c.get("related"):
        lines.append("## Related\n")
        lines.extend(f"* [{t}](/{cid}.md) - {d}" for t, cid, d in c["related"])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"

def okf_build(embed: bool = True, force: bool = False) -> dict:
    """Compiles facts + project maps into an OKF v0.2 bundle at OKF_DIR and refreshes the concepts table.
    Skips all work when the facts/meta/project signature is unchanged (cheap to call before every read)."""
    ensure_dirs()
    conn = get_db()
    facts = [dict(r) for r in conn.execute(
        "SELECT f.id, f.entity, f.category, f.fact, f.source, f.timestamp, e.kernel, e.nvidia FROM facts f "
        "LEFT JOIN fact_env e ON e.fact_id = f.id ORDER BY f.id").fetchall()]
    meta = get_okf_meta(conn)
    projects = discover_projects()
    signature = okf_signature(facts, meta, projects)
    stats = {"path": str(OKF_DIR), "skipped": False, "written": 0, "removed": 0, "embedded": 0}

    if not force and get_brain_meta(conn, "okf_signature") == signature and (OKF_DIR / "index.md").exists():
        stats["skipped"] = True
    else:
        # 1. Group facts into entity concepts (case/punctuation variants of an entity merge by slug).
        by_slug = {}
        for f in facts:
            slug = slugify(f["entity"] or "General")
            by_slug.setdefault(slug, []).append(f)

        placement = {}
        for slug in by_slug:
            kind, group = okf_entity_group(slug, projects, (meta.get(slug) or {}).get("group"))
            placement[slug] = (kind, group)

        # Topic sub-grouping by leading token when >= 2 topic entities share it.
        token_counts = {}
        for slug, (kind, group) in placement.items():
            if kind == "topic" and not group:
                tok = slug.split("-")[0]
                token_counts[tok] = token_counts.get(tok, 0) + 1
        for slug, (kind, group) in list(placement.items()):
            if kind == "topic" and not group:
                tok = slug.split("-")[0]
                if token_counts.get(tok, 0) >= 2 and len(tok) >= 3 and tok not in GENERIC_GROUP_TOKENS:
                    placement[slug] = ("topic", tok)

        concepts = {}
        # 2. Project overview concepts (a project with facts, child concepts, or a map/state file).
        project_children = {}
        for slug, (kind, group) in placement.items():
            if kind == "project":
                project_children.setdefault(group, []).append(slug)
        for ps, proj in projects.items():
            if ps not in project_children and not proj.get("map_file") and not proj.get("registered"):
                continue
            own = by_slug.get(ps, [])
            cid = f"projects/{ps}/overview"
            concepts[cid] = {"concept_id": cid, "slug": ps, "type": "Project", "title": proj["name"], "facts": own,
                             "entities": sorted({f["entity"] for f in own}), "project": proj}

        for slug, (kind, group) in placement.items():
            if kind == "project" and slug == group:
                continue
            entity_facts = by_slug[slug]
            if kind == "project":
                cid = f"projects/{group}/{slug}"
            elif group:
                cid = f"topics/{group}/{slug}"
            else:
                cid = f"topics/{slug}"
            names = {}
            for f in entity_facts:
                names[f["entity"]] = names.get(f["entity"], 0) + 1
            title = max(names.items(), key=lambda kv: kv[1])[0]
            concepts[cid] = {"concept_id": cid, "slug": slug, "type": "Topic", "title": title,
                             "facts": entity_facts, "entities": sorted(names), "parent_project": group if kind == "project" else None}

        # 3. Metadata, descriptions, sources, trust.
        for cid, c in concepts.items():
            m = meta.get(c["slug"]) or {}
            c["facts"] = sorted(c["facts"], key=lambda f: (to_iso_utc(f["timestamp"]) or "", f["id"]), reverse=True)
            c["fact_ids"] = [f["id"] for f in c["facts"]]
            latest = to_iso_utc(c["facts"][0]["timestamp"]) if c["facts"] else None
            proj = c.get("project")
            if proj and not latest:
                for key in ("map_file", "readme"):
                    if proj.get(key):
                        try:
                            latest = to_iso_utc(datetime.fromtimestamp(Path(proj[key]).stat().st_mtime, tz=timezone.utc))
                        except Exception:
                            pass
                        break
            c["generated_at"] = latest
            if m.get("type"):
                c["type"] = m["type"]
            desc = m.get("description")
            if not desc and proj:
                desc = project_doc_description(proj)
            if not desc and c["facts"]:
                desc = okf_short_description(c["facts"][0]["fact"])
            c["description"] = desc or f"{c['title']} ({c['type']})"
            tag_counts = {}
            for f in c["facts"]:
                for t in re.findall(r"(?<![\w&])#([A-Za-z][\w-]{1,30})", str(f["fact"])):
                    tag_counts[t.lower()] = tag_counts.get(t.lower(), 0) + 1
            tags = [t for t, _ in sorted(tag_counts.items(), key=lambda kv: (-kv[1], kv[0]))][:8]
            if not tags:
                tags = sorted({str(f["category"]).lower() for f in c["facts"]})
            c["tags"] = tags
            c["status"] = m.get("status") or "stable"
            c["stale_after"] = m.get("stale_after")
            c["verified"] = m.get("verified") or []
            c["trust"] = okf_trust_tier(c["verified"])
            sources = []
            if c["facts"]:
                ents = ", ".join(json.dumps(e, ensure_ascii=False) for e in c["entities"])
                sources.append({"id": "facts", "resource": f"central-brain facts where entity in [{ents}]",
                                "title": f"Central Brain facts store ({len(c['facts'])} facts)", "last_modified": latest})
            if proj:
                c["resource"] = Path(proj["path"]).as_uri()
                c["project_path"] = proj["path"]
                for key, sid, stitle in (("map_file", "project-map", "Project map"), ("readme", "readme", "Project README")):
                    if proj.get(key):
                        try:
                            lm = to_iso_utc(datetime.fromtimestamp(Path(proj[key]).stat().st_mtime, tz=timezone.utc))
                        except Exception:
                            lm = None
                        sources.append({"id": sid, "resource": proj[key], "title": stitle, "last_modified": lm})
                if proj.get("map_file"):
                    try:
                        _, outline = extract_markdown_section(Path(proj["map_file"]).read_text(encoding="utf-8", errors="ignore"), "__none__")
                        c["map_outline"] = outline[:40]
                        c["map_file"] = proj["map_file"]
                    except Exception:
                        pass
            c["sources"] = sources

        # 4. Graph edges: parent/child and mentions of other concepts' titles in fact text.
        by_slug_cid = {c["slug"]: cid for cid, c in concepts.items()}
        name_index = []
        for cid, c in concepts.items():
            for name in {c["title"], *c["entities"]}:
                if len(name) >= 5 and name.lower() not in ("general", "system"):
                    name_index.append((re.compile(r"(?<![\w/.~-])" + re.escape(name.lower()) + r"(?![\w/-])"), cid))
        for cid, c in concepts.items():
            text = " ".join(str(f["fact"]).lower() for f in c["facts"])
            counts = {}
            for rx, other in name_index:
                if other != cid:
                    n = len(rx.findall(text))
                    if n:
                        counts[other] = counts.get(other, 0) + n
            parent = c.get("parent_project")
            if parent and f"projects/{parent}/overview" in concepts:
                counts[f"projects/{parent}/overview"] = counts.get(f"projects/{parent}/overview", 0) + 1000
            ranked = sorted(counts.items(), key=lambda kv: -kv[1])[:8]
            c["related"] = [(concepts[o]["title"], o, concepts[o]["description"]) for o, _ in ranked]
            if c["type"] == "Project" or c["concept_id"].endswith("/overview"):
                kids = [k for k in concepts if k.startswith(c["concept_id"].rsplit("/", 1)[0] + "/") and k != cid]
                c["children"] = [(concepts[k]["title"], k, concepts[k]["description"]) for k in sorted(kids)]
                c["related"] = [r for r in c["related"] if r[1] not in kids]
            if c.get("project"):
                c["summary"] = f"{c['description']}\n\nProject root: `{c['project_path']}`"

        # 5. Render files: concepts, per-directory index.md, root index.md, log.md.
        files = {}
        for cid, c in concepts.items():
            files[f"{cid}.md"] = okf_render_concept(c)

        dirs = {}
        for cid in concepts:
            parts = cid.split("/")
            for depth in range(1, len(parts)):
                dirs.setdefault("/".join(parts[:depth]), set())
            dirs.setdefault("/".join(parts[:-1]), set()).add(cid)

        def dir_stats(d):
            ids = [k for k in concepts if k.startswith(d + "/")]
            return len(ids), sum(len(concepts[k]["facts"]) for k in ids)

        def dir_title(d):
            leaf = d.split("/")[-1]
            if d.startswith("projects/") and leaf in projects:
                return projects[leaf]["name"]
            if d.count("/"):
                for k in sorted(concepts):
                    if k.startswith(d + "/"):
                        word = re.split(r"[\s_\-]+", concepts[k]["title"])[0]
                        if slugify(word) == leaf:
                            return word
                return leaf.replace("-", " ").title()
            return leaf.title()

        for d in sorted(dirs):
            depth_children = sorted({k.split("/")[d.count("/") + 1] for k in concepts if k.startswith(d + "/") and k.count("/") > d.count("/") + 1})
            out = [f"# {dir_title(d)}\n"]
            if depth_children:
                out.append("## Groups\n" if d == "topics" else "## Directories\n")
                entries = []
                for sub in depth_children:
                    sd = f"{d}/{sub}"
                    n_c, n_f = dir_stats(sd)
                    ov = concepts.get(f"{sd}/overview")
                    desc = ov["description"] if ov else ", ".join(sorted(concepts[k]["title"] for k in concepts if k.startswith(sd + "/"))[:4])
                    latest = max((concepts[k]["generated_at"] or "" for k in concepts if k.startswith(sd + "/")), default="")
                    entries.append((latest, f"* [{dir_title(sd)}]({sub}/) - {desc} ({n_c} concepts, {n_f} facts)"))
                out.extend(e for _, e in sorted(entries, key=lambda x: x[0], reverse=True))
                out.append("")
            direct = sorted(dirs[d], key=lambda k: (concepts[k]["type"] != "Project", -(len(concepts[k]["facts"])), concepts[k]["title"].lower()))
            if direct:
                out.append("## Concepts\n")
                for k in direct:
                    c = concepts[k]
                    flag = f" [{c['status']}]" if c["status"] != "stable" else ""
                    out.append(f"* [{c['title']}]({k.split('/')[-1]}.md) - {c['description']}{flag} ({len(c['facts'])} facts)")
                out.append("")
            files[f"{d}/index.md"] = "\n".join(out).rstrip() + "\n"

        latest_all = max((c["generated_at"] or "" for c in concepts.values()), default="")
        n_proj = len({k.split("/")[1] for k in concepts if k.startswith("projects/")})
        n_topic = sum(1 for k in concepts if k.startswith("topics/"))
        root = [dump_frontmatter({"okf_version": OKF_VERSION}),
                "# Central Brain Knowledge Bundle\n",
                f"OKF v{OKF_VERSION} bundle compiled by {OKF_PRODUCER} from {len(facts)} verified facts into {len(concepts)} concepts "
                f"(latest change {latest_all[:10] or 'n/a'}). Concepts carry `[#id]` fact references for `brain correct --id` / `brain forget --id`.\n",
                "## Sections\n",
                f"* [Projects](projects/) - {n_proj} projects with their maps, decisions and fixes",
                f"* [Topics](topics/) - {n_topic} system, hardware and domain topics",
                "* [Update Log](log.md) - chronological history of fact changes\n",
                "## Recently Updated\n"]
        for k in sorted(concepts, key=lambda k: concepts[k]["generated_at"] or "", reverse=True)[:10]:
            c = concepts[k]
            root.append(f"* [{c['title']}]({k}.md) - {c['description']}")
        files["index.md"] = "\n".join(root).rstrip() + "\n"

        cid_by_entity_slug = {c["slug"]: cid for cid, c in concepts.items()}
        for cid, c in concepts.items():
            for e in c["entities"]:
                cid_by_entity_slug.setdefault(slugify(e), cid)
        log_lines = ["# Central Brain Update Log\n"]
        cutoff = datetime.now(timezone.utc).timestamp() - 90 * 86400
        by_day = {}
        for f in sorted(facts, key=lambda f: (to_iso_utc(f["timestamp"]) or "", f["id"]), reverse=True):
            dt = parse_iso(f["timestamp"])
            if not dt or dt.timestamp() < cutoff:
                continue
            by_day.setdefault(dt.strftime("%Y-%m-%d"), []).append(f)
        for day in sorted(by_day, reverse=True):
            log_lines.append(f"## {day}")
            for f in by_day[day]:
                cid = cid_by_entity_slug.get(slugify(f["entity"] or "General"))
                ref = f"[{concepts[cid]['title']}](/{cid}.md)" if cid else f["entity"]
                log_lines.append(f"* **{f['category']}**: {ref} [#{f['id']}] {okf_short_description(f['fact'], 110)}")
            log_lines.append("")
        files["log.md"] = "\n".join(log_lines).rstrip() + "\n"

        # 6. Write changed files; remove files we generated earlier that no longer exist.
        OKF_DIR.mkdir(parents=True, exist_ok=True)
        for rel, content in files.items():
            fp = OKF_DIR / rel
            try:
                if fp.exists() and fp.read_text(encoding="utf-8") == content:
                    continue
            except Exception:
                pass
            fp.parent.mkdir(parents=True, exist_ok=True)
            tmp = fp.with_suffix(f".tmp.{os.getpid()}")
            tmp.write_text(content, encoding="utf-8")
            os.replace(tmp, fp)
            stats["written"] += 1
        for fp in sorted(OKF_DIR.rglob("*.md"), reverse=True):
            rel = str(fp.relative_to(OKF_DIR))
            if rel in files:
                continue
            try:
                head = fp.read_text(encoding="utf-8", errors="ignore")[:1500]
            except Exception:
                continue
            m, _ = parse_frontmatter(head + "\n")
            generated_by = str(((m or {}).get("generated") or {}).get("by", "")) if isinstance((m or {}).get("generated"), dict) else ""
            if fp.name in ("index.md", "log.md") or generated_by.startswith("central-brain/"):
                fp.unlink()
                stats["removed"] += 1
        for d in sorted((p for p in OKF_DIR.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
            try:
                d.rmdir()
            except OSError:
                pass

        # 7. Concepts table (origin='facts').
        with conn:
            keep = set()
            for cid, c in concepts.items():
                key = f"okf:{cid}"
                keep.add(key)
                fact_text = "\n".join(f"[{f['category']}] {f['fact']}" for f in c["facts"])
                if c.get("project") and not c["facts"]:
                    fact_text = project_doc_excerpt(c["project"])
                card = (f"{c['title']}\n{c['type']}\n{c['description']}\nentities: {', '.join(c['entities'])}\n"
                        f"tags: {', '.join(c['tags'])}\n\n{fact_text}")[:1800]
                links = [f"okf:{o}" for _, o, _ in c.get("related", [])] + [f"okf:{o}" for _, o, _ in c.get("children", [])]
                upsert_concept(conn, {
                    "key": key, "origin": "facts", "bundle": str(OKF_DIR), "concept_id": cid,
                    "file_path": str(OKF_DIR / f"{cid}.md"), "type": c["type"], "title": c["title"],
                    "description": c["description"], "tags": c["tags"], "status": c["status"],
                    "stale_after": c["stale_after"], "trust": c["trust"], "generated_at": c["generated_at"],
                    "resource": c.get("resource"), "links": links, "entities": c["entities"],
                    "fact_ids": c["fact_ids"], "card": card,
                }, fts_body=fact_text + "\n" + "\n".join(c.get("map_outline") or []))
            for (old_key,) in conn.execute("SELECT key FROM concepts WHERE origin = 'facts'").fetchall():
                if old_key not in keep:
                    conn.execute("DELETE FROM concepts WHERE key = ?", (old_key,))
                    conn.execute("DELETE FROM concepts_fts WHERE key = ?", (old_key,))
        set_brain_meta(conn, "okf_signature", signature)
        set_brain_meta(conn, "okf_built_at", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))

    if embed:
        stats["embedded"] = ensure_concept_embeddings(conn)
    row = conn.execute("SELECT COUNT(*), SUM(origin = 'facts'), SUM(origin = 'file') FROM concepts").fetchone()
    stats.update({"concepts": row[0] or 0, "generated_concepts": row[1] or 0, "file_concepts": row[2] or 0})
    return stats

def okf_refresh_quiet():
    """Keeps the on-disk bundle current after fact mutations; never fails the calling command."""
    try:
        okf_build(embed=False)
    except Exception as e:
        print(f"[Brain Warning] OKF bundle refresh failed: {e}", file=sys.stderr)

def ensure_concept_embeddings(conn=None) -> int:
    conn = conn or get_db()
    rows = conn.execute("SELECT key, card FROM concepts WHERE embedding IS NULL AND card IS NOT NULL").fetchall()
    done = 0
    for i in range(0, len(rows), EMBED_BATCH_SIZE):
        batch = rows[i:i + EMBED_BATCH_SIZE]
        vecs = get_embeddings_batch([r["card"] for r in batch])
        if not any(vecs):
            break
        with conn:
            for r, v in zip(batch, vecs):
                if v:
                    conn.execute("UPDATE concepts SET embedding = ? WHERE key = ?", (encode_vector_blob(v), r["key"]))
                    done += 1
    return done

def okf_resolve(conn, target: str) -> tuple[list[dict], list[str]]:
    """Resolves a concept by key, concept ID, slug, title, entity name, or a directory/group prefix.
    Returns (matching concept rows, suggestions)."""
    rows = [dict(r) for r in conn.execute("SELECT * FROM concepts").fetchall()]
    t = str(target or "").strip().strip("/")
    if t.endswith(".md"):
        t = t[:-3]
    tl = t.lower()
    ts = slugify(t)
    for pred in (
        lambda r: r["key"].lower() == tl or r["concept_id"].lower() == tl,
        lambda r: r["concept_id"].lower().split("/")[-1] == tl or r["concept_id"].lower().split("/")[-1] == ts,
        lambda r: (r["title"] or "").lower() == tl or slugify(r["title"] or "") == ts,
        lambda r: any(slugify(e) == ts for e in json.loads(r["entities"] or "[]")),
        lambda r: r["concept_id"] == f"projects/{ts}/overview",
    ):
        hits = [r for r in rows if pred(r)]
        if hits:
            return hits, []
    group = [r for r in rows if r["concept_id"].lower().startswith(tl + "/") or r["concept_id"].lower().startswith(f"topics/{ts}/")
             or r["concept_id"].lower().startswith(f"projects/{ts}/")]
    if group:
        return group, []
    titles = {r["title"]: r for r in rows if r["title"]}
    sugg = difflib.get_close_matches(t, list(titles), n=5, cutoff=0.5)
    sugg += [r["title"] for r in rows if tl and tl in (r["title"] or "").lower() and r["title"] not in sugg][:5]
    return [], sugg[:6]

def concept_entity(row: dict) -> str:
    """The canonical entity name facts of a generated concept are filed under."""
    ents = json.loads(row.get("entities") or "[]")
    if row["concept_id"].endswith("/overview"):
        return ents[0] if ents else row["title"]
    return row["title"] if row["title"] in ents or not ents else ents[0]

def okf_merge(sources: list[str], target: str, dry_run: bool = False, create: bool = False) -> dict:
    """Folds the facts of one or more concepts/groups/entities into a target concept by renaming their
    entity. Fact IDs, text, categories and timestamps are unchanged; overrides of merged-away entities are dropped."""
    okf_build(embed=False)
    conn = get_db()
    thits, tsugg = okf_resolve(conn, target)
    thits = [h for h in thits if h["origin"] == "facts"]
    if create and len(thits) != 1:
        # Consolidate into a new, better-named entity (e.g. ten "Shopify Video …" variants -> "Shopify Horizon Video").
        trow = {"key": None, "concept_id": f"(new) {slugify(target)}"}
        tentity, tents = target.strip(), set()
    elif len(thits) != 1:
        return {"ok": False, "error": f"Target '{target}' must match exactly one generated concept "
                f"(matched {len(thits)})." + (f" Suggestions: {', '.join(tsugg)}" if tsugg else "")
                + " Use --create to fold into a new entity with that name."}
    else:
        trow = thits[0]
        tentity = concept_entity(trow)
        tents = set(json.loads(trow["entities"] or "[]"))
    move, unresolved = set(), []
    for src in sources:
        hits, _ = okf_resolve(conn, src)
        hits = [h for h in hits if h["origin"] == "facts" and h["key"] != trow["key"]]
        ents = {e for h in hits for e in json.loads(h["entities"] or "[]")}
        if not ents:
            ents = {r[0] for r in conn.execute("SELECT DISTINCT entity FROM facts WHERE entity = ? COLLATE NOCASE", (src,)).fetchall()}
        ents -= tents
        if not ents:
            unresolved.append(src)
        move |= ents
    if not move:
        return {"ok": False, "error": f"Nothing to merge (unresolved: {', '.join(unresolved) or 'none'})."}
    ids = [r[0] for r in conn.execute(
        f"SELECT id FROM facts WHERE entity IN ({','.join('?' * len(move))}) ORDER BY id", sorted(move)).fetchall()]
    plan = {"ok": True, "target": trow["concept_id"], "target_entity": tentity, "entities": sorted(move),
            "fact_ids": ids, "unresolved": unresolved, "dry_run": dry_run}
    if dry_run:
        return plan
    with conn:
        conn.execute(f"UPDATE facts SET entity = ? WHERE entity IN ({','.join('?' * len(move))})", (tentity, *sorted(move)))
        for e in move:
            if slugify(e) != slugify(tentity):
                conn.execute("DELETE FROM okf_meta WHERE slug = ?", (slugify(e),))
    sync_facts_json()
    append_episode(f"**[Merge]** Folded {len(ids)} fact(s) from {', '.join(sorted(move))} into ({tentity}): "
                   + ", ".join(f"#{i}" for i in ids))
    plan["build"] = okf_build(embed=True)
    new_row = conn.execute("SELECT concept_id FROM concepts WHERE origin = 'facts' AND entities LIKE ?",
                           (f'%{json.dumps(tentity)[1:-1]}%',)).fetchone()
    if new_row:
        plan["target"] = new_row[0]
    return plan

def okf_duplicate_candidates(min_ratio: float = 0.8) -> list[dict]:
    """Clusters of generated topic concepts that look like variants of one entity: equal after dropping
    suffix words, near-identical names, or one name extending the other. Project children are excluded
    (they are already grouped under their project)."""
    conn = get_db()
    rows = [dict(r) for r in conn.execute(
        "SELECT key, concept_id, title, entities, fact_ids, status FROM concepts WHERE origin = 'facts' AND concept_id LIKE 'topics/%'").fetchall()]
    parent = {r["key"]: r["key"] for r in rows}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    reasons = {}
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            sa, sb = slugify(a["title"]), slugify(b["title"])
            why = None
            if entity_key(a["title"]) == entity_key(b["title"]):
                why = "same name without suffix words"
            elif difflib.SequenceMatcher(None, sa, sb).ratio() >= max(min_ratio, 0.88):
                why = "near-identical names"
            elif min(sa, sb, key=len).count("-") >= 1 and (sa.startswith(sb + "-") or sb.startswith(sa + "-")):
                why = "one name extends the other"  # shorter name must be 2+ words ('flutter' vs 'flutter-ui-…' is not a dupe)
            if why:
                parent[find(a["key"])] = find(b["key"])
                reasons.setdefault(find(b["key"]), set()).add(why)
    clusters = {}
    for r in rows:
        clusters.setdefault(find(r["key"]), []).append(r)
    out = []
    for root, members in clusters.items():
        if len(members) < 2:
            continue
        members.sort(key=lambda r: (-len(json.loads(r["fact_ids"] or "[]")), len(r["title"])))
        target = members[0]
        out.append({"target": target["concept_id"], "target_title": target["title"],
                    "members": [{"concept_id": m["concept_id"], "title": m["title"], "facts": len(json.loads(m["fact_ids"] or "[]")),
                                 "status": m["status"]} for m in members],
                    "reasons": sorted({w for m in members for w in reasons.get(find(m["key"]), set())}),
                    "command": "brain okf merge " + " ".join(json.dumps(m["concept_id"]) for m in members[1:]) + f" --into {json.dumps(target['concept_id'])}"})
    out = sorted(out, key=lambda c: -len(c["members"]))
    # Sprawl: topic groups made of many one-fact concepts are usually one topic split by naming drift.
    groups = {}
    for r in rows:
        parts = r["concept_id"].split("/")
        if len(parts) == 3:
            groups.setdefault(parts[1], []).append(r)
    for g, members in sorted(groups.items()):
        sizes = sorted(len(json.loads(m["fact_ids"] or "[]")) for m in members)
        if len(members) >= 4 and sizes[len(sizes) // 2] <= 1:
            out.append({"kind": "group", "target": f"topics/{g}", "target_title": f"topics/{g}",
                        "members": [{"concept_id": m["concept_id"], "title": m["title"], "facts": len(json.loads(m["fact_ids"] or "[]")),
                                     "status": m["status"]} for m in members],
                        "reasons": [f"{len(members)} concepts, median {sizes[len(sizes) // 2]} fact each — likely one topic split by naming"],
                        "command": f"brain okf merge topics/{g} --into \"<Consolidated Name>\" --create"})
    return out

def okf_concept_flags(row: dict, now: datetime = None) -> list[str]:
    flags = []
    if row.get("status") and row["status"] != "stable":
        flags.append(row["status"])
    if okf_is_stale(row.get("stale_after"), now):
        flags.append(f"stale since {str(row['stale_after'])[:10]}")
    return flags

def _age_label(iso: str | None) -> str:
    dt = parse_iso(iso)
    if not dt:
        return "age n/a"
    days = int((datetime.now(timezone.utc) - dt).total_seconds() // 86400)
    return "today" if days <= 0 else f"{days}d ago"

def quick_map(query: str = None, target_dir: Path = None, top_n: int = 5, max_items: int = 3, show_all: bool = False) -> dict:
    """RAG + OKF quick map. With a query: ranks concepts by card similarity, concept BM25, and fact/chunk
    evidence from hybrid search, then attaches the best evidence and 1-hop neighbours per concept.
    Without a query: project view when inside a known project, else the bundle overview."""
    build = okf_build(embed=True)
    conn = get_db()
    rows = [dict(r) for r in conn.execute("SELECT * FROM concepts").fetchall()]
    by_key = {r["key"]: r for r in rows}
    now = datetime.now(timezone.utc)
    result = {"query": query, "bundle": str(OKF_DIR), "okf_version": OKF_VERSION, "concept_count": len(rows),
              "concepts": [], "other_facts": [], "documents": []}

    def concept_brief(r):
        return {"key": r["key"], "concept_id": r["concept_id"], "title": r["title"], "type": r["type"],
                "description": r["description"], "trust": r["trust"], "status": r["status"],
                "flags": okf_concept_flags(r, now), "updated": r["generated_at"], "age": _age_label(r["generated_at"]),
                "origin": r["origin"], "file": r["file_path"],
                "fact_count": len(json.loads(r["fact_ids"] or "[]"))}

    if not query:
        target = Path(target_dir or Path.cwd()).resolve()
        proj_row = None
        if not show_all:
            best_len = 0
            for r in rows:
                res = r.get("resource") or ""
                if r["origin"] == "facts" and res.startswith("file://"):
                    ppath = res[len("file://"):]
                    ppath = unquote(ppath)
                    if (str(target) == ppath or str(target).startswith(ppath + "/")) and len(ppath) > best_len:
                        proj_row, best_len = r, len(ppath)
        if proj_row:
            result["mode"] = "project"
            result["project"] = concept_brief(proj_row)
            prefix = proj_row["concept_id"].rsplit("/", 1)[0] + "/"
            kids = [r for r in rows if r["concept_id"].startswith(prefix) and r["key"] != proj_row["key"]]
            result["concepts"] = [concept_brief(r) for r in sorted(kids, key=lambda r: r["generated_at"] or "", reverse=True)]
            links = json.loads(proj_row["links"] or "[]")
            result["related"] = [concept_brief(by_key[k]) for k in links if k in by_key and not by_key[k]["concept_id"].startswith(prefix)][:6]
            fact_ids = json.loads(proj_row["fact_ids"] or "[]")[:max_items + 2]
            result["recent_facts"] = [dict(r) for r in conn.execute(
                f"SELECT id, entity, category, fact, timestamp FROM facts WHERE id IN ({','.join('?' * len(fact_ids))}) ORDER BY timestamp DESC",
                fact_ids).fetchall()] if fact_ids else []
            try:
                txt = Path(proj_row["file_path"]).read_text(encoding="utf-8", errors="ignore")
                sec, _ = extract_markdown_section(txt, "Project Map")
                result["map_outline"] = [l[2:] for l in (sec or "").splitlines() if l.startswith("- ")]
                result["map_file"] = next((s for s in re.findall(r"Source: `([^`]+)`", sec or "")), None)
            except Exception:
                result["map_outline"] = []
            return result
        result["mode"] = "overview"
        groups = {}
        for r in rows:
            parts = r["concept_id"].split("/")
            if r["origin"] == "file":
                g = ("external", Path(r["bundle"]).name)
            elif parts[0] == "projects":
                g = ("projects", parts[1])
            else:
                g = ("topics", parts[1] if len(parts) > 2 else "")
            e = groups.setdefault(g, {"concepts": 0, "facts": 0, "latest": "", "titles": []})
            e["concepts"] += 1
            e["facts"] += len(json.loads(r["fact_ids"] or "[]"))
            e["latest"] = max(e["latest"], r["generated_at"] or "")
            if r["concept_id"].endswith("/overview"):
                e["titles"].insert(0, r["title"])
            else:
                e["titles"].append(r["title"])
        result["groups"] = [{"section": k[0], "group": k[1], **v} for k, v in sorted(groups.items(), key=lambda kv: kv[1]["latest"], reverse=True)]
        result["recent"] = [concept_brief(r) for r in sorted(rows, key=lambda r: r["generated_at"] or "", reverse=True)[:8]]
        result["flagged"] = [concept_brief(r) for r in rows if okf_concept_flags(r, now)]
        # Post-upgrade check: system facts (in live concepts) recorded on an older kernel series / NVIDIA major.
        now_env = current_env()
        result["env"] = now_env
        live_ids = []
        for r in rows:
            if r["origin"] == "facts" and r["status"] != "deprecated":
                live_ids.extend(json.loads(r["fact_ids"] or "[]"))
        drifted = []
        if live_ids:
            frows = [dict(x) for x in conn.execute(
                f"SELECT id, entity, category, fact FROM facts WHERE id IN ({','.join('?' * len(live_ids))})", live_ids).fetchall()]
            drift = fact_drift_map(conn, frows, now_env)
            drifted = [{"id": f["id"], "entity": f["entity"], "category": f["category"], "warning": drift[f["id"]]} for f in frows if f["id"] in drift]
        result["env_drift"] = drifted
        return result

    result["mode"] = "query"
    now_env = current_env()
    result["env"] = now_env
    qvec = get_embedding(query)
    res = search_brain(query, top_k=12, query_vec=qvec)

    card_sim = {}
    if qvec:
        emb_rows = [r for r in rows if r["embedding"]]
        for r, sim in zip(emb_rows, batch_cosine(qvec, [r["embedding"] for r in emb_rows])):
            card_sim[r["key"]] = sim
    fts_score = {}
    fq = build_fts_query(query)
    if fq:
        try:
            for i, r in enumerate(conn.execute("SELECT key FROM concepts_fts WHERE concepts_fts MATCH ? ORDER BY bm25(concepts_fts) LIMIT 30", (fq,)).fetchall()):
                fts_score[r[0]] = 1.0 / (i + 1)
        except sqlite3.Error:
            pass

    entity_key = {}
    for r in rows:
        if r["origin"] == "facts":
            for e in json.loads(r["entities"] or "[]"):
                entity_key[slugify(e)] = r["key"]
    file_key = {r["file_path"]: r["key"] for r in rows if r["origin"] == "file"}
    project_prefixes = []
    for r in rows:
        if r["origin"] == "facts" and (r.get("resource") or "").startswith("file://"):
            project_prefixes.append((unquote(r["resource"][len("file://"):]) + "/", r["key"]))
    project_prefixes.sort(key=lambda x: -len(x[0]))

    def chunk_concept(fp):
        if fp in file_key:
            return file_key[fp]
        for pre, key in project_prefixes:
            if fp.startswith(pre):
                return key
        return None

    fact_ev, chunk_ev, fact_hits, chunk_hits = {}, {}, {}, {}
    for f in res["facts"]:
        key = entity_key.get(slugify(f["entity"] or "General"))
        if key:
            fact_ev[key] = max(fact_ev.get(key, 0.0), f.get("score", 0.0))
            fact_hits.setdefault(key, []).append(f)
    episodes_prefix = str(EPISODES_DIR.resolve()) + "/"
    for c in res["chunks"]:
        if c.get("score", 0.0) < QM_DOC_FLOOR:
            continue
        key = chunk_concept(c["file_path"])
        if key:
            chunk_ev[key] = max(chunk_ev.get(key, 0.0), min(1.0, c.get("score", 0.0)))
            chunk_hits.setdefault(key, []).append(c)

    scored = []
    for key in set(card_sim) | set(fts_score) | set(fact_ev) | set(chunk_ev):
        r = by_key.get(key)
        if not r:
            continue
        evidence = max(fact_ev.get(key, 0.0), 0.8 * chunk_ev.get(key, 0.0))
        score = QM_W_CARD * card_sim.get(key, 0.0) + QM_W_FTS * fts_score.get(key, 0.0) + QM_W_EVIDENCE * evidence
        if not qvec:
            score = 0.6 * fts_score.get(key, 0.0) + 0.4 * evidence
        mult = 1.0
        if r["status"] == "deprecated":
            mult *= 0.6
        elif r["status"] == "draft":
            mult *= 0.9
        if okf_is_stale(r["stale_after"], now):
            mult *= 0.85
        if r["trust"] == "human-reviewed":
            mult *= 1.1
        scored.append((score * mult, key))
    scored.sort(reverse=True)
    if not scored:
        return result
    best = scored[0][0]
    chosen = [k for sc, k in scored if sc >= best * 0.72 and sc >= QM_MIN_SCORE][:top_n]
    chosen_set = set(chosen)

    for key in chosen:
        r = by_key[key]
        item = concept_brief(r)
        item["score"] = round(next(sc for sc, k in scored if k == key), 4)
        evidence = []
        if r["origin"] == "facts":
            ids = json.loads(r["fact_ids"] or "[]")
            if ids:
                frows = conn.execute(
                    f"SELECT f.id, f.entity, f.category, f.fact, f.timestamp, v.embedding FROM facts f "
                    f"LEFT JOIN fact_vectors v ON v.fact_id = f.id WHERE f.id IN ({','.join('?' * len(ids))})", ids).fetchall()
                if qvec:
                    sims = batch_cosine(qvec, [fr["embedding"] or b"" for fr in frows])
                    hit_ids = {f["id"] for f in fact_hits.get(key, [])}
                    ranked = sorted(zip(sims, frows), key=lambda x: (x[1]["id"] in hit_ids, x[0]), reverse=True)
                else:
                    ranked = [(0.0, fr) for fr in sorted(frows, key=lambda fr: fr["timestamp"] or "", reverse=True)]
                drift = fact_drift_map(conn, [{"id": fr["id"], "fact": fr["fact"]} for fr in frows], now_env)
                for sim, fr in ranked[:max_items]:
                    evidence.append({"kind": "fact", "id": fr["id"], "category": fr["category"], "entity": fr["entity"],
                                     "text": fr["fact"], "date": _fact_date(fr["timestamp"]), "env_warning": drift.get(fr["id"])})
                if drift and r["status"] != "deprecated":
                    item["flags"].append(f"{len(drift)} fact(s) from an older kernel/driver")
        for c in chunk_hits.get(key, [])[:max(0, max_items - len(evidence)) or 1]:
            evidence.append({"kind": "chunk", "file": c["file_path"], "header": c.get("header"), "text": c.get("content", "")})
        item["evidence"] = evidence[:max_items + 1]
        item["related"] = [{"title": by_key[k]["title"], "concept_id": by_key[k]["concept_id"]}
                           for k in json.loads(r["links"] or "[]") if k in by_key and k not in chosen_set][:4]
        result["concepts"].append(item)

    for f in res["facts"]:
        if f.get("score", 0.0) < QUICKMAP_OTHER_FACT_FLOOR:
            continue
        if entity_key.get(slugify(f["entity"] or "General")) not in chosen_set:
            result["other_facts"].append({"id": f["id"], "category": f["category"], "entity": f["entity"], "text": f["fact"],
                                          "env_warning": f.get("env_warning")})
    for c in res["chunks"]:
        if c.get("score", 0.0) < QM_DOC_FLOOR:
            continue
        if c["file_path"].startswith(episodes_prefix) or re.search(r"/episodes/\d{4}-\d{2}-\d{2}\.md$", c["file_path"]):
            continue  # episode logs duplicate facts, which are already surfaced with their [#id]
        if chunk_concept(c["file_path"]) in chosen_set:
            continue
        result["documents"].append({"file": c["file_path"], "header": c.get("header"), "text": c.get("content", ""), "score": c.get("score")})
    result["other_facts"] = result["other_facts"][:3]
    result["documents"] = result["documents"][:3]
    return result

def _clip(text: str, n: int) -> str:
    t = " ".join(str(text or "").split())
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + "…"

def render_quick_map(qm: dict, snippet_chars: int = 220) -> str:
    home = str(Path.home())
    short = lambda p: str(p).replace(home, "~", 1) if p else p
    out = []
    mode = qm.get("mode")
    if mode == "query":
        out.append(f"🗺️  QUICK MAP: \"{qm['query']}\"  (OKF v{qm['okf_version']} · {qm['concept_count']} concepts · {short(qm['bundle'])})")
        if not qm["concepts"]:
            raw = " Raw retrieval hits are listed below." if (qm.get("other_facts") or qm.get("documents")) else " Try other keywords or `brain query`."
            out.append("  No concept clears the relevance floor." + raw)
        for i, c in enumerate(qm["concepts"], 1):
            flags = "".join(f" ⚠ {f}" for f in c["flags"])
            out.append(f"\n{i}. {c['title']}  [{c['type']} · {c['concept_id']} · {c['fact_count']} facts · {c['trust']} · {c['age']}]{flags}")
            out.append(f"   {_clip(c['description'], 200)}")
            for e in c.get("evidence", []):
                if e["kind"] == "fact":
                    ent = f" ({e['entity']})" if e["entity"] != c["title"] else ""
                    warn = f" ⚠ {e['env_warning']}" if e.get("env_warning") else ""
                    out.append(f"   • [#{e['id']}] [{e['category']}]{ent} {e['date']}{warn}: {_clip(e['text'], snippet_chars)}")
                else:
                    out.append(f"   • 📄 {Path(e['file']).name} > {e.get('header')}: {_clip(e['text'], snippet_chars - 40)}")
            if c.get("related"):
                out.append("   ↳ related: " + ", ".join(r["title"] for r in c["related"]))
        if qm.get("other_facts"):
            out.append("\n📌 Other matching facts:")
            for f in qm["other_facts"]:
                warn = f" ⚠ {f['env_warning']}" if f.get("env_warning") else ""
                out.append(f"   • [#{f['id']}] [{f['category']}] ({f['entity']}){warn}: {_clip(f['text'], snippet_chars - 40)}")
        if qm.get("documents"):
            out.append("\n📄 Other documents:")
            for d in qm["documents"]:
                out.append(f"   • {short(d['file'])} > {d.get('header')}: {_clip(d['text'], snippet_chars - 60)}")
        out.append("\n💡 Drill down: brain okf show <concept_id>  ·  fix a fact: brain correct --id <N> \"...\"  ·  retire a concept: brain okf set <concept> --status deprecated")
    elif mode == "project":
        p = qm["project"]
        flags = "".join(f" ⚠ {f}" for f in p["flags"])
        out.append(f"🗺️  QUICK MAP · PROJECT {p['title']}  [{p['concept_id']} · {p['fact_count']} facts · {p['age']}]{flags}")
        out.append(f"   {_clip(p['description'], 220)}")
        if qm.get("map_outline"):
            out.append(f"\n🧭 Project map sections ({short(qm.get('map_file'))}):")
            out.append("   " + " · ".join(qm["map_outline"][:24]))
        if qm.get("recent_facts"):
            out.append("\n📌 Recent facts:")
            for f in qm["recent_facts"]:
                out.append(f"   • [#{f['id']}] [{f['category']}] {_fact_date(f['timestamp'])}: {_clip(f['fact'], snippet_chars)}")
        if qm.get("concepts"):
            out.append(f"\n📦 Concepts in this project ({len(qm['concepts'])}):")
            for c in qm["concepts"][:12]:
                flags = "".join(f" ⚠ {f}" for f in c["flags"])
                out.append(f"   • {c['title']} ({c['fact_count']} facts, {c['age']}){flags}: {_clip(c['description'], 120)}")
        if qm.get("related"):
            out.append("\n↳ Related topics: " + ", ".join(r["title"] for r in qm["related"]))
        out.append(f"\n💡 Search inside it: brain quickmap \"<question>\"  ·  full concept: brain okf show {p['concept_id']}")
    else:
        out.append(f"🗺️  QUICK MAP · OVERVIEW  (OKF v{qm['okf_version']} · {qm['concept_count']} concepts · {short(qm['bundle'])}/index.md)")
        for section, label in (("projects", "📁 Projects"), ("topics", "🧩 Topics"), ("external", "📚 External OKF bundles")):
            gs = [g for g in qm.get("groups", []) if g["section"] == section]
            if not gs:
                continue
            out.append(f"\n{label} ({len(gs)}):")
            for g in gs[:14]:
                name = g["titles"][0] if (section == "projects" and g["titles"]) else (g["group"] or "(ungrouped)")
                extra = "" if section == "projects" else f" — {', '.join(g['titles'][:3])}{'…' if len(g['titles']) > 3 else ''}"
                out.append(f"   • {name}: {g['concepts']} concepts, {g['facts']} facts, updated {_age_label(g['latest'])}{extra}")
            if len(gs) > 14:
                out.append(f"   … {len(gs) - 14} more (see {short(qm['bundle'])}/{section}/index.md)")
        if qm.get("recent"):
            out.append("\n🕒 Recently updated:")
            for c in qm["recent"]:
                out.append(f"   • {c['title']} [{c['concept_id']}] {c['age']}: {_clip(c['description'], 110)}")
        if qm.get("flagged"):
            out.append(f"\n⚠ Flagged concepts ({len(qm['flagged'])}): " + ", ".join(f"{c['title']} ({'/'.join(c['flags'])})" for c in qm["flagged"][:10]))
        if qm.get("env_drift"):
            env = qm.get("env") or {}
            by_ent = {}
            for d in qm["env_drift"]:
                by_ent.setdefault(d["entity"], []).append(d["id"])
            out.append(f"\n🔧 Re-verify after upgrade — {len(qm['env_drift'])} system fact(s) were recorded on an older kernel series"
                       f" or NVIDIA driver (running kernel {env.get('kernel')}" + (f", NVIDIA {env.get('nvidia')}" if env.get("nvidia") else "") + "):")
            for ent, ids in sorted(by_ent.items(), key=lambda kv: -len(kv[1]))[:8]:
                out.append(f"   • {ent}: " + ", ".join(f"#{i}" for i in ids[:8]) + ("…" if len(ids) > 8 else ""))
            out.append("   Still valid? `brain reverify <id>...`  ·  Changed? `brain correct --id <N> \"...\"`  ·  Obsolete? `brain okf set <concept> --status deprecated`")
        out.append("\n💡 brain quickmap \"<question>\" for a ranked map · brain okf show <concept_id|group> to open one")
    return "\n".join(out) + "\n"

def render_quick_map_fitted(qm: dict, max_tokens: int = None) -> str:
    """Renders the quick map within a token budget by degrading gracefully (drop other documents, other
    facts, related lists, shorten snippets, fewer evidence items, fewer concepts) before hard truncation."""
    if not max_tokens:
        return render_quick_map(qm)
    budget = max_tokens * 4
    q = json.loads(json.dumps(qm, default=str))
    snippet = 220

    def cap_evidence(n):
        for c in q.get("concepts", []):
            c["evidence"] = (c.get("evidence") or [])[:n]

    steps = [
        lambda: None,
        lambda: q.update(documents=[]),
        lambda: q.update(other_facts=[]),
        lambda: [c.update(related=[]) for c in q.get("concepts", [])],
        lambda: cap_evidence(2),
        lambda: q.update(concepts=q.get("concepts", [])[:4], recent=(q.get("recent") or [])[:4]),
        lambda: cap_evidence(1),
        lambda: q.update(concepts=q.get("concepts", [])[:3], groups=(q.get("groups") or [])[:8], map_outline=(q.get("map_outline") or [])[:10]),
        lambda: q.update(concepts=q.get("concepts", [])[:2], recent=[]),
        lambda: q.update(concepts=q.get("concepts", [])[:1]),
    ]
    for step in steps:
        step()
        for snippet in (220, 160, 110):
            text = render_quick_map(q, snippet_chars=snippet)
            first = step is steps[0] and snippet == 220
            note = "" if first else f"(condensed to fit --max-tokens {max_tokens}; raise it or use --json for everything)\n"
            if len(text) + len(note) <= budget:
                return text + note
    return apply_token_budget(render_quick_map(q, snippet_chars=110), max_tokens)

def okf_validate(bundle_dir: Path) -> dict:
    """OKF v0.2 §11 conformance check. Errors break conformance; warnings are soft guidance."""
    bundle_dir = Path(bundle_dir).resolve()
    errors, warnings = [], []
    concepts = 0
    if not bundle_dir.is_dir():
        return {"bundle": str(bundle_dir), "conformant": False, "errors": [f"Not a directory: {bundle_dir}"], "warnings": [], "concepts": 0}
    ids = {str(p.relative_to(bundle_dir).with_suffix("")) for p in bundle_dir.rglob("*.md")}
    for fp in sorted(bundle_dir.rglob("*.md")):
        rel = str(fp.relative_to(bundle_dir))
        try:
            text = fp.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"{rel}: not valid UTF-8")
            continue
        meta, body = parse_frontmatter(text)
        if fp.name == "index.md":
            if meta and (fp.parent != bundle_dir or set(meta) - {"okf_version"}):
                errors.append(f"{rel}: index.md may only carry frontmatter at bundle root, and only okf_version (§8)")
            continue
        if fp.name == "log.md":
            for h in re.findall(r"^##\s+(.+)$", body, flags=re.M):
                if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", h.strip()):
                    errors.append(f"{rel}: log date heading '{h.strip()}' is not ISO YYYY-MM-DD (§9)")
            continue
        concepts += 1
        if meta is None:
            errors.append(f"{rel}: missing or unparseable YAML frontmatter (§4.1)")
            continue
        if not str(meta.get("type") or "").strip():
            errors.append(f"{rel}: frontmatter has no non-empty 'type' (§4.1)")
        if meta.get("status") and meta["status"] not in ("draft", "stable", "deprecated"):
            warnings.append(f"{rel}: unknown status '{meta['status']}' (§5.4)")
        for key in ("stale_after",):
            if meta.get(key) and not re.search(r"(Z|[+-]\d{2}:?\d{2})$", str(meta[key])):
                warnings.append(f"{rel}: {key} lacks an explicit UTC offset (§5)")
        gen = meta.get("generated")
        if gen is not None and (not isinstance(gen, dict) or not gen.get("by")):
            warnings.append(f"{rel}: generated.by is required within generated (§5.2)")
        for src in meta.get("sources") or []:
            if not isinstance(src, dict) or not src.get("resource"):
                warnings.append(f"{rel}: every sources entry needs a resource (§5.1)")
        for target in MD_LINK_RE.findall(body):
            if re.match(r"^[a-z]+://", target):
                continue
            resolved = (bundle_dir / target.lstrip("/")) if target.startswith("/") else (fp.parent / target)
            if not resolved.exists():
                warnings.append(f"{rel}: broken link {target} (allowed, §6.1)")
    root_idx = bundle_dir / "index.md"
    if not root_idx.exists():
        warnings.append("index.md: no bundle-root index (allowed, consumers synthesize one)")
    return {"bundle": str(bundle_dir), "conformant": not errors, "concepts": concepts,
            "errors": errors, "warnings": warnings[:50], "warning_count": len(warnings)}

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

    notice = f"\n\n[... output truncated: reached limit of --max-tokens {max_tokens} ...]"
    limit = max(0, max_chars - len(notice))  # the notice counts toward the budget
    slice_point = text.rfind("\n", 0, limit)
    if slice_point == -1 or slice_point < limit // 2:
        slice_point = limit

    return text[:slice_point] + notice

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

ROLE_QUERIES = {
    "hardware": "kernel driver firmware udev modprobe bluetooth wifi audio pipewire nvidia suspend acpi usb pcie",
    "backend": "backend api server database sqlite fastapi docker deploy service systemd",
    "frontend": "frontend ui css javascript html react flutter browser dom layout",
    "security": "security permission credential secret auth token policy firewall",
}
ROLE_ALIASES = {"system": "hardware", "kernel": "hardware", "audio": "hardware", "wifi": "hardware", "bluetooth": "hardware",
                "api": "backend", "web": "frontend", "ui": "frontend", "browser": "frontend", "audit": "security"}

def select_role_facts(conn, role_norm: str, st: dict, limit: int = 30) -> list[dict]:
    """Rules/fixes for a subagent: the active project's first, then role-relevant (hybrid-ranked) ones,
    newest first otherwise. Facts in deprecated OKF concepts are skipped."""
    deprecated = set()
    for r in conn.execute("SELECT entities FROM concepts WHERE origin = 'facts' AND status = 'deprecated'").fetchall():
        deprecated |= set(json.loads(r[0] or "[]"))
    picked, seen = [], set()

    def add(rows):
        for f in rows:
            f = dict(f)
            if f["id"] in seen or f["entity"] in deprecated:
                continue
            seen.add(f["id"])
            picked.append(f)

    proj_path = st.get("project_path") if "error" not in st else None
    if proj_path:
        prow = conn.execute("SELECT fact_ids FROM concepts WHERE origin = 'facts' AND resource = ?",
                            (Path(proj_path).resolve().as_uri(),)).fetchone()
        ids = json.loads(prow[0] or "[]") if prow else []
        if ids:
            add(conn.execute(f"SELECT id, entity, category, fact, timestamp FROM facts WHERE id IN ({','.join('?' * len(ids))}) "
                             "AND category IN ('Rule', 'Fix') ORDER BY category = 'Rule' DESC, timestamp DESC LIMIT 8", ids).fetchall())
    role_key = ROLE_ALIASES.get(role_norm, role_norm)
    if role_key in ROLE_QUERIES:
        q = ROLE_QUERIES[role_key]
        add(rank_facts(conn, q, get_embedding(q), ["f.category IN ('Rule', 'Fix')"], [], limit))
    add(conn.execute("SELECT id, entity, category, fact, timestamp FROM facts WHERE category IN ('Rule', 'Fix') "
                     "ORDER BY timestamp DESC, id DESC LIMIT ?", (limit,)).fetchall())
    return picked[:limit]

def generate_role_context(role: str = "general", target_dir: Path = None, max_tokens: int = 800) -> str:
    """Compact, role-tailored system prompt snippet for subagent context injection.
    Budget-aware: the fixed sections are always kept whole and the rules section is filled with one-line
    fact summaries until the token budget is reached (instead of truncating the tail)."""
    role_norm = (role or "general").strip().lower()
    st = get_project_state(target_dir)
    conn = get_db()

    env = current_env()
    home = str(Path.home())
    short = lambda p: str(p).replace(home, "~", 1)
    head = [
        f"# AGENT CONTEXT: {role_norm.upper()}",
        "## 1. Platform",
        "- Arch Linux · ASUS TUF A15 FA506NFR (AMD CPU + NVIDIA GPU) · kernel " + env["kernel"]
        + (f" · NVIDIA {env['nvidia']}" if env.get("nvidia") else ""),
        "- Wi-Fi/BT: MediaTek MT7921 (`14c3:7961`, `mt7921e`) + USB BT (`13d3:3563`, `btusb`/`btmtk`); Realtek RTL8852BE removed.",
        "- ACPI sleep: S0 (s2idle), S4, S5 only (no S3).",
        "## 2. Project",
    ]
    if "error" not in st:
        head.append(f"- Root `{short(st.get('project_path', ''))}` · state `{short(st.get('resolved_file', ''))}` ({st.get('file_type', '')})")
        content = st.get("content", "")
        phase_m = re.search(r"\*\*Active Phase:\*\*\s*([^\n]+)", content)
        status_m = re.search(r"\*\*Status:\*\*\s*([^\n]+)", content)
        if phase_m or status_m:
            head.append(f"- Phase: {phase_m.group(1).strip() if phase_m else '?'} · Status: {status_m.group(1).strip() if status_m else '?'}")
    else:
        head.append("- No .planning/ or project_map.md here.")
    head += ["## 3. Knowledge Map (OKF)"]
    n_concepts = conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]
    head.append(f"- {n_concepts} concepts in `{short(OKF_DIR)}`. Orient: `brain quickmap \"<task>\"` · open: `brain okf show <id>`.")
    proj_path = st.get("project_path") if "error" not in st else None
    if proj_path:
        prow = conn.execute("SELECT concept_id, fact_ids, links FROM concepts WHERE origin = 'facts' AND resource = ?",
                            (Path(proj_path).resolve().as_uri(),)).fetchone()
        if prow:
            rel = [conn.execute("SELECT title FROM concepts WHERE key = ?", (k,)).fetchone() for k in json.loads(prow["links"] or "[]")[:6]]
            rel_titles = ", ".join(r[0] for r in rel if r)
            head.append(f"- Project concept `{prow['concept_id']}` ({len(json.loads(prow['fact_ids'] or '[]'))} facts)" + (f"; related: {rel_titles}" if rel_titles else ""))

    tail = [
        "## 5. Directives",
        "- Loop: DISCUSS → PLAN → EXECUTE → VERIFY → SHIP & REMEMBER. Verify empirically before calling anything done.",
        '- Persist verified findings: `brain remember "<fact>" -e "<Topic>" --fix|--rule`; fix old ones: `brain correct --id <N> "..."`.',
        "- ⚠ on a fact = recorded on another kernel/driver: re-check it before relying on it.",
    ]
    section_title = f"## 4. Rules & Fixes ({role_norm})"
    budget = (max_tokens or 800) * 4
    room = budget - len("\n".join(head + [section_title] + tail)) - 110
    facts = select_role_facts(conn, role_norm, st)
    drift = fact_drift_map(conn, facts, env)
    body, used = [], 0
    for f in facts:
        warn = f" ⚠ {drift[f['id']]}" if f["id"] in drift else ""
        text = re.sub(r"\s#[A-Za-z][\w-]*", "", str(f["fact"]))
        line = f"- [#{f['id']}] {f['category']} · {f['entity']}{warn}: {_clip(text, 160)}"
        if used + len(line) + 1 > room:
            break
        body.append(line)
        used += len(line) + 1
    omitted = len(facts) - len(body)
    if not body:
        body.append("- No specific rules found. Query dynamically via `brain quickmap \"<task>\"`.")
    if omitted > 0:
        body.append(f"- …{omitted} more: `brain quickmap \"<task>\"`.")
    return apply_token_budget("\n".join(head + [section_title] + body + tail), max_tokens)

def clean_orphans(dry_run: bool = False):
    """Finds indexed files that no longer exist on disk and purges their chunks & FTS entries."""
    conn = get_db()
    indexed_files = [r[0] for r in conn.execute("SELECT DISTINCT file_path FROM chunks").fetchall()]
    orphan_files = 0
    orphan_chunks = 0

    if dry_run:
        for fp_str in indexed_files:
            p = Path(fp_str)
            if not p.exists():
                count = conn.execute("SELECT COUNT(*) FROM chunks WHERE file_path = ?", (fp_str,)).fetchone()[0]
                orphan_files += 1
                orphan_chunks += count
        return orphan_files, orphan_chunks

    with conn:
        for fp_str in indexed_files:
            p = Path(fp_str)
            if not p.exists():
                count = conn.execute("SELECT COUNT(*) FROM chunks WHERE file_path = ?", (fp_str,)).fetchone()[0]
                conn.execute("DELETE FROM chunks WHERE file_path = ?", (fp_str,))
                conn.execute("DELETE FROM chunks_fts WHERE file_path = ?", (fp_str,))
                orphan_files += 1
                orphan_chunks += count
        for key, fp_str in conn.execute("SELECT key, file_path FROM concepts WHERE origin = 'file'").fetchall():
            if not fp_str or not Path(fp_str).exists():
                conn.execute("DELETE FROM concepts WHERE key = ?", (key,))
                conn.execute("DELETE FROM concepts_fts WHERE key = ?", (key,))

    return orphan_files, orphan_chunks

def prune_brain(dry_run: bool = False):
    """Cleans orphan files, deduplicates facts, and vacuums the SQLite database."""
    orphans_files, orphan_chunks = clean_orphans(dry_run=dry_run)
    conn = get_db()

    dupe_facts_count = conn.execute("""
        SELECT COUNT(*) FROM facts WHERE id NOT IN (
            SELECT MAX(id) FROM facts GROUP BY entity, category, fact
        )
    """).fetchone()[0]

    if dry_run:
        return orphans_files, orphan_chunks, dupe_facts_count

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

    return orphans_files, orphan_chunks, dupe_facts_count

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

def restore_brain(backup_path: Path | str = "latest", force: bool = False) -> tuple[bool, str]:
    """Restores Central Brain database and vault from a backup archive."""
    ensure_dirs()
    if not backup_path or str(backup_path).lower() in ["latest", "last"]:
        backups = sorted(BACKUP_DIR.glob("*.tar.gz"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not backups:
            return False, f"No backup archives found in {BACKUP_DIR}."
        target_path = backups[0]
    else:
        target_path = Path(backup_path)
        if not target_path.exists() and (BACKUP_DIR / backup_path).exists():
            target_path = BACKUP_DIR / backup_path
        target_path = target_path.resolve()

    if not target_path.exists():
        return False, f"Backup file {target_path} does not exist."

    # Create safety backup of current state
    if not force:
        safety_ok, safety_msg, _ = backup_brain(include_vault=True)
        if not safety_ok:
            return False, f"Could not create pre-restore safety snapshot: {safety_msg}"

    extract_tmp = BACKUP_DIR / f"restore_tmp_{int(time.time())}"
    extract_tmp.mkdir(parents=True, exist_ok=True)

    try:
        with tarfile.open(target_path, "r:gz") as tar:
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

        return True, f"Central Brain restored successfully from {target_path}"
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

def add_source(path: Path | str) -> tuple[bool, str, int]:
    """Registers a new file or directory in sources.json and ingests it into vector DB."""
    ensure_dirs()
    p = Path(path).resolve()
    if not p.exists():
        return False, f"Path does not exist: {p}", 0

    sources = []
    if SOURCES_PATH.exists():
        try:
            with open(SOURCES_PATH, "r", encoding="utf-8") as f:
                sources = json.load(f)
        except Exception:
            sources = []

    p_str = str(p)
    if p_str in sources:
        return True, f"Source already registered: {p_str}", 0

    sources.append(p_str)
    try:
        with open(SOURCES_PATH, "w", encoding="utf-8") as f:
            json.dump(sources, f, indent=2)
    except Exception as e:
        return False, f"Failed to save sources.json: {e}", 0

    if p.is_file():
        chunks = ingest_file(p)
        return True, f"Successfully registered and indexed source ({chunks} chunks): {p_str}", chunks
    chunks, ex_files, ex_chunks = sync_directory_source(p, sources)
    dropped = f"; dropped {ex_chunks} chunks from {ex_files} excluded files" if ex_chunks else ""
    return True, f"Successfully registered and indexed source ({chunks} chunks{dropped}): {p_str}", chunks

def remove_source(path: Path | str, purge_chunks: bool = True) -> tuple[bool, str, int]:
    """Removes a source from sources.json and optionally purges its indexed chunks from DB."""
    ensure_dirs()
    p = Path(path).resolve()
    p_str = str(p)

    sources = []
    if SOURCES_PATH.exists():
        try:
            with open(SOURCES_PATH, "r", encoding="utf-8") as f:
                sources = json.load(f)
        except Exception:
            sources = []

    matched = None
    for s in sources:
        if s == p_str or s == str(path) or (Path(s).exists() and Path(s).resolve() == p):
            matched = s
            break

    if not matched:
        if purge_chunks:
            files, purged = purge_path_chunks(p)
            if purged:
                return True, f"'{p_str}' is not a registered source; purged {purged} ad-hoc chunks from {files} files.", purged
        return False, f"Source not found in registry: {path}", 0

    sources.remove(matched)
    try:
        with open(SOURCES_PATH, "w", encoding="utf-8") as f:
            json.dump(sources, f, indent=2)
    except Exception as e:
        return False, f"Failed to update sources.json: {e}", 0

    purged_count = 0
    if purge_chunks:
        conn = get_db()
        with conn:
            purged_count = conn.execute("DELETE FROM chunks WHERE file_path = ? OR file_path LIKE ?", (matched, f"{matched}/%")).rowcount
            for (ckey,) in conn.execute("SELECT key FROM concepts WHERE origin = 'file' AND (file_path = ? OR file_path LIKE ?)", (matched, f"{matched}/%")).fetchall():
                conn.execute("DELETE FROM concepts WHERE key = ?", (ckey,))
                conn.execute("DELETE FROM concepts_fts WHERE key = ?", (ckey,))
            if purged_count > 0:
                conn.execute("DELETE FROM chunks_fts WHERE file_path = ? OR file_path LIKE ?", (matched, f"{matched}/%"))

    return True, f"Successfully removed source '{matched}' (purged {purged_count} chunks from database).", purged_count

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
    excluded_total = [0, 0]
    for src in sources:
        p = Path(src)
        if p.is_file():
            cnt = ingest_file(p)
            total_chunks += cnt
            synced_paths += 1
        elif p.is_dir():
            # Keeps the index equal to the source's file set (e.g. drops a .venv ingested ad hoc earlier).
            cnt, excluded_files, excluded_chunks = sync_directory_source(p, sources)
            total_chunks += cnt
            synced_paths += 1
            excluded_total[0] += excluded_files
            excluded_total[1] += excluded_chunks

    orphan_files, orphan_chunks = clean_orphans()
    orphan_files += excluded_total[0]
    orphan_chunks += excluded_total[1]
    sync_facts_json()
    backfilled_count = backfill_missing_embeddings()
    backfilled_count += ensure_fact_vectors()
    try:
        backfill_fact_env()
    except Exception as e:
        print(f"[Brain Warning] fact environment backfill failed: {e}", file=sys.stderr)
    try:
        okf_stats = okf_build(embed=True)
    except Exception as e:
        okf_stats = {"error": str(e)}

    return synced_paths, total_chunks, orphan_files, orphan_chunks, backfilled_count, okf_stats

def get_status():
    conn = get_db()
    total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    total_files = conn.execute("SELECT COUNT(DISTINCT file_path) FROM chunks").fetchone()[0]
    total_facts = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    total_concepts = conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]

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
        "okf_concepts": total_concepts,
        "okf_bundle": str(OKF_DIR),
        "ollama_embedding_status": f"Connected ({DEFAULT_EMBED_MODEL} via /api/embed)" if ollama_ok else "Unavailable / Fallback to FTS",
        "database_size_bytes": db_size,
        "database_size_mb": round(db_size / (1024 * 1024), 2)
    }

def run_doctor(fix: bool = False) -> tuple[bool, dict]:
    """Runs a 8-point health check across SQLite, Ollama, vector completeness, registries, and backups.
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
        "backup_freshness": {"passed": False, "detail": ""},
        "okf_bundle": {"passed": False, "detail": ""}
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

    # 8. OKF Knowledge Bundle (current, conformant, concept vectors complete)
    try:
        facts_now = [dict(r) for r in conn.execute(
            "SELECT f.id, f.entity, f.category, f.fact, f.source, f.timestamp, e.kernel, e.nvidia FROM facts f "
            "LEFT JOIN fact_env e ON e.fact_id = f.id ORDER BY f.id").fetchall()]
        current = get_brain_meta(conn, "okf_signature") == okf_signature(facts_now, get_okf_meta(conn), discover_projects())
        val = okf_validate(OKF_DIR) if (OKF_DIR / "index.md").exists() else {"conformant": False, "errors": ["bundle not built"], "concepts": 0}
        missing_vec = conn.execute("SELECT COUNT(*) FROM concepts WHERE embedding IS NULL").fetchone()[0]
        n_concepts = conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]
        problems = []
        if not current:
            problems.append("out of date with facts")
        if not val["conformant"]:
            problems.append(f"{len(val['errors'])} OKF conformance error(s)")
        if missing_vec:
            problems.append(f"{missing_vec} concept(s) missing embeddings")
        if not problems:
            results["okf_bundle"]["passed"] = True
            results["okf_bundle"]["detail"] = f"OKF v{OKF_VERSION} bundle current & conformant ({n_concepts} concepts)"
        else:
            results["okf_bundle"]["detail"] = "; ".join(problems)
            if fix:
                st = okf_build(embed=True, force=True)
                val = okf_validate(OKF_DIR)
                left = conn.execute("SELECT COUNT(*) FROM concepts WHERE embedding IS NULL").fetchone()[0]
                fixes_applied.append(f"Rebuilt OKF bundle ({st.get('concepts')} concepts, {st.get('written')} files written, {st.get('embedded')} embedded).")
                results["okf_bundle"]["passed"] = val["conformant"] and left == 0
    except Exception as e:
        results["okf_bundle"]["detail"] = f"OKF check exception: {e}"

    all_passed = all(v["passed"] for v in results.values())
    results["_meta"] = {
        "all_passed": all_passed,
        "fixes_applied": fixes_applied,
        "timestamp": datetime.now().isoformat()
    }
    return all_passed, results

def list_brain_items(kind: str = "facts", category: str = None, entity: str = None, limit: int = 25, query: str = None) -> tuple[str, list]:
    """Lists facts, sources, discovered projects, or backups with optional query filtering."""
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
        if query:
            wheres.append("(fact LIKE ? OR entity LIKE ?)")
            params.extend([f"%{query}%", f"%{query}%"])
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
        if query:
            items = [i for i in items if query.lower() in i["path"].lower()]
        return "sources", items[:limit]

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
        if query:
            items = [
                i for i in items 
                if query.lower() in i["name"].lower() 
                or query.lower() in i["path"].lower() 
                or (i.get("tech_stack") and any(query.lower() in t.lower() for t in i["tech_stack"]))
            ]
        return "projects", items[:limit]

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
        if query:
            items = [i for i in items if query.lower() in i["filename"].lower()]
        return "backups", items[:limit]

    else:
        return "error", [{"error": f"Unknown list type '{kind}'. Choose from 'facts', 'sources', 'projects', 'backups'."}]

MCP_PROTOCOL_VERSIONS = ["2025-06-18", "2025-03-26", "2024-11-05"]

def run_mcp_server():
    """Runs a standard Model Context Protocol (MCP) JSON-RPC stdio server."""
    sys.stderr.write("Starting Central Brain MCP Server (stdio)...\n")
    sys.stderr.flush()

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            if not line.strip():
                continue
            req = json.loads(line.strip())
            method = req.get("method")
            req_id = req.get("id")
            if "id" not in req:
                continue  # JSON-RPC notification (e.g. notifications/initialized): must not be answered

            if method == "initialize":
                wanted = (req.get("params") or {}).get("protocolVersion")
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": wanted if wanted in MCP_PROTOCOL_VERSIONS else MCP_PROTOCOL_VERSIONS[0],
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "central-brain", "version": BRAIN_VERSION}
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
                                "name": "brain_quickmap",
                                "description": "Quick map (RAG + OKF): rank knowledge concepts (projects/topics) for a question and return the best [#id] facts, document hits, trust/lifecycle flags (deprecated, stale, older-kernel), and related concepts. Omit query for a project/overview map. Start here when orienting on a task.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "query": {"type": "string", "description": "Question or keywords (optional)"},
                                        "top": {"type": "integer", "default": 5},
                                        "path": {"type": "string", "description": "Optional project path for the no-query view"},
                                        "max_tokens": {"type": "integer", "default": 1500, "description": "Output budget; the map condenses gracefully to fit"},
                                        "format": {"type": "string", "enum": ["text", "json"], "default": "text"}
                                    }
                                }
                            },
                            {
                                "name": "brain_reverify",
                                "description": "Mark facts as still valid on the running kernel/NVIDIA driver (after an upgrade flagged them).",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {"ids": {"type": "array", "items": {"type": "integer"}}},
                                    "required": ["ids"]
                                }
                            },
                            {
                                "name": "brain_okf_show",
                                "description": "Open an OKF concept (by id, title, entity) or a group index from the Central Brain knowledge bundle.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "target": {"type": "string", "description": "Concept id/title/entity or group (omit for root index)"}
                                    }
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
                                        "category": {"type": "string", "default": "Knowledge"},
                                        "tags": {"type": "array", "items": {"type": "string"}},
                                        "exact_entity": {"type": "boolean", "default": False, "description": "Skip snapping to an existing entity variant"}
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
                                        "category": {"type": "string", "description": "Optional; by ID the fact keeps its category"}
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
                                "description": "Run 8-point health check across SQLite, Ollama, vector completeness, registries, and backups, with optional auto-repair.",
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
                elif name == "brain_quickmap":
                    qm = quick_map(args.get("query") or None, target_dir=args.get("path"), top_n=int(args.get("top", 5)))
                    text = json.dumps(qm, indent=2, default=str) if args.get("format") == "json" else render_quick_map_fitted(qm, int(args.get("max_tokens", 1500)))
                    resp = {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": text}]}}
                elif name == "brain_okf_show":
                    okf_build(embed=False)
                    tgt = (args.get("target") or "").strip().strip("/")
                    if not tgt:
                        text = (OKF_DIR / "index.md").read_text(encoding="utf-8")
                    elif (OKF_DIR / tgt / "index.md").is_file():
                        text = (OKF_DIR / tgt / "index.md").read_text(encoding="utf-8")
                    else:
                        hits, sugg = okf_resolve(get_db(), tgt)
                        if len(hits) == 1 and Path(hits[0]["file_path"]).exists():
                            text = Path(hits[0]["file_path"]).read_text(encoding="utf-8", errors="ignore")
                        elif hits:
                            text = "Multiple matches: " + ", ".join(h["concept_id"] for h in hits)
                        else:
                            text = f"No concept matches '{tgt}'. Suggestions: {', '.join(sugg) or 'none'}"
                    resp = {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": text}]}}
                elif name == "brain_reverify":
                    done = reverify_facts([int(i) for i in args.get("ids", [])])
                    resp = {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": f"Re-verified on the running kernel/driver: {done or 'none found'}"}]}}
                elif name == "brain_remember":
                    info = remember(args.get("fact"), args.get("entity", "General"), args.get("category", "Knowledge"), source="MCP",
                                    tags=args.get("tags"), exact_entity=bool(args.get("exact_entity")))
                    msg = f"Saved fact #{info['id']} [{info['category']}] under entity '{info['entity']}'."
                    if info["resolution"] in ("normalized", "snapped"):
                        msg += f" ('{info['requested_entity']}' was filed under the existing entity.)"
                    elif info["suggestions"]:
                        msg += f" New entity; similar existing: {', '.join(info['suggestions'])}."
                    resp = {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": msg}]}}
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
                    ok = correct(args.get("entity"), args.get("new_fact"), args.get("old_fact_search"), args.get("category"), source="MCP", fact_id=args.get("id"))
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
            elif method == "ping":
                resp = {"jsonrpc": "2.0", "id": req_id, "result": {}}
            else:
                resp = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method '{method}' not found"}}

            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        except json.JSONDecodeError as e:
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": f"Parse error: {e}"}}) + "\n")
            sys.stdout.flush()
        except Exception as e:
            # Tool failures are reported as an MCP tool result with isError so the client keeps the session.
            rid = req.get("id") if isinstance(locals().get("req"), dict) else None
            if isinstance(locals().get("req"), dict) and req.get("method") == "tools/call":
                err_resp = {"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": f"Error: {e}"}], "isError": True}}
            else:
                err_resp = {"jsonrpc": "2.0", "id": rid, "error": {"code": -32603, "message": str(e)}}
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
    epilog_text = """COMMAND DEFINITIONS & ARGUMENT SPECIFICATIONS:

  1. DYNAMIC MEMORY & SEARCH
     brain query <text> [options] (alias: search)
       Query verified facts, rules, and indexed document chunks.
       Positional:
         <text>                     Search query or keywords (required string)
       Options:
         -k, --top-k INT            Maximum results to return (default: 5)
         -l, --limit INT            Alias for --top-k limit
         -e, --entity NAME          Filter facts by exact/case-insensitive entity name
         -c, --category CAT         Filter facts by category: Fix, Rule, Knowledge, Project
         --rule, --fix, --project, --knowledge
                                    Direct category filter shortcuts
         -s, --source NAME          Filter facts by source origin (e.g. CLI, MCP)
         --since YYYY-MM-DD         Filter facts created on or after date
         --until YYYY-MM-DD         Filter facts created on or before date
         -p, --path PATTERN         Filter indexed document chunks by path substring
         --facts-only               Search and return structured facts only (skips vector/chunks)
         --chunks-only              Search and return document chunks only (skips facts)
         -c, --compact, --terse     Token-efficient single-line output format with deterministic [#id]
         --max-tokens INT           Limit output to approximate token budget
         --json                     Output in structured JSON format

     brain remember <fact> [options]
       Persist a verified discovery, solution, rule, or architectural decision.
       Positional:
         <fact>                     Verified fact, rule, or fix to store (required string)
       Options:
         -e, --entity NAME          Entity/topic (auto-detected from current project if omitted)
         -c, --category CAT         Category: Knowledge (default), Fix, Rule, Project
         --rule, --fix, --project, --knowledge
                                    Direct category shortcuts
         -t, --tags TAGS            Comma-separated tags to append (e.g. 'wifi,driver,kernel')
         -s, --source NAME          Origin identifier (default: 'CLI')
         --exact-entity             Keep the entity exactly as typed (otherwise spelling variants
                                    snap to an existing entity and near matches are suggested)
         --json                     Output in structured JSON format

     brain forget [<target>] [options]
       Purge wrong or outdated memories from Central Brain.
       Positional:
         [target]                   Fact ID (integer) or search phrase to remove (optional)
       Options:
         --id INT                   Deterministic deletion by exact fact ID
         -e, --entity NAME          Filter deletion by entity/topic
         -c, --category CAT         Filter deletion by category (Fix, Rule, Knowledge, Project)
         --json                     Output in structured JSON format

     brain correct [<entity>|<id>] [<new_fact>] [options]
       In-place update or superseding of a memory with verified findings.
       Positional:
         [entity]                   Entity name OR fact ID (if first argument is integer)
         [new_fact]                 New verified replacement text or solution
       Options:
         --id INT                   Deterministic in-place update by exact fact ID
         -o, --old OLD              Old text substring to replace (if not using --id)
         -c, --category CAT         Updated category (default: keep the fact's category)
         --rule, --fix              Direct category shortcuts
         -s, --source NAME          Source identifier (default: 'CLI')
         --json                     Output in structured JSON format

     brain list [kind] [options] (alias: ls)
       Inventory facts, registered sources, discovered projects, or backups.
       Positional:
         [kind]                     Item type: facts (default), sources, projects, backups
       Options:
         -c, --category CAT         Filter facts by category (Fix, Rule, Knowledge, Project)
         -e, --entity NAME          Filter facts by entity/topic
         -q, --query, --search TEXT Search/filter items by keyword across all types
         -l, --limit INT            Maximum number of items to display (default: 25)
         --json                     Output in structured JSON format

  2. PROJECT STATE & MAP
     brain state [path] [options] (alias: plan)
       Inspect or mutate spec-driven project state (.planning/STATE.md or project map).
       Positional:
         [path]                     Target project directory (default: current directory)
       Options:
         -s, --section NAME         Filter output to a specific markdown section header
         -c, --compact, --summary   Token-efficient overview with section list
         --outline                  Display section outline of the document only
         --full                     Display full un-truncated content (disables preview capping)
         --max-tokens INT           Limit output to approximate token budget
         --add TYPE TEXT            Append entry: --add <action|decision|blocker> '<text>'
         --action TEXT              Direct shortcut: add action item to active STATE.md
         --decision TEXT            Direct shortcut: record architectural decision in STATE.md
         --blocker TEXT             Direct shortcut: record blocker in STATE.md
         --phase PHASE              Update active phase in STATE.md
         --status STATUS            Update project status in STATE.md
         --json                     Output in structured JSON format

     brain map [subcommand] [args...]
       Inspect or mutate autonomous project map (.agents/project_map.md).
       Subcommands:
         show [path]                Display active project map (supports -s, -c, --outline, --full)
         list-sections [path]       List all markdown section headers in the active project map
         add <sec> <entry> [p]      Append bullet point or text entry to a section
         set-section <sec> <c> [p]  Replace body content of a specific section in project map
         init [path]                Scaffold a new .agents/project_map.md in project directory
       Options:
         --json                     Output in structured JSON format

     brain init-project <name> [path] [options]
       Scaffold spec-driven .planning/ structure (PROJECT.md, ROADMAP.md, STATE.md).
       Positional:
         <name>                     Project name (required string)
         [path]                     Target directory (default: current directory)
       Options:
         -d, --description DESC     Brief project summary description
         --json                     Output in structured JSON format

     brain quickmap [question...] [options] (aliases: qmap, qm)
       Quick map = RAG + OKF. Ranks OKF concepts (projects/topics) for a question using
       concept-card vectors, concept BM25, and fact/chunk evidence, then shows the best
       [#id] facts, document hits, and 1-hop related concepts per concept.
       With no question: project view inside a known project, else bundle overview.
       Options:
         -n, --top INT              Concepts to show (default: 5)
         -i, --items INT            Evidence items per concept (default: 3)
         -p, --path PATH            Project directory for the no-question view
         -a, --all                  Force the bundle overview (ignore current project)
         --max-tokens INT           Limit output to approximate token budget
         --json                     Output in structured JSON format

     brain okf [subcommand] [args...]
       Open Knowledge Format (OKF v0.2) bundle compiled from facts + project maps at
       ~/.central_brain/okf (index.md, log.md, projects/, topics/). Authored OKF concepts
       (markdown with `type:` frontmatter) in any registered source are indexed too.
       Subcommands:
         build [--force] [--no-embed]  Rebuild the bundle (auto-runs on remember/sync/quickmap)
         show [target]              Print a concept file, or a group/dir index (default: root index)
         validate [dir]             OKF §11 conformance check (default: the generated bundle)
         set <target> [opts]        Concept lifecycle/placement overrides (target: concept id,
                                    title, entity, or group like topics/realtek):
                                      --status draft|stable|deprecated   --stale-after YYYY-MM-DD
                                      --description TEXT  --group NAME  --type TYPE
                                      (pass an empty string to clear a field)
         verify <target> [--by ACTOR]  Record a verification (actor: human:<id>, agent/<ver>,
                                    process:<id>; defaults to human:$USER only on a TTY)
         merge <src>... --into <t>  Fold duplicate concepts/entities/groups into one (fact IDs kept;
                                    --dry-run to preview, --create to consolidate into a new name)
         dupes                      List clusters of concepts that look like variants of one entity
         path                       Print the bundle path

     brain reverify <id>...
       Mark facts as re-verified on the running kernel/NVIDIA driver (text unchanged).
       Facts record the kernel/driver they were written on; quickmap and query flag system
       facts from an older kernel series (major.minor) or NVIDIA major with ⚠.

  3. SOURCES & INDEXING
     brain sources [subcommand] [args...]
       Manage registered Central Brain knowledge sources and directories.
       Subcommands:
         add <path>                 Register a file or directory in sources.json and index immediately
         remove, rm <path>          Unregister a source and purge its indexed chunks from DB
                                    (also purges ad-hoc chunks of an unregistered path)
         list, ls                   List all registered sources with status and chunk counts (default)
       Options:
         --keep-chunks              On remove: do not delete chunks from vector database
         -q, --query TEXT           On list: filter sources by keyword
         --json                     Output in structured JSON format

     brain ingest <path> [options]
       Ingest markdown file or directory into vector index.
       Positional:
         <path>                     File or directory path to ingest (required)
       Options:
         --json                     Output in structured JSON format

     brain sync [options]
       Rescan all registered sources, ingest updated files, and backfill embeddings.
       Directory sources honour .gitignore (own git repos) and skip dependency/build/cache
       folders and files > 512 KB; chunks for files a source no longer covers are dropped.
       Options:
         --json                     Output in structured JSON format

     brain prune [options]
       Clean deleted files, deduplicate facts, and vacuum database.
       Options:
         --dry-run                  Preview deletions without modifying database
         --json                     Output in structured JSON format

  4. SYSTEM, DIAGNOSTICS & BACKUPS
     brain info [path] [options] (alias: paths)
       Introspect Central Brain storage paths, Git status, tech stack, and toolchains.
       Positional:
         [path]                     Target workspace directory (default: current directory)
       Options:
         --paths                    Display Central Brain paths and storage locations
         --json                     Output in structured JSON format

     brain doctor [options] (alias: repair)
       Run 8-point health check and self-heal Central Brain systems.
       Options:
         --fix                      Automatically repair detected issues (backfill vectors,
                                    prune dead sources, rebuild FTS5, fresh backup)
         --json                     Output in structured JSON format

     brain inject [role] [options]
       Generate a role-tailored system prompt context snippet (<800 tokens).
       Positional:
         [role]                     Subagent role: general (default), system, hardware,
                                    frontend, web, backend, api, security, audit
       Options:
         -p, --path PATH            Workspace path to inject project context from
         -t, --tokens INT           Maximum token budget (default: 800)
         --json                     Output in structured JSON format

     brain backup [output] [options]
       Create a transactional snapshot and backup archive of Central Brain.
       Positional:
         [output]                   Destination archive file path (.tar.gz) (optional)
       Options:
         --no-vault                 Backup SQLite DB only, omit markdown vaults
         --json                     Output in structured JSON format

     brain restore [archive] [options]
       Restore Central Brain database and vault from a backup archive.
       Positional:
         [archive]                  Backup archive path or 'latest' (default: latest)
       Options:
         --force                    Skip pre-restore safety snapshot
         --json                     Output in structured JSON format

     brain export [output] [options]
       Export compiled memory digest (MEMORY.md or JSON).
       Positional:
         [output]                   Output destination file path (optional, prints to stdout)
       Options:
         --format FORMAT            Digest format: markdown (default) or json
         -c, --category CAT         Filter facts by category
         -e, --entity NAME          Filter facts by entity
         -d, --days INT             Days of episode history to include (default: 30)
         --json                     Output in structured JSON format

     brain status [options]
       Display system metrics, fact counts, vector chunks, and database size.
       Options:
         --json                     Output in structured JSON format

     brain mcp
       Start Central Brain Model Context Protocol (MCP) JSON-RPC 2.0 stdio server.
"""
    parser = BrainArgumentParser(
        description="Central Brain - Unified Local Agent Memory & State CLI",
        epilog=epilog_text,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--json", action="store_true", help="Output all results in structured JSON format")
    sub = parser.add_subparsers(dest="command")

    # query / search
    q_p = sub.add_parser("query", aliases=["search"], help="Query the central brain for verified facts and document chunks")
    q_p.add_argument("text", type=str, help="Search query string or keywords")
    q_p.add_argument("-k", "--top-k", type=int, default=5, help="Number of results to return (default: 5)")
    q_p.add_argument("-l", "--limit", type=int, default=None, help="Alias for --top-k limit")
    q_p.add_argument("-e", "--entity", type=str, default=None, help="Filter facts by entity name (case-insensitive)")
    q_p.add_argument("-c", "--category", type=str, default=None, help="Filter facts by category: Fix, Rule, Knowledge, Project")
    q_p.add_argument("--rule", action="store_true", help="Shortcut filter for category Rule")
    q_p.add_argument("--fix", action="store_true", help="Shortcut filter for category Fix")
    q_p.add_argument("--project", action="store_true", help="Shortcut filter for category Project")
    q_p.add_argument("--knowledge", action="store_true", help="Shortcut filter for category Knowledge")
    q_p.add_argument("-s", "--source", type=str, default=None, help="Filter facts by source origin (e.g. CLI, MCP)")
    q_p.add_argument("--since", type=str, default=None, help="Filter facts created on or after date (YYYY-MM-DD)")
    q_p.add_argument("--until", type=str, default=None, help="Filter facts created on or before date (YYYY-MM-DD)")
    q_p.add_argument("-p", "--path", type=str, default=None, help="Filter indexed document chunks by file path pattern")
    q_p.add_argument("--facts-only", action="store_true", help="Search and return structured facts only (skips vector/chunks)")
    q_p.add_argument("--chunks-only", action="store_true", help="Search and return document chunks only (skips facts)")
    q_p.add_argument("--compact", "--terse", action="store_true", dest="compact", help="Token-efficient single-line output format with [#id]")
    q_p.add_argument("--max-tokens", type=int, default=None, help="Limit output to approximate token budget")
    q_p.add_argument("--json", action="store_true", help="Output in JSON format")

    # state / plan
    st_cmd = sub.add_parser("state", aliases=["plan"], help="Inspect or mutate spec-driven project state (.planning/STATE.md)")
    st_cmd.add_argument("path", nargs="?", default=None, help="Optional project directory (default: current dir)")
    st_cmd.add_argument("-s", "--section", type=str, default=None, help="Filter output to a specific section name")
    st_cmd.add_argument("-c", "--compact", "--summary", action="store_true", dest="compact", help="Token-efficient summary view with section directory")
    st_cmd.add_argument("--outline", action="store_true", help="Display only the section outline of the document")
    st_cmd.add_argument("--full", action="store_true", help="Display full un-truncated content (disables preview capping)")
    st_cmd.add_argument("--max-tokens", type=int, default=None, help="Limit output to approximate token budget")
    st_cmd.add_argument("--add", nargs=2, metavar=("TYPE", "TEXT"), help="Add an entry: --add <action|decision|blocker> '<text>'")
    st_cmd.add_argument("--action", type=str, default=None, metavar="TEXT", help="Direct shortcut: add action item to active STATE.md")
    st_cmd.add_argument("--decision", type=str, default=None, metavar="TEXT", help="Direct shortcut: record architectural decision in STATE.md")
    st_cmd.add_argument("--blocker", type=str, default=None, metavar="TEXT", help="Direct shortcut: record blocker in STATE.md")
    st_cmd.add_argument("--phase", type=str, default=None, help="Update active phase in STATE.md")
    st_cmd.add_argument("--status", type=str, default=None, help="Update status in STATE.md")
    st_cmd.add_argument("--json", action="store_true", help="Output in JSON format")

    # map
    map_p = sub.add_parser("map", help="Inspect and mutate spec-driven project map (.agents/project_map.md)")
    map_sub = map_p.add_subparsers(dest="map_action")

    map_show = map_sub.add_parser("show", help="Display the active project map")
    map_show.add_argument("path", nargs="?", default=None, help="Optional project directory (default: current dir)")
    map_show.add_argument("-s", "--section", type=str, default=None, help="Filter output to a specific section name")
    map_show.add_argument("-c", "--compact", "--summary", action="store_true", dest="compact", help="Token-efficient summary view with section directory")
    map_show.add_argument("--outline", action="store_true", help="Display only the section outline of the document")
    map_show.add_argument("--full", action="store_true", help="Display full un-truncated content")
    map_show.add_argument("--max-tokens", type=int, default=None, help="Limit output to approximate token budget")
    map_show.add_argument("--json", action="store_true", help="Output in JSON format")

    map_ls = map_sub.add_parser("list-sections", help="List all markdown section headers in the active project map")
    map_ls.add_argument("path", nargs="?", default=None, help="Optional project directory (default: current dir)")
    map_ls.add_argument("--json", action="store_true", help="Output in JSON format")

    map_add = map_sub.add_parser("add", help="Append an entry to a specific section in the project map")
    map_add.add_argument("section", type=str, help="Target section header name (e.g. 'Active System Rules')")
    map_add.add_argument("entry", type=str, help="Bullet point or text entry to append")
    map_add.add_argument("path", nargs="?", default=None, help="Optional project directory (default: current dir)")
    map_add.add_argument("--json", action="store_true", help="Output in JSON format")

    map_set = map_sub.add_parser("set-section", help="Replace the body of a specific section in the project map")
    map_set.add_argument("section", type=str, help="Target section header name")
    map_set.add_argument("content", type=str, help="New body content for the section")
    map_set.add_argument("path", nargs="?", default=None, help="Optional project directory (default: current dir)")
    map_set.add_argument("--json", action="store_true", help="Output in JSON format")

    map_init_cmd = map_sub.add_parser("init", help="Scaffold a new .agents/project_map.md in the project directory")
    map_init_cmd.add_argument("path", nargs="?", default=None, help="Target project directory (default: current dir)")
    map_init_cmd.add_argument("--json", action="store_true", help="Output in JSON format")

    # quickmap
    qm_p = sub.add_parser("quickmap", aliases=["qmap", "qm"], help="Quick map (RAG + OKF): ranked concepts, evidence facts, and related topics")
    qm_p.add_argument("text", nargs="*", help="Question or keywords (omit for project view / overview)")
    qm_p.add_argument("-n", "--top", type=int, default=5, help="Number of concepts to show (default: 5)")
    qm_p.add_argument("-i", "--items", type=int, default=3, help="Evidence items per concept (default: 3)")
    qm_p.add_argument("-p", "--path", type=str, default=None, help="Project directory for the no-question view")
    qm_p.add_argument("-a", "--all", action="store_true", help="Show the bundle overview even inside a project")
    qm_p.add_argument("--max-tokens", type=int, default=None, help="Limit output to approximate token budget")
    qm_p.add_argument("--json", action="store_true", help="Output in JSON format")

    # okf
    okf_p = sub.add_parser("okf", help="Manage the Open Knowledge Format (OKF v0.2) knowledge bundle")
    okf_sub = okf_p.add_subparsers(dest="okf_action")
    okf_b = okf_sub.add_parser("build", help="Rebuild the OKF bundle from facts and project maps")
    okf_b.add_argument("--force", action="store_true", help="Rebuild even if nothing changed")
    okf_b.add_argument("--no-embed", action="store_true", help="Skip concept embeddings")
    okf_b.add_argument("--json", action="store_true", help="Output in JSON format")
    okf_s = okf_sub.add_parser("show", help="Print a concept file or a group index")
    okf_s.add_argument("target", nargs="?", default=None, help="Concept id, title, entity, or group (default: root index)")
    okf_s.add_argument("--max-tokens", type=int, default=None, help="Limit output to approximate token budget")
    okf_s.add_argument("--json", action="store_true", help="Output in JSON format")
    okf_v = okf_sub.add_parser("validate", help="Check OKF v0.2 conformance of a bundle directory")
    okf_v.add_argument("dir", nargs="?", default=None, help="Bundle directory (default: generated bundle)")
    okf_v.add_argument("--json", action="store_true", help="Output in JSON format")
    okf_set = okf_sub.add_parser("set", help="Set lifecycle/placement overrides for a concept or group")
    okf_set.add_argument("target", type=str, help="Concept id, title, entity, or group (e.g. topics/realtek)")
    okf_set.add_argument("--status", choices=["draft", "stable", "deprecated", ""], default=None, help="Lifecycle status")
    okf_set.add_argument("--stale-after", type=str, default=None, help="Date/datetime after which content is stale ('' clears)")
    okf_set.add_argument("--description", type=str, default=None, help="Override the one-line description ('' clears)")
    okf_set.add_argument("--group", type=str, default=None, help="Place under a project slug or topic group ('' clears)")
    okf_set.add_argument("--type", type=str, default=None, help="Override the concept type ('' clears)")
    okf_set.add_argument("--json", action="store_true", help="Output in JSON format")
    okf_ver = okf_sub.add_parser("verify", help="Record a verification event on a concept or group")
    okf_ver.add_argument("target", type=str, help="Concept id, title, entity, or group")
    okf_ver.add_argument("--by", type=str, default=None, help="Actor: human:<id>, <agent>/<version>, or process:<id>")
    okf_ver.add_argument("--json", action="store_true", help="Output in JSON format")
    okf_m = okf_sub.add_parser("merge", help="Fold duplicate concepts/entities into one concept (fact IDs kept)")
    okf_m.add_argument("sources", nargs="+", help="Concept ids, titles, entities, or groups to fold in")
    okf_m.add_argument("--into", required=True, dest="into", help="Target concept (id, title, or entity)")
    okf_m.add_argument("--dry-run", action="store_true", help="Show what would move without changing anything")
    okf_m.add_argument("--create", action="store_true", help="Target is a new entity name (consolidate into a fresh concept)")
    okf_m.add_argument("--json", action="store_true", help="Output in JSON format")
    okf_d = okf_sub.add_parser("dupes", help="List clusters of concepts that look like variants of one entity")
    okf_d.add_argument("--json", action="store_true", help="Output in JSON format")
    okf_path = okf_sub.add_parser("path", help="Print the OKF bundle path")
    okf_path.add_argument("--json", action="store_true", help="Output in JSON format")

    # sources
    src_p = sub.add_parser("sources", help="Manage registered Central Brain knowledge sources and directories")
    src_sub = src_p.add_subparsers(dest="sources_action")

    src_add = src_sub.add_parser("add", help="Register a file or directory in sources.json and index immediately")
    src_add.add_argument("path", type=str, help="File or directory path to register")
    src_add.add_argument("--json", action="store_true", help="Output in JSON format")

    src_rm = src_sub.add_parser("remove", aliases=["rm"], help="Unregister a source and optionally purge its chunks from DB")
    src_rm.add_argument("path", type=str, help="File or directory path to unregister")
    src_rm.add_argument("--keep-chunks", action="store_true", help="Do not delete chunks from vector database")
    src_rm.add_argument("--json", action="store_true", help="Output in JSON format")

    src_ls = src_sub.add_parser("list", aliases=["ls"], help="List all registered sources with existence and chunk counts")
    src_ls.add_argument("-q", "--query", "--search", type=str, default=None, dest="query", help="Filter sources by keyword")
    src_ls.add_argument("--json", action="store_true", help="Output in JSON format")

    # info / paths
    info_p = sub.add_parser("info", aliases=["paths"], help="Display Central Brain paths, configuration introspection, and active workspace map")
    info_p.add_argument("path", nargs="?", default=None, help="Optional workspace directory to inspect")
    info_p.add_argument("--paths", action="store_true", help="Display paths and storage locations")
    info_p.add_argument("--json", action="store_true", help="Output in JSON format")

    # doctor / repair
    doc_p = sub.add_parser("doctor", aliases=["repair"], help="Run 8-point health check and self-heal Central Brain systems")
    doc_p.add_argument("--fix", action="store_true", help="Automatically repair detected issues (backfill embeddings, sync facts, prune dead sources, rebuild FTS5)")
    doc_p.add_argument("--json", action="store_true", help="Output diagnostic results in JSON format")

    # list / ls
    list_p = sub.add_parser("list", aliases=["ls"], help="List facts, registered sources, discovered projects, or backups")
    list_p.add_argument("kind", nargs="?", default="facts", choices=["facts", "sources", "projects", "backups"], help="Type of items to list (default: facts)")
    list_p.add_argument("-c", "--category", type=str, default=None, help="Filter facts by category (Fix, Rule, Knowledge, Project)")
    list_p.add_argument("-e", "--entity", type=str, default=None, help="Filter facts by entity/topic")
    list_p.add_argument("-q", "--query", "--search", type=str, default=None, dest="query", help="Filter items across all types by keyword/text")
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
    r_p = sub.add_parser("remember", help="Save a verified discovery, solution, rule, or decision")
    r_p.add_argument("fact", type=str, help="Verified fact, rule, or fix to persist")
    r_p.add_argument("-e", "--entity", type=str, default="General", help="Entity or topic name (auto-detected if in project directory)")
    r_p.add_argument("-c", "--category", type=str, default="Knowledge", help="Category: Knowledge (default), Fix, Rule, Project")
    r_p.add_argument("--rule", action="store_true", help="Shortcut for --category Rule")
    r_p.add_argument("--fix", action="store_true", help="Shortcut for --category Fix")
    r_p.add_argument("--project", action="store_true", help="Shortcut for --category Project")
    r_p.add_argument("--knowledge", action="store_true", help="Shortcut for --category Knowledge")
    r_p.add_argument("-t", "--tags", type=str, default=None, metavar="TAGS", help="Comma-separated tags to append (e.g. 'wifi,driver,kernel')")
    r_p.add_argument("-s", "--source", type=str, default="CLI", help="Source identifier (default: 'CLI')")
    r_p.add_argument("--exact-entity", action="store_true", help="Store the entity exactly as given (skip snapping to an existing variant)")
    r_p.add_argument("--json", action="store_true", help="Output in JSON format")

    # forget
    f_p = sub.add_parser("forget", help="Remove wrong or outdated memory from the central brain")
    f_p.add_argument("target", type=str, nargs="?", default=None, help="Search term/phrase of the fact to remove OR fact ID")
    f_p.add_argument("--id", type=int, default=None, help="Deterministic deletion by exact fact ID")
    f_p.add_argument("-e", "--entity", type=str, default=None, help="Filter deletion by entity/topic")
    f_p.add_argument("-c", "--category", type=str, default=None, help="Filter deletion by category (Fix, Rule, Knowledge, Project)")
    f_p.add_argument("--json", action="store_true", help="Output in JSON format")

    # correct
    c_p = sub.add_parser("correct", help="Correct/supersede a memory with a new finding")
    c_p.add_argument("entity", type=str, nargs="?", default=None, help="Entity or topic name OR fact ID")
    c_p.add_argument("new_fact", type=str, nargs="?", default=None, help="The new, corrected fact or solution")
    c_p.add_argument("--id", type=int, default=None, help="Deterministic in-place update by exact fact ID")
    c_p.add_argument("-o", "--old", type=str, default=None, help="Old keyword or fact to replace")
    c_p.add_argument("-c", "--category", type=str, default=None, help="Category: Fix, Rule, Knowledge, Project (default: keep the fact's category; Fix for new facts)")
    c_p.add_argument("--rule", action="store_true", help="Shortcut for --category Rule")
    c_p.add_argument("--fix", action="store_true", help="Shortcut for --category Fix")
    c_p.add_argument("-s", "--source", type=str, default="CLI", help="Source identifier (default: 'CLI')")
    c_p.add_argument("--json", action="store_true", help="Output in JSON format")

    # reverify
    rv_p = sub.add_parser("reverify", help="Mark facts as re-verified on the running kernel/NVIDIA driver (text unchanged)")
    rv_p.add_argument("ids", nargs="+", help="Fact IDs (e.g. 42 #43)")
    rv_p.add_argument("--json", action="store_true", help="Output in JSON format")

    # ingest
    i_p = sub.add_parser("ingest", help="Ingest markdown file or directory into vector index")
    i_p.add_argument("path", type=str, help="File or directory path to ingest")
    i_p.add_argument("--json", action="store_true", help="Output in JSON format")

    # sync
    s_p = sub.add_parser("sync", help="Sync all registered knowledge bases & files")
    s_p.add_argument("--json", action="store_true", help="Output in JSON format")

    # prune
    p_p = sub.add_parser("prune", help="Clean deleted files, deduplicate facts, and reclaim disk space")
    p_p.add_argument("--dry-run", action="store_true", help="Preview deletions without modifying database")
    p_p.add_argument("--json", action="store_true", help="Output in JSON format")

    # backup
    bk_p = sub.add_parser("backup", help="Create a transactional snapshot and backup of Central Brain")
    bk_p.add_argument("output", nargs="?", default=None, help="Destination archive path (.tar.gz)")
    bk_p.add_argument("--no-vault", action="store_true", help="Backup SQLite DB only, omit markdown vaults")
    bk_p.add_argument("--json", action="store_true", help="Output in JSON format")

    # restore
    rst_p = sub.add_parser("restore", help="Restore Central Brain from a backup archive")
    rst_p.add_argument("archive", nargs="?", default="latest", help="Backup archive file (.tar.gz) or 'latest' (default: latest)")
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
        category = args.category
        if getattr(args, "rule", False):
            category = "Rule"
        elif getattr(args, "fix", False):
            category = "Fix"
        elif getattr(args, "project", False):
            category = "Project"
        elif getattr(args, "knowledge", False):
            category = "Knowledge"

        top_k = args.limit if getattr(args, "limit", None) else args.top_k
        res = search_brain(
            args.text, top_k=top_k, entity=args.entity, category=category,
            source=args.source, since=args.since, until=args.until, path_filter=args.path,
            facts_only=getattr(args, "facts_only", False),
            chunks_only=getattr(args, "chunks_only", False)
        )
        is_compact = getattr(args, "compact", False)
        max_tokens = getattr(args, "max_tokens", None)

        if is_json:
            out_data = res
            if is_compact:
                out_data = {
                    "facts": [{"id": f["id"], "category": f["category"], "entity": f["entity"], "fact": f["fact"],
                               **({"env_warning": f["env_warning"]} if f.get("env_warning") else {})} for f in res.get("facts", [])],
                    "chunks": [{"file": Path(c["file_path"]).name, "header": c.get("header"), "snippet": c.get("content", "")[:120].strip()} for c in res.get("chunks", [])]
                }
            json_str = json.dumps({"status": "success", "command": "query", "data": out_data, "timestamp": datetime.now().isoformat()}, indent=2)
            print(apply_token_budget(json_str, max_tokens))
        else:
            if is_compact:
                lines = []
                if res["facts"]:
                    for f in res["facts"]:
                        warn = f" ⚠ {f['env_warning']}" if f.get("env_warning") else ""
                        lines.append(f"[#{f['id']}] [{f['category']}] ({f['entity']}){warn}: {f['fact']}")
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
                        warn = f" ⚠ {f['env_warning']}" if f.get("env_warning") else ""
                        out_lines.append(f"  • [#{f['id']}] [{f['category']}] ({f['entity']}){warn}: {f['fact']} ({f['timestamp']})")
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
        # Handle direct action / decision / blocker shortcuts if passed
        action_text = getattr(args, "action", None)
        decision_text = getattr(args, "decision", None)
        blocker_text = getattr(args, "blocker", None)
        if action_text:
            ok, msg = state_add_entry(args.path, "action", action_text)
            if is_json:
                print(json.dumps({"status": "success" if ok else "error", "command": "state", "action": "add", "data": {"type": "action", "message": msg}}, indent=2))
            else:
                print(f"{'✅' if ok else '❌'} {msg}")
            return
        elif decision_text:
            ok, msg = state_add_entry(args.path, "decision", decision_text)
            if is_json:
                print(json.dumps({"status": "success" if ok else "error", "command": "state", "action": "add", "data": {"type": "decision", "message": msg}}, indent=2))
            else:
                print(f"{'✅' if ok else '❌'} {msg}")
            return
        elif blocker_text:
            ok, msg = state_add_entry(args.path, "blocker", blocker_text)
            if is_json:
                print(json.dumps({"status": "success" if ok else "error", "command": "state", "action": "add", "data": {"type": "blocker", "message": msg}}, indent=2))
            else:
                print(f"{'✅' if ok else '❌'} {msg}")
            return

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

    elif args.command in ["quickmap", "qmap", "qm"]:
        question = " ".join(args.text).strip() or None
        qm = quick_map(question, target_dir=args.path, top_n=max(1, args.top), max_items=max(1, args.items), show_all=args.all)
        if is_json:
            print(apply_token_budget(json.dumps({"status": "success", "command": "quickmap", "data": qm}, indent=2, default=str), args.max_tokens))
        else:
            print(render_quick_map_fitted(qm, args.max_tokens))

    elif args.command == "okf":
        action = getattr(args, "okf_action", None) or "show"
        if action == "build":
            st = okf_build(embed=not args.no_embed, force=args.force)
            if is_json:
                print(json.dumps({"status": "success", "command": "okf", "action": "build", "data": st}, indent=2))
            else:
                print(f"🗺️  OKF bundle {'unchanged' if st['skipped'] else 'built'}: {st['concepts']} concepts "
                      f"({st['generated_concepts']} generated, {st['file_concepts']} from sources) at {st['path']} — "
                      f"{st['written']} files written, {st['removed']} removed, {st['embedded']} embedded.")
        elif action == "path":
            print(json.dumps({"status": "success", "command": "okf", "data": {"path": str(OKF_DIR)}}) if is_json else str(OKF_DIR))
        elif action == "merge":
            res = okf_merge(args.sources, args.into, dry_run=args.dry_run, create=args.create)
            if is_json:
                print(json.dumps({"status": "success" if res.get("ok") else "error", "command": "okf", "action": "merge", "data": res}, indent=2, default=str))
            elif not res.get("ok"):
                print(f"❌ {res['error']}")
            else:
                verb = "Would fold" if args.dry_run else "Folded"
                print(f"{'🔍' if args.dry_run else '✅'} {verb} {len(res['fact_ids'])} fact(s) from {', '.join(repr(e) for e in res['entities'])} "
                      f"into {res['target']} (entity '{res['target_entity']}'). Fact IDs unchanged: {', '.join('#' + str(i) for i in res['fact_ids'][:20])}"
                      + ("…" if len(res['fact_ids']) > 20 else ""))
                if res.get("unresolved"):
                    print(f"   ⚠ Not found: {', '.join(res['unresolved'])}")
        elif action == "dupes":
            okf_build(embed=False)
            clusters = okf_duplicate_candidates()
            if is_json:
                print(json.dumps({"status": "success", "command": "okf", "action": "dupes", "data": clusters}, indent=2))
            elif not clusters:
                print("✅ No duplicate-looking concepts found.")
            else:
                print(f"🔎 {len(clusters)} cluster(s) of concepts that look like variants of one entity (review before merging):")
                for c in clusters:
                    label = "Group" if c.get("kind") == "group" else "Duplicate"
                    print(f"\n• {label}: {c['target_title']}  [{'; '.join(c['reasons'])}]")
                    for m in c["members"]:
                        flag = f" [{m['status']}]" if m["status"] != "stable" else ""
                        print(f"     - {m['concept_id']} ({m['facts']} facts){flag}")
                    print(f"   {c['command']}")
        elif action == "validate":
            if not args.dir:
                okf_build(embed=False)
            val = okf_validate(Path(args.dir) if args.dir else OKF_DIR)
            if is_json:
                print(json.dumps({"status": "success" if val["conformant"] else "error", "command": "okf", "action": "validate", "data": val}, indent=2))
            else:
                print(f"{'✅' if val['conformant'] else '❌'} {val['bundle']}: {'conformant with' if val['conformant'] else 'NOT conformant with'} OKF v{OKF_VERSION} "
                      f"({val['concepts']} concepts, {len(val['errors'])} errors, {val.get('warning_count', 0)} warnings)")
                for e in val["errors"][:30]:
                    print(f"  ✗ {e}")
                for w in val["warnings"][:15]:
                    print(f"  • {w}")
        elif action == "show":
            okf_build(embed=False)
            conn = get_db()
            text, info = None, {}
            if not args.target:
                fp = OKF_DIR / "index.md"
                text, info = fp.read_text(encoding="utf-8"), {"file": str(fp)}
            else:
                t = args.target.strip().strip("/")
                dir_idx = OKF_DIR / t / "index.md"
                if dir_idx.is_file():
                    text, info = dir_idx.read_text(encoding="utf-8"), {"file": str(dir_idx)}
                else:
                    hits, sugg = okf_resolve(conn, t)
                    if len(hits) == 1:
                        fp = Path(hits[0]["file_path"])
                        text = fp.read_text(encoding="utf-8", errors="ignore") if fp.exists() else None
                        info = {"file": str(fp), "concept_id": hits[0]["concept_id"], "trust": hits[0]["trust"], "status": hits[0]["status"]}
                    elif hits:
                        text = "Multiple concepts match; pick one:\n" + "\n".join(f"  • {h['concept_id']} — {h['title']}" for h in hits)
                        info = {"matches": [h["concept_id"] for h in hits]}
                    else:
                        info = {"error": f"No concept matches '{args.target}'.", "suggestions": sugg}
            if is_json:
                print(apply_token_budget(json.dumps({"status": "success" if text else "error", "command": "okf", "action": "show", "data": {**info, "content": text}}, indent=2), args.max_tokens))
            elif text:
                print(apply_token_budget(text, args.max_tokens))
            else:
                print(f"❌ {info['error']}" + (f"\n💡 Did you mean: {', '.join(info['suggestions'])}" if info.get("suggestions") else ""))
        elif action in ("set", "verify"):
            conn = get_db()
            okf_build(embed=False)
            hits, sugg = okf_resolve(conn, args.target)
            hits = [h for h in hits if h["origin"] == "facts"]
            if not hits:
                msg = f"No generated concept matches '{args.target}'." + (f" Did you mean: {', '.join(sugg)}?" if sugg else "") + \
                      " (Authored OKF files are edited directly in their frontmatter.)"
                print(json.dumps({"status": "error", "command": "okf", "action": action, "error": msg}) if is_json else f"❌ {msg}")
                return
            if action == "set":
                updates = {"status": args.status, "description": args.description, "group": args.group, "type": args.type}
                if args.stale_after is not None:
                    if args.stale_after == "":
                        updates["stale_after"] = ""
                    else:
                        iso = to_iso_utc(args.stale_after if "T" in args.stale_after or " " in args.stale_after else args.stale_after + "T00:00:00+00:00")
                        if not iso:
                            print(f"❌ Invalid --stale-after value: {args.stale_after}")
                            return
                        updates["stale_after"] = iso
                if all(v is None for v in updates.values()):
                    print("❌ Nothing to set. Use --status, --stale-after, --description, --group, or --type.")
                    return
            else:
                actor = args.by
                if not actor:
                    if sys.stdin.isatty():
                        actor = f"human:{os.getenv('USER') or 'user'}"
                    else:
                        print("❌ Non-interactive verify needs --by <actor> (human:<id> only for a real human review).")
                        return
            changed = []
            for h in hits:
                # Meta is keyed by entity slug (stable across regrouping); a project overview keys by project slug.
                if h["concept_id"].endswith("/overview"):
                    slugs = {h["concept_id"].split("/")[-2]}
                else:
                    slugs = {slugify(e) for e in json.loads(h["entities"] or "[]")} or {h["concept_id"].split("/")[-1]}
                for sl in slugs:
                    if action == "set":
                        set_okf_meta(conn, sl, updates)
                    else:
                        cur = get_okf_meta(conn).get(sl, {})
                        ver = [v for v in cur.get("verified", []) if v.get("by") != actor]
                        ver.append({"by": actor, "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
                        set_okf_meta(conn, sl, {"verified": ver})
                changed.append(h["concept_id"])
            st = okf_build(embed=True)
            if is_json:
                print(json.dumps({"status": "success", "command": "okf", "action": action, "data": {"concepts": changed, "build": st}}, indent=2))
            else:
                print(f"✅ Updated {len(changed)} concept(s): {', '.join(changed[:8])}{'…' if len(changed) > 8 else ''} (bundle rebuilt, {st['written']} files written)")

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
            print("\n🩺 CENTRAL BRAIN HEALTH DIAGNOSTICS (8 Quality Gates)")
            print("="*68)
            gate_names = {
                "sqlite_integrity": "SQLite DB Integrity",
                "ollama_embed": "Ollama Embedding Engine",
                "facts_sync": "Facts Registry Sync",
                "vector_completeness": "Vector Embeddings",
                "sources_health": "Sources Registry Health",
                "fts5_index": "Full-Text Search (FTS5)",
                "backup_freshness": "Backup Freshness",
                "okf_bundle": "OKF Knowledge Bundle"
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

    elif args.command == "sources":
        action = getattr(args, "sources_action", None) or "list"
        if action == "add":
            ok, msg, chunks = add_source(args.path)
            if is_json:
                print(json.dumps({"status": "success" if ok else "error", "command": "sources", "action": "add", "data": {"message": msg, "path": args.path, "chunks": chunks}}, indent=2))
            else:
                print(f"{'✅' if ok else '❌'} {msg}")
        elif action in ["remove", "rm"]:
            purge = not getattr(args, "keep_chunks", False)
            ok, msg, purged = remove_source(args.path, purge_chunks=purge)
            if is_json:
                print(json.dumps({"status": "success" if ok else "error", "command": "sources", "action": "remove", "data": {"message": msg, "path": args.path, "purged_chunks": purged}}, indent=2))
            else:
                print(f"{'✅' if ok else '❌'} {msg}")
        else:
            # list
            query = getattr(args, "query", None)
            kind, items = list_brain_items("sources", query=query)
            if is_json:
                print(json.dumps({"status": "success", "command": "sources", "action": "list", "data": items, "count": len(items)}, indent=2))
            else:
                header_suffix = f" (Filter: '{query}')" if query else ""
                print(f"\n📁 REGISTERED CENTRAL BRAIN SOURCES ({len(items)} entries{header_suffix})")
                print("="*68)
                if not items:
                    print("  No registered sources found.")
                for item in items:
                    stat = "EXISTS" if item["exists"] else "MISSING"
                    print(f"  • {item['path']:<45} [{stat}] ({item['type']}, {item['indexed_chunks']} chunks)")
                print("="*68 + "\n")

    elif args.command in ["list", "ls"]:
        query = getattr(args, "query", None)
        kind, items = list_brain_items(args.kind, args.category, args.entity, args.limit, query=query)
        if is_json:
            print(json.dumps({"status": "success", "command": "list", "kind": kind, "data": items, "count": len(items)}, indent=2))
        else:
            header_suffix = f" (Filter: '{query}')" if query else ""
            print(f"\n📋 CENTRAL BRAIN LIST: {kind.upper()} ({len(items)} items{header_suffix})")
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
        category = args.category
        if getattr(args, "rule", False):
            category = "Rule"
        elif getattr(args, "fix", False):
            category = "Fix"
        elif getattr(args, "project", False):
            category = "Project"
        elif getattr(args, "knowledge", False):
            category = "Knowledge"

        tags_str = getattr(args, "tags", None)
        tags_list = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else None

        info = remember(args.fact, args.entity, category, args.source, tags=tags_list, exact_entity=args.exact_entity)
        if is_json:
            print(json.dumps({"status": "success", "command": "remember", "data": {**info, "source": args.source, "tags": tags_list}}, indent=2))
        else:
            tag_msg = f" (tags: {', '.join(tags_list)})" if tags_list else ""
            print(f"✅ Saved memory to Central Brain: [#{info['id']}] [{info['category']}] ({info['entity']}): {args.fact}{tag_msg}")
            if info["resolution"] in ("normalized", "snapped"):
                print(f"   ↪ Entity '{info['requested_entity']}' filed under existing '{info['entity']}' (use --exact-entity to keep it separate).")
            elif info["resolution"] == "new" and info["suggestions"]:
                print(f"   💡 New entity '{info['entity']}'. Similar existing: {', '.join(repr(x) for x in info['suggestions'])}. "
                      f"If it is the same topic, fix with: brain okf merge \"{info['entity']}\" --into \"{info['suggestions'][0]}\"")
            if info.get("deprecated_concept"):
                print(f"   ⚠ Concept {info['deprecated_concept']} is deprecated. If this is current again: brain okf set {info['deprecated_concept']} --status stable")

    elif args.command == "forget":
        fact_id = getattr(args, "id", None)
        target = args.target
        if fact_id is None and target and re.match(r"^#?(\d+)$", target.strip()):
            fact_id = int(re.match(r"^#?(\d+)$", target.strip()).group(1))
            target = None

        category = getattr(args, "category", None)
        cnt = forget(target, args.entity, fact_id=fact_id, category=category)
        if is_json:
            print(json.dumps({"status": "success", "command": "forget", "data": {"purged_count": cnt, "target": target, "id": fact_id, "entity": args.entity, "category": category}}, indent=2))
        else:
            if fact_id is not None:
                if cnt > 0:
                    print(f"🗑️ Central Brain: Purged fact #{fact_id}.")
                else:
                    print(f"❌ Central Brain: Fact #{fact_id} not found.")
            else:
                cat_info = f", Category: {category}" if category else ""
                print(f"🗑️ Central Brain: Purged {cnt} matching fact(s) matching '{target}' (Entity: {args.entity or 'Any'}{cat_info}).")

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

        category = args.category
        if getattr(args, "rule", False):
            category = "Rule"
        elif getattr(args, "fix", False):
            category = "Fix"

        ok = correct(entity, new_fact, args.old, category, args.source, fact_id=fact_id)
        if is_json:
            print(json.dumps({"status": "success" if ok else "error", "command": "correct", "data": {"entity": entity, "category": category, "new_fact": new_fact, "id": fact_id, "source": args.source}}, indent=2))
        else:
            if ok:
                if fact_id is not None:
                    print(f"✨ Central Brain: Successfully updated fact #{fact_id} -> {new_fact}")
                else:
                    print(f"✨ Central Brain: Successfully corrected memory for [{category}] ({entity}) -> {new_fact}")
            else:
                print(f"❌ Central Brain: Could not correct memory (fact not found).")

    elif args.command == "reverify":
        ids = [int(m.group(1)) for x in args.ids for m in [re.match(r"^#?(\d+)$", x.strip())] if m]
        done = reverify_facts(ids)
        env = current_env()
        missing = sorted(set(ids) - set(done))
        if is_json:
            print(json.dumps({"status": "success" if done else "error", "command": "reverify", "data": {"reverified": done, "not_found": missing, "env": env}}, indent=2))
        else:
            if done:
                print(f"✅ Re-verified {', '.join('#' + str(i) for i in done)} on kernel {env['kernel']}" + (f", NVIDIA {env['nvidia']}" if env.get("nvidia") else ""))
            if missing:
                print(f"❌ Not found: {', '.join('#' + str(i) for i in missing)}")

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
        paths, chunks, del_files, del_chunks, backfilled, okf_stats = sync_brain()
        if is_json:
            print(json.dumps({"status": "success", "command": "sync", "data": {"synced_paths": paths, "total_chunks": chunks, "purged_files": del_files, "purged_chunks": del_chunks, "backfilled_embeddings": backfilled, "okf": okf_stats}}, indent=2))
        else:
            msg = f"🔄 Central Brain Sync Complete: Processed {paths} sources ({chunks} total chunks active)."
            if del_files > 0:
                msg += f" Purged {del_files} deleted files ({del_chunks} orphan chunks removed)."
            if backfilled > 0:
                msg += f" Backfilled {backfilled} missing vector embeddings."
            if okf_stats.get("error"):
                msg += f"\n⚠️  OKF bundle build failed: {okf_stats['error']}"
            else:
                msg += (f"\n🗺️  OKF bundle: {okf_stats.get('concepts', 0)} concepts ({okf_stats.get('file_concepts', 0)} from sources)"
                        f" at {okf_stats.get('path')} — {okf_stats.get('written', 0)} files written"
                        + (" (unchanged)" if okf_stats.get("skipped") else "") + ".")
            print(msg)

    elif args.command == "prune":
        is_dry = getattr(args, "dry_run", False)
        del_files, del_chunks, dupe_facts = prune_brain(dry_run=is_dry)
        if is_json:
            print(json.dumps({"status": "success", "command": "prune", "data": {"dry_run": is_dry, "purged_files": del_files, "purged_chunks": del_chunks, "duplicate_facts": dupe_facts}}, indent=2))
        else:
            if is_dry:
                print(f"🔍 Prune Preview (Dry Run): Found {del_files} deleted files ({del_chunks} orphan chunks) and {dupe_facts} duplicate facts eligible for removal.")
            else:
                print(f"🧹 Central Brain Prune Complete: Cleaned {del_files} deleted files ({del_chunks} chunks removed), deduplicated {dupe_facts} facts, and vacuumed database.")

    elif args.command == "backup":
        ok, msg, manifest = backup_brain(args.output, include_vault=not args.no_vault)
        if is_json:
            print(json.dumps({"status": "success" if ok else "error", "command": "backup", "data": {"message": msg, "manifest": manifest}}, indent=2))
        else:
            print(f"{'📦' if ok else '❌'} {msg}")

    elif args.command == "restore":
        archive_target = getattr(args, "archive", "latest") or "latest"
        ok, msg = restore_brain(archive_target, force=args.force)
        if is_json:
            print(json.dumps({"status": "success" if ok else "error", "command": "restore", "data": {"message": msg, "target": str(archive_target)}}, indent=2))
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
