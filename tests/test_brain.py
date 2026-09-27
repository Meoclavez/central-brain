"""Hermetic tests for Central Brain.

Runs against a throwaway HOME and CENTRAL_BRAIN_DIR with deterministic fake embeddings, so it never
touches the live brain (~/.central_brain) and does not need Ollama:

    python3 -m unittest discover -s tests -v
"""
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="brain-test-"))
HOME = TMP / "home"
HOME.mkdir()
os.environ["HOME"] = str(HOME)
os.environ["CENTRAL_BRAIN_DIR"] = str(TMP / "brain")

_spec = importlib.util.spec_from_file_location("brain_under_test", ROOT / "brain.py")
b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b)

DIM = 256


def fake_embed(text):
    """Bag-of-hashed-tokens vector: texts sharing words are similar, unrelated texts are ~orthogonal."""
    v = [0.0] * DIM
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        if len(tok) < 2 or tok in b.FTS_STOPWORDS:
            continue
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        v[h % DIM] += 1.0
        v[(h // DIM) % DIM] += 0.5
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


b.get_embeddings_batch = lambda texts, model=None, max_retries=3: [fake_embed(t) if t and t.strip() else None for t in texts]
ENV = {"kernel": "7.2.6-arch2-1", "nvidia": "615.71.09"}
b.current_env = lambda: dict(ENV)

F = {}  # name -> fact id
PROJECT = HOME / "Projects" / "demo-app"


def facts_of(entity):
    return [r[0] for r in b.get_db().execute("SELECT id FROM facts WHERE entity = ? ORDER BY id", (entity,)).fetchall()]


def setUpModule():
    # brain.py opens a short-lived SQLite connection per call (closed on GC); not a leak, just noise here.
    warnings.filterwarnings("ignore", category=ResourceWarning)
    add = lambda key, fact, entity, cat: F.__setitem__(key, b.remember(fact, entity, cat, exact_entity=True)["id"])
    add("wifi", "MediaTek MT7921 uses the mt7921e kernel driver; never blacklist the btmtk module in /etc/modprobe.d", "Wi-Fi Card", "Rule")
    add("ollama", "Ollama GPU layers: set PARAMETER num_gpu 10 in the Modelfile to fit the 4GB VRAM budget", "Ollama", "Fix")
    add("starship", "Starship git timeout warning in fish shell fixed by command_timeout = 5000 in starship.toml", "Starship", "Fix")
    add("shop_sound", "Shopify video sound button must call stopPropagation so taps do not open the zoom modal", "Shopify Video Sound", "Fix")
    add("shop_zoom", "Shopify fullscreen zoom dialog videos need object-fit contain to keep frame alignment", "Shopify Video Zoom", "Fix")
    add("demo", "demo-app deploys with docker compose up -d on the VPS; never rsync the build directory", "demo-app", "Rule")

    (PROJECT / ".agents").mkdir(parents=True)
    (PROJECT / "README.md").write_text("# demo-app\n\nA small demo web application used by the test suite for deployment docs.\n")
    (PROJECT / ".agents" / "project_map.md").write_text("# Project Map - demo-app\n\n## Deploy\n- docker compose up -d\n")
    (PROJECT / ".venv" / "lib").mkdir(parents=True)
    (PROJECT / ".venv" / "lib" / "junk.md").write_text("# venv junk\n\nshould never be indexed at all\n")
    (PROJECT / "node_modules" / "pkg").mkdir(parents=True)
    (PROJECT / "node_modules" / "pkg" / "README.md").write_text("# dependency readme\n\nshould never be indexed at all\n")
    (PROJECT / "secret-notes").mkdir()
    (PROJECT / "secret-notes" / "notes.md").write_text("# gitignored\n\nshould never be indexed at all\n")
    (PROJECT / "big.md").write_text("# big\n\n" + "x" * (600 * 1024))
    (PROJECT / ".gitignore").write_text("secret-notes/\n.venv/\nnode_modules/\n")
    subprocess.run(["git", "init", "-q", str(PROJECT)], check=True)
    b.okf_build(embed=True, force=True)


def tearDownModule():
    shutil.rmtree(TMP, ignore_errors=True)


class FrontmatterAndQueryTests(unittest.TestCase):
    def test_frontmatter_roundtrip_matches_fallback_parser(self):
        meta = {"type": "Topic", "title": "Wi-Fi: MT7921 \"combo\"", "tags": ["wifi", "0.2"],
                "generated": {"by": "central-brain/2.5.0", "at": "2026-09-27T10:00:00Z"},
                "sources": [{"id": "facts", "resource": "central-brain facts where entity in [\"X\"]", "title": "t"}]}
        text = b.dump_frontmatter(meta) + "\n# Body\n"
        with_yaml, _ = b.parse_frontmatter(text)
        saved = b.yaml
        try:
            b.yaml = None
            fallback, body = b.parse_frontmatter(text)
        finally:
            b.yaml = saved
        self.assertEqual(with_yaml, fallback)
        self.assertEqual(fallback["title"], meta["title"])
        self.assertEqual(fallback["tags"], ["wifi", "0.2"])
        self.assertIn("# Body", body)

    def test_fts_query_is_or_of_prefix_terms_without_stopwords(self):
        q = b.build_fts_query("How do I fix the starship timeout?")
        self.assertIn('"fix"*', q)
        self.assertIn(" OR ", q)
        self.assertNotIn('"how"', q)

    def test_natural_language_fact_query(self):
        res = b.search_brain("how do I fix the starship timeout in fish", top_k=3, facts_only=True)
        self.assertEqual(res["facts"][0]["id"], F["starship"])

    def test_old_style_insert_is_searchable_via_triggers(self):
        conn = b.get_db()
        with conn:  # what v2.3 does: plain INSERT, no knowledge of facts_fts
            fid = conn.execute("INSERT INTO facts (entity, category, fact, source) VALUES ('Legacy', 'Fix', 'zebra quokka legacy insert', 'CLI')").lastrowid
        try:
            res = b.search_brain("zebra quokka", top_k=3, facts_only=True)
            self.assertIn(fid, [f["id"] for f in res["facts"]])
        finally:
            b.forget(fact_id=fid)


class EntityTests(unittest.TestCase):
    def tearDown(self):
        for fid in getattr(self, "created", []):
            b.forget(fact_id=fid)

    def test_resolution_snaps_variants_and_suggests_near_matches(self):
        conn = b.get_db()
        self.assertEqual(b.resolve_entity(conn, "ollama")[:2], ("Ollama", "normalized"))
        self.assertEqual(b.resolve_entity(conn, "Ollama Fix")[:2], ("Ollama", "snapped"))
        self.assertEqual(b.resolve_entity(conn, "Ollamma")[:2], ("Ollama", "snapped"))
        entity, how, sugg = b.resolve_entity(conn, "Shopify Video Glitches")
        self.assertEqual(how, "new")
        self.assertTrue(any(s.startswith("Shopify") for s in sugg), sugg)
        self.assertEqual(b.resolve_entity(conn, "Completely Unrelated Topic")[1], "new")

    def test_remember_files_under_existing_entity(self):
        info = b.remember("extra ollama note", "OLLAMA", "Knowledge")
        self.created = [info["id"]]
        self.assertEqual(info["entity"], "Ollama")
        self.assertEqual(info["resolution"], "normalized")

    def test_correct_by_id_keeps_category(self):
        b.correct(None, "MediaTek MT7921 uses the mt7921e kernel driver; never blacklist the btmtk module in /etc/modprobe.d",
                  category=None, fact_id=F["wifi"])
        row = b.get_db().execute("SELECT category FROM facts WHERE id = ?", (F["wifi"],)).fetchone()
        self.assertEqual(row[0], "Rule")

    def test_merge_keeps_fact_ids_and_create_consolidates(self):
        before = facts_of("Shopify Video Zoom") + facts_of("Shopify Video Sound")
        plan = b.okf_merge(["Shopify Video Zoom"], "Shopify Video Sound", dry_run=True)
        self.assertTrue(plan["ok"])
        self.assertEqual(facts_of("Shopify Video Zoom"), [F["shop_zoom"]])  # dry run changed nothing
        res = b.okf_merge(["Shopify Video Zoom", "Shopify Video Sound"], "Shopify Horizon Video", create=True)
        self.assertTrue(res["ok"], res)
        self.assertEqual(sorted(facts_of("Shopify Horizon Video")), sorted(before))
        self.assertEqual(res["target"], "topics/shopify-horizon-video")


class OkfAndQuickMapTests(unittest.TestCase):
    def test_bundle_is_conformant(self):
        b.okf_build(embed=True, force=True)
        val = b.okf_validate(b.OKF_DIR)
        self.assertTrue(val["conformant"], val["errors"])
        root_meta, _ = b.parse_frontmatter((b.OKF_DIR / "index.md").read_text())
        self.assertEqual(root_meta["okf_version"], b.OKF_VERSION)
        self.assertTrue((b.OKF_DIR / "topics" / "starship.md").exists())
        self.assertTrue((b.OKF_DIR / "log.md").exists())

    def test_quickmap_routes_and_has_relevance_floor(self):
        qm = b.quick_map("starship git timeout in fish")
        self.assertEqual(qm["concepts"][0]["title"], "Starship")
        self.assertEqual(b.quick_map("zzqxv wqpfk")["concepts"], [])

    def test_deprecated_concept_is_flagged(self):
        conn = b.get_db()
        b.set_okf_meta(conn, "starship", {"status": "deprecated"})
        try:
            b.okf_build(embed=True)
            qm = b.quick_map("starship git timeout in fish")
            star = next(c for c in qm["concepts"] if c["title"] == "Starship")
            self.assertIn("deprecated", star["flags"])
        finally:
            b.set_okf_meta(conn, "starship", {"status": ""})
            b.okf_build(embed=True)

    def test_budgeted_quickmap_fits(self):
        qm = b.quick_map("shopify video zoom sound")
        for tokens in (400, 150, 60):
            self.assertLessEqual(len(b.render_quick_map_fitted(qm, tokens)), tokens * 4)


class EnvironmentDriftTests(unittest.TestCase):
    def test_old_kernel_flags_system_facts_only_until_reverified(self):
        conn = b.get_db()
        with conn:
            for key in ("wifi", "starship"):
                b.record_fact_env(conn, F[key], source="test", env={"kernel": "7.1.8-arch1-1", "nvidia": None})
        rows = [dict(r) for r in conn.execute("SELECT id, fact FROM facts WHERE id IN (?, ?)", (F["wifi"], F["starship"]))]
        drift = b.fact_drift_map(conn, rows, ENV)
        self.assertIn(F["wifi"], drift)          # kernel driver rule, recorded on 7.1
        self.assertNotIn(F["starship"], drift)   # shell config: kernel-independent
        b.reverify_facts([F["wifi"]])
        self.assertNotIn(F["wifi"], b.fact_drift_map(conn, rows, ENV))

    def test_backfill_from_pacman_log(self):
        log = TMP / "pacman.log"
        log.write_text(
            "[2020-01-01T10:00:00+0000] [ALPM] installed linux (7.0.1.arch1-1)\n"
            "[2020-01-01T10:00:01+0000] [ALPM] installed nvidia-open (600.10.01-1)\n"
            "[2030-01-01T10:00:00+0000] [ALPM] upgraded linux (7.0.1.arch1-1 -> 9.9.9.arch1-1)\n")
        conn = b.get_db()
        with conn:
            conn.execute("DELETE FROM fact_env WHERE fact_id = ?", (F["ollama"],))
        saved = b.PACMAN_LOG
        try:
            b.PACMAN_LOG = log
            self.assertGreaterEqual(b.backfill_fact_env(conn), 1)
        finally:
            b.PACMAN_LOG = saved
        row = conn.execute("SELECT kernel, nvidia, source FROM fact_env WHERE fact_id = ?", (F["ollama"],)).fetchone()
        self.assertEqual(tuple(row), ("7.0.1-arch1-1", "600.10.01", "pacman-log"))


class SourceTests(unittest.TestCase):
    def test_directory_source_file_set(self):
        rels = {str(p.relative_to(PROJECT)) for p in b.iter_source_files(PROJECT)}
        self.assertIn("README.md", rels)
        self.assertIn(".agents/project_map.md", rels)
        for excluded in (".venv/lib/junk.md", "node_modules/pkg/README.md", "secret-notes/notes.md", "big.md"):
            self.assertNotIn(excluded, rels)

    def test_register_drops_adhoc_junk_and_remove_purges_unregistered(self):
        junk = PROJECT / ".venv" / "lib" / "junk.md"
        b.ingest_file(junk)  # ad-hoc ingest, like the old ~/.cargo pollution
        conn = b.get_db()
        count = lambda path: conn.execute("SELECT COUNT(*) FROM chunks WHERE file_path = ?", (str(path.resolve()),)).fetchone()[0]
        self.assertGreater(count(junk), 0)
        ok, msg, _ = b.add_source(PROJECT)
        self.assertTrue(ok, msg)
        self.assertEqual(count(junk), 0)
        self.assertGreater(count(PROJECT / "README.md"), 0)
        other = HOME / "scratch-docs"
        other.mkdir()
        (other / "a.md").write_text("# scratch\n\nsome scratch notes for purge testing\n")
        b.ingest_file(other / "a.md")
        ok, msg, purged = b.remove_source(other)
        self.assertTrue(ok, msg)
        self.assertGreater(purged, 0)
        b.remove_source(PROJECT)

    def test_identical_files_do_not_collide(self):
        a, c = HOME / "copy-a", HOME / "copy-b"
        for d in (a, c):
            d.mkdir()
            (d / "README.md").write_text("# Same\n\nidentical content in two different places\n")
        self.assertGreater(b.ingest_file(a / "README.md"), 0)
        self.assertGreater(b.ingest_file(c / "README.md"), 0)


class OutputBudgetTests(unittest.TestCase):
    def test_inject_fits_budget_and_keeps_directives(self):
        for tokens in (800, 400, 250):
            ctx = b.generate_role_context("hardware", PROJECT, tokens)
            self.assertLessEqual(len(ctx), tokens * 4)
            self.assertIn("## 5. Directives", ctx)

    def test_apply_token_budget_counts_notice(self):
        self.assertLessEqual(len(b.apply_token_budget("line\n" * 500, 50)), 200)


class McpTests(unittest.TestCase):
    def run_mcp(self, messages):
        stdin, stdout = sys.stdin, sys.stdout
        sys.stdin = io.StringIO("".join(json.dumps(m) + "\n" for m in messages))
        sys.stdout = io.StringIO()
        try:
            b.run_mcp_server()
            out = sys.stdout.getvalue()
        finally:
            sys.stdin, sys.stdout = stdin, stdout
        return [json.loads(line) for line in out.splitlines() if line.strip()]

    def test_protocol(self):
        replies = self.run_mcp([
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "ping"},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 4, "method": "resources/list"},
            {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "brain_quickmap", "arguments": {"query": "starship timeout", "max_tokens": 200}}},
        ])
        by_id = {r["id"]: r for r in replies}
        self.assertEqual(len(replies), 5)  # the notification is not answered
        self.assertEqual(by_id[1]["result"]["protocolVersion"], "2025-06-18")
        self.assertEqual(by_id[2]["result"], {})
        self.assertIn("brain_quickmap", [t["name"] for t in by_id[3]["result"]["tools"]])
        self.assertEqual(by_id[4]["error"]["code"], -32601)
        self.assertLessEqual(len(by_id[5]["result"]["content"][0]["text"]), 800)


if __name__ == "__main__":
    unittest.main()
