#!/usr/bin/env python3
"""Retrieval-accuracy evaluation against a *copy* of a real brain (never the live DB itself).

Copies <brain-dir>/db/brain.db (via the SQLite online-backup API), facts.json and sources.json into a
temporary sandbox, points this repo's brain.py at it, and scores:
  * facts:    hit@5 and MRR of expected fact IDs for natural-language questions
  * concepts: hit@1 / hit@3 of expected quick-map concepts (matched by the concept's slug)

Cases live outside the repo because they describe personal data (default: <brain-dir>/eval_cases.json):
  {"facts":    [{"q": "wifi driver suspend", "expect": [249, 250]}],
   "concepts": [{"q": "ollama gpu layers", "expect": ["ollama"]}]}

Usage:  python3 tests/live_eval.py [--brain-dir ~/.central_brain] [--cases FILE] [--min-hit 0.9]
Exit status is 1 when a metric falls below --min-hit.
"""
import argparse
import importlib.util
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--brain-dir", default=str(Path.home() / ".central_brain"))
    ap.add_argument("--cases", default=None)
    ap.add_argument("--min-hit", type=float, default=0.9)
    ap.add_argument("--keep", action="store_true", help="Keep the sandbox directory for inspection")
    args = ap.parse_args()

    src = Path(args.brain_dir).expanduser().resolve()
    cases = json.loads(Path(args.cases or src / "eval_cases.json").read_text())
    sandbox = Path(tempfile.mkdtemp(prefix="brain-eval-"))
    (sandbox / "db").mkdir()
    for d in ("knowledge", "projects", "episodes"):
        (sandbox / d).mkdir()
    live = sqlite3.connect(f"file:{src / 'db' / 'brain.db'}?mode=ro", uri=True)
    copy = sqlite3.connect(sandbox / "db" / "brain.db")
    live.backup(copy)
    live.close()
    copy.close()
    for name in ("facts.json", "sources.json"):
        if (src / name).exists():
            shutil.copy2(src / name, sandbox / name)

    os.environ["CENTRAL_BRAIN_DIR"] = str(sandbox)
    spec = importlib.util.spec_from_file_location("brain_eval", ROOT / "brain.py")
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    b.okf_build(embed=True)

    failed = False
    fc = cases.get("facts", [])
    if fc:
        hits, mrr, t0 = 0, 0.0, time.time()
        for c in fc:
            ids = [f["id"] for f in b.search_brain(c["q"], top_k=5, facts_only=True)["facts"]]
            rank = next((i + 1 for i, x in enumerate(ids) if x in set(c["expect"])), None)
            hits += bool(rank)
            mrr += 1.0 / rank if rank else 0.0
            if not rank:
                print(f"  fact miss: {c['q']!r} -> {ids}")
        n = len(fc)
        print(f"facts:    hit@5 {hits}/{n}  MRR {mrr / n:.3f}  ({(time.time() - t0) / n * 1000:.0f} ms/query)")
        failed |= hits / n < args.min_hit

    cc = cases.get("concepts", [])
    if cc:
        def slug(c):
            cid = c["concept_id"]
            return cid.split("/")[-2] if cid.endswith("/overview") else cid.split("/")[-1]
        h1 = h3 = 0
        for c in cc:
            got = [slug(x) for x in b.quick_map(c["q"], top_n=3)["concepts"]]
            exp = set(c["expect"])
            h1 += bool(got[:1]) and got[0] in exp
            h3 += any(g in exp for g in got[:3])
            if not any(g in exp for g in got[:3]):
                print(f"  concept miss: {c['q']!r} -> {got}")
        n = len(cc)
        print(f"concepts: hit@1 {h1}/{n}  hit@3 {h3}/{n}")
        failed |= h3 / n < args.min_hit

    if args.keep:
        print(f"sandbox kept at {sandbox}")
    else:
        shutil.rmtree(sandbox, ignore_errors=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
