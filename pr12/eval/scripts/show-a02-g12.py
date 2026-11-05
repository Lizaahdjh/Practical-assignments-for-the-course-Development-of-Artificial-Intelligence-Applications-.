"""Показати відповіді a02 (drafts) і g12 (agent) — витяг інструкції."""
import json
from pathlib import Path

RUNS = Path("eval") / "runs"

def show(run_file: str, case_id: str) -> None:
    path = RUNS / run_file
    if not path.exists():
        print(f"НЕМАЄ: {path}")
        return
    d = json.loads(path.read_text(encoding="utf-8"))
    print(f"\n=== {case_id} ({run_file}) ===")
    for r in d["results"]:
        if r["id"] != case_id:
            continue
        text = r["output"].get("text") or "(немає)"
        print(f"--- attempt {r['attempt']}, passed={r['passed']}:")
        print(text[:400])
        failed = [c for c in r["checks"] if not c["passed"]]
        if failed:
            print(f"  FAILED: {[(c['name'], c['detail'][:100]) for c in failed]}")


show("drafts-v2.json", "a02")
show("agent-v1.json", "g12")