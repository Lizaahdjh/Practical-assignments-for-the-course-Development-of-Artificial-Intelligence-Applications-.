"""Зібрати heldout з груп."""
import json
from pathlib import Path

RUNS = Path("eval") / "runs"
parts = [f"drafts-heldout-{h}.json" for h in ("h01", "h02", "h03", "h04")]

all_data = {"meta": {}, "cases": [], "results": []}
for p in parts:
    path = RUNS / p
    if not path.exists():
        print(f"НЕМАЄ: {path}")
        continue
    d = json.loads(path.read_text(encoding="utf-8"))
    all_data["cases"].extend(d["cases"])
    all_data["results"].extend(d["results"])

if all_data["results"]:
    first = json.loads((RUNS / parts[0]).read_text(encoding="utf-8"))
    all_data["meta"] = first["meta"]
    all_data["meta"]["label"] = "heldout (зібрано)"
    (RUNS / "drafts-heldout.json").write_text(
        json.dumps(all_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"drafts-heldout.json: {len(all_data['results'])} спроб")