"""Зібрати кілька прогонів в один файл для compare.py."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
while HERE.name != "pr12" and HERE.parent != HERE:
    HERE = HERE.parent
RUNS = HERE / "eval" / "runs"


def merge(parts: list[str], out: str, label: str) -> None:
    all_data = {"meta": {}, "cases": [], "results": []}
    for p in parts:
        path = RUNS / p
        if not path.exists():
            print(f"НЕМАЄ: {path}")
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        all_data["cases"].extend(d["cases"])
        all_data["results"].extend(d["results"])
    if not all_data["results"]:
        print(f"{out}: нічого не зібрано")
        return
    first = json.loads((RUNS / parts[0]).read_text(encoding="utf-8"))
    all_data["meta"] = first["meta"]
    all_data["meta"]["label"] = label
    (RUNS / out).write_text(
        json.dumps(all_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{out}: {len(all_data['results'])} спроб")


merge(
    ["drafts-v1a.json", "drafts-v1-negative.json", "drafts-v1-adversarial.json"],
    "drafts-v1.json",
    "v1 (зібрано з груп)",
)

merge(
    [f"drafts-v2-group{i}-{f}.json" for i, f in enumerate(
        ["d01", "e02", "d02", "n03", "a05", "e08"], 1)],
    "drafts-v2.json",
    "v2 (посилена інструкція)",
)