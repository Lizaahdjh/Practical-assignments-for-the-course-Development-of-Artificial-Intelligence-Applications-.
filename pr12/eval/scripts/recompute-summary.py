"""Перерахувати summary для drafts-v1.json і drafts-v2.json."""
import json
import sys
from pathlib import Path

HERE = Path("eval")
sys.path.insert(0, str(HERE))
import report

for name in ("drafts-v1.json", "drafts-v2.json"):
    path = HERE / "runs" / name
    if not path.exists():
        print(f"НЕМАЄ: {path}")
        continue
    d = json.loads(path.read_text(encoding="utf-8"))
    d["summary"] = report.summarize(d)
    path.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    s = d["summary"]["разом"]
    print(f"{name}: cases={s['cases']}, attempts={s['attempts']}, "
          f"tokens_avg={s['tokens_avg']}, cost={s['cost_total']}")