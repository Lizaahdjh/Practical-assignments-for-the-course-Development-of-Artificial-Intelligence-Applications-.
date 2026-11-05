"""Ціна прогонів v1 і v2 — з перерахованим summary."""
import json
from pathlib import Path

RUNS = Path("eval") / "runs"

for name in ("drafts-v1.json", "drafts-v2.json"):
    path = RUNS / name
    if not path.exists():
        print(f"НЕМАЄ: {path}")
        continue
    d = json.loads(path.read_text(encoding="utf-8"))
    s = d["summary"]["разом"]
    print(f"\n{name}:")
    print(f"  cases={s['cases']}, attempts={s['attempts']}, "
          f"passed={s['attempts_passed']}, errors={s['errors']}")
    print(f"  tokens_avg={s['tokens_avg']}, cost={s['cost_total']} $")
    print(f"  duration_s={d['meta']['duration_s']} (тільки перша група)")
    print(f"  pass^k={s['pass_all']}/{s['cases']}, pass@k={s['pass_any']}/{s['cases']}, flaky={s['flaky']}")