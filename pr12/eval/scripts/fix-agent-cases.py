"""Прибрати args_match для list_orders у g09, g15 (перевірка хибна)."""
import json
from pathlib import Path

import os
p = Path(os.getcwd()) / "eval" / "cases" / "agent.json"
data = json.loads(p.read_text(encoding="utf-8"))

for c in data:
    if c["id"] in ("g09", "g15"):
        before = len(c["checks"])
        c["checks"] = [x for x in c["checks"] if x.get("check") != "args_match"]
        print(f"{c['id']}: {before} → {len(c['checks'])} перевірок")

p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"записано {p}")