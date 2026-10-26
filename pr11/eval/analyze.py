"""Аналіз журналу: повільні, невдалі, повторні, витрати, кеш.

Запуск:
    python eval/analyze.py
    python eval/analyze.py logs/app.jsonl --since 2026-10-23 --section slow
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Примусово UTF-8, щоб українські літери не ламалися в консолі Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load(path: Path, since: str | None = None) -> list[dict]:
    records: list[dict] = []
    broken = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                broken += 1
                continue
            if since and str(record.get("ts", "")) < since:
                continue
            records.append(record)
    if broken:
        print(f"(пропущено рядків, що не розбираються: {broken})",
              file=sys.stderr)
    return records


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return ordered[index]


def group(records: list[dict], key: str) -> dict:
    groups: dict = defaultdict(list)
    for record in records:
        groups[record.get(key)].append(record)
    return groups


def summary(records: list[dict]) -> None:
    if not records:
        print("записів немає")
        return
    stamps = [r.get("ts", "") for r in records]
    print(f"записів: {len(records)}; від {min(stamps)} до {max(stamps)}")
    print("рівні:", dict(Counter(r.get("level") for r in records)))
    print("події:")
    for event, count in Counter(r.get("event") for r in records).most_common():
        print(f"  {count:>6}  {event}")


def slow(records: list[dict]) -> None:
    attempts = [r for r in records
                if r.get("event") == "llm.attempt"
                and r.get("status") == "ok"
                and not r.get("cached")]
    durations = [r["duration_ms"] for r in attempts
                 if isinstance(r.get("duration_ms"), (int, float))]
    if not durations:
        print("успішних викликів немає")
        return
    print(f"викликів: {len(durations)}")
    print(f"тривалість, мс: медіана {percentile(durations, 0.5):.0f}, "
          f"p95 {percentile(durations, 0.95):.0f}, "
          f"max {max(durations):.0f}")

    print("\n10 найдовших:")
    for r in sorted(attempts,
                    key=lambda x: x.get("duration_ms", 0),
                    reverse=True)[:10]:
        print(f"  {r.get('duration_ms', 0):>7.0f} мс  "
              f"{r.get('provider')}/{r.get('model')}  "
              f"draft={r.get('draft_id')}  attempt={r.get('attempt')}  "
              f"queue={r.get('queue_ms', '—')}  "
              f"first={r.get('first_token_ms', '—')}")

    queues = [r["queue_ms"] for r in attempts
              if isinstance(r.get("queue_ms"), (int, float))]
    firsts = [r["first_token_ms"] for r in attempts
              if isinstance(r.get("first_token_ms"), (int, float))]
    if queues:
        print(f"\nчерга, мс: медіана {percentile(queues, 0.5):.0f}, "
              f"p95 {percentile(queues, 0.95):.0f}")
    if firsts:
        print(f"до першого фрагмента, мс: "
              f"медіана {percentile(firsts, 0.5):.0f}, "
              f"p95 {percentile(firsts, 0.95):.0f}")
        after = []
        for r in attempts:
            d = r.get("duration_ms")
            ft = r.get("first_token_ms")
            if isinstance(d, (int, float)) and isinstance(ft, (int, float)):
                after.append(d - ft)
        if after:
            print(f"після першого фрагмента, мс: "
                  f"медіана {percentile(after, 0.5):.0f}, "
                  f"p95 {percentile(after, 0.95):.0f}")

    print("\nза провайдером:")
    for prov, rows in group(attempts, "provider").items():
        ds = [r["duration_ms"] for r in rows
              if isinstance(r.get("duration_ms"), (int, float))]
        if ds:
            print(f"  {prov}: n={len(ds)} "
                  f"медіана {percentile(ds, 0.5):.0f} мс, "
                  f"p95 {percentile(ds, 0.95):.0f} мс")


def failed(records: list[dict]) -> None:
    failed_attempts = [r for r in records
                       if r.get("event") == "llm.attempt"
                       and r.get("status") == "error"]
    print(f"невдалих спроб: {len(failed_attempts)}")
    by_kind = Counter(r.get("kind") for r in failed_attempts)
    print("за видом збою:")
    for kind, count in by_kind.most_common():
        print(f"  {count:>4}  {kind}")

    first_attempt_fail = set()
    for r in failed_attempts:
        if r.get("attempt") == 1 and r.get("draft_id"):
            first_attempt_fail.add(r["draft_id"])
    drafts_done = {r["draft_id"] for r in records
                   if r.get("event") == "draft.done" and r.get("draft_id")}
    drafts_failed = {r["draft_id"] for r in records
                     if r.get("event") == "draft.failed" and r.get("draft_id")}
    saved = first_attempt_fail & drafts_done
    lost = first_attempt_fail & drafts_failed
    print(f"\nчернеток з невдалою першою спробою: "
          f"{len(first_attempt_fail)}")
    print(f"з них врятовано повторами: {len(saved)}; втрачено: {len(lost)}")

    failed_cost = sum(r.get("cost") or 0 for r in failed_attempts)
    estimated_count = sum(1 for r in failed_attempts
                          if r.get("cost_estimated"))
    failed_tokens = sum((r.get("prompt_tokens") or 0)
                        + (r.get("completion_tokens") or 0)
                        for r in failed_attempts)
    print(f"\nвартість невдалих спроб: {failed_cost:.6f} $ "
          f"(з них оцінка: {estimated_count}; токенів: {failed_tokens})")


def repeated(records: list[dict]) -> None:
    attempts = [r for r in records if r.get("event") == "llm.attempt"]

    attempts_per_draft = Counter(
        r.get("draft_id") for r in attempts if r.get("draft_id")
    )
    multi = {d: n for d, n in attempts_per_draft.items() if n > 1}
    print(f"чернеток із кількома спробами: {len(multi)}")
    for d, n in sorted(multi.items(), key=lambda kv: kv[1],
                       reverse=True)[:10]:
        print(f"  draft={d}: {n} спроб")

    by_client = Counter(r.get("client") for r in records
                        if r.get("event") == "draft.created")
    print("\nзамовлень за клієнтом:")
    for client, count in by_client.most_common(10):
        print(f"  {client}: {count}")

    by_request = Counter(r.get("request") for r in records
                         if r.get("event") == "draft.created")
    print("\nзвернень, замовлених більше одного разу:")
    shown = False
    for rid, count in by_request.most_common(10):
        if count > 1 and rid:
            print(f"  {rid}: {count}")
            shown = True
    if not shown:
        print("  (немає)")


def spend(records: list[dict]) -> None:
    attempts = [r for r in records
                if r.get("event") == "llm.attempt"
                and r.get("status") == "ok"]
    drafts_done = [r for r in records if r.get("event") == "draft.done"]

    total = sum(r.get("cost", 0) or 0 for r in attempts)
    billed = sum(r.get("cost", 0) or 0 for r in attempts
                 if not r.get("cached"))
    total_tokens_in = sum(r.get("prompt_tokens", 0) or 0 for r in attempts)
    total_tokens_out = sum(r.get("completion_tokens", 0) or 0
                           for r in attempts)
    estimated = sum(1 for r in attempts if r.get("cost_estimated"))

    print(f"успішних викликів: {len(attempts)}")
    print(f"токенів: вхід {total_tokens_in}, вихід {total_tokens_out}")
    print(f"вартість усіх чернеток: {total:.6f} $ "
          f"(з кешу безкоштовно: {total - billed:.6f} $)")
    print(f"реально витрачено (без кешу): {billed:.6f} $")
    print(f"з них оцінка: {estimated} записів")
    if drafts_done:
        print(f"середня вартість чернетки: "
              f"{total / len(drafts_done):.6f} $")

    print("\nза моделлю:")
    for model, rows in group(attempts, "model").items():
        s = sum(r.get("cost", 0) or 0 for r in rows)
        print(f"  {model}: n={len(rows)}, {s:.6f} $")

    print("\nза клієнтом:")
    by_client = defaultdict(float)
    for r in attempts:
        by_client[r.get("client")] += r.get("cost", 0) or 0
    for client, cost in sorted(by_client.items(),
                               key=lambda kv: kv[1], reverse=True)[:10]:
        print(f"  {client}: {cost:.6f} $")

    all_attempts = [r for r in records if r.get("event") == "llm.attempt"]
    failed_cost = sum(r.get("cost") or 0 for r in all_attempts
                      if r.get("status") == "error")
    fallback_cost = sum(r.get("cost", 0) or 0 for r in attempts
                        if r.get("fallback"))
    if total > 0:
        print(f"\nчастка на збої: "
              f"{failed_cost / (total + failed_cost) * 100:.1f} %")
        print(f"частка на fallback: {fallback_cost / total * 100:.1f} %")


def cache_report(records: list[dict]) -> None:
    attempts = [r for r in records if r.get("event") == "llm.attempt"]
    hits = [r for r in attempts if r.get("cached")]
    misses = [r for r in attempts if not r.get("cached")]
    created = [r for r in records if r.get("event") == "draft.created"]
    bypassed = [r for r in created if r.get("fresh")]

    total = len(hits) + len(misses)
    print(f"влучань: {len(hits)}, промахів: {len(misses)}, "
          f"обходів: {len(bypassed)}")
    if total:
        print(f"частка влучань: {len(hits) / total * 100:.1f} %")

    saved_calls = len(hits)
    miss_costs = [r.get("cost", 0) or 0 for r in misses if r.get("cost")]
    if miss_costs:
        avg = sum(miss_costs) / len(miss_costs)
        print(f"оцінка заощадженого: {avg * saved_calls:.6f} $ "
              f"({saved_calls} × {avg:.6f})")

    by_request = Counter(r.get("request") for r in created)
    print("\nнайчастіші звернення:")
    shown = False
    for rid, count in by_request.most_common(10):
        if count > 1 and rid:
            print(f"  {rid}: {count} разів")
            shown = True
    if not shown:
        print("  (немає)")


SECTIONS = {
    "summary": summary,
    "slow": slow,
    "failed": failed,
    "repeated": repeated,
    "spend": spend,
    "cache": cache_report,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", nargs="?", default="logs/app.jsonl")
    parser.add_argument("--since", help="ISO, напр. 2026-10-23")
    parser.add_argument("--section", choices=["all", *SECTIONS],
                        default="all")
    args = parser.parse_args()

    path = Path(args.log)
    if not path.exists():
        print(f"журналу немає: {path}", file=sys.stderr)
        return 1
    records = load(path, args.since)

    names = list(SECTIONS) if args.section == "all" else [args.section]
    for name in names:
        print(f"\n=== {name}")
        try:
            SECTIONS[name](records)
        except Exception as exc:
            print(f"(помилка: {type(exc).__name__}: {exc})")
    return 0


if __name__ == "__main__":
    sys.exit(main())