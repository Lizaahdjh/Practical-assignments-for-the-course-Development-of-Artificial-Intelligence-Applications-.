"""Workflow обробки звернення клієнта: кроки, переходи, обмеження.

Рішення:

* кроки: parse (модель, structured output) → check (код) → fetch
  (інструменти, читання) → route (код, вибір гілки) → prepare_action
  (код, формування дії) → пауза на підтвердження → execute (інструмент,
  один раз) → reply (модель, за фактами) → перевірка відповіді кодом;
* обмеження: WORKFLOW_MAX_STEPS, WORKFLOW_TIMEOUT, WORKFLOW_TOKEN_BUDGET;
* статуси: running / awaiting_approval / done / rejected / needs_info /
  handed_off / stopped / failed; переходи перевіряються;
* пауза: звернення зі статусом awaiting_approval переживає перезапуск;
* один раз: `execute` перевіряє статус перед дією, а після — зберігає
  результат. Повторне підтвердження бачить статус уже не awaiting_approval
  і нічого не робить;
* клієнт — з сеансу, не з тексту; інструменти отримують `Context`.
"""

import os
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv

from . import llm, schema, store, tools
from .state import Run, Step

load_dotenv()

MAX_STEPS = int(os.getenv("WORKFLOW_MAX_STEPS", "12"))
TIMEOUT = float(os.getenv("WORKFLOW_TIMEOUT", "60"))
TOKEN_BUDGET = int(os.getenv("WORKFLOW_TOKEN_BUDGET", "8000"))

MAX_REPLY_RETRIES = 1


def limits() -> dict:
    return {"max_steps": MAX_STEPS, "timeout": TIMEOUT, "token_budget": TOKEN_BUDGET}


# ---------------------------------------------------------------------------
# Хелпери
# ---------------------------------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_run(text: str, customer_id: str) -> Run:
    return Run(
        run_id=f"run-{uuid.uuid4().hex[:8]}",
        customer_id=customer_id,
        text=text.strip(),
        status="running",
        created_at=_now(),
        updated_at=_now(),
        elapsed={"model": 0.0, "tools": 0.0},
        usage={"prompt_tokens": 0, "completion_tokens": 0},
    )


def _append(run: Run, step: Step) -> None:
    step.n = len(run.steps) + 1
    step.at = _now()
    run.steps.append(step)
    run.updated_at = _now()


def _total_tokens(run: Run) -> int:
    u = run.usage or {}
    return int(u.get("prompt_tokens", 0)) + int(u.get("completion_tokens", 0))


def _check_limits(run: Run, started: float) -> Optional[str]:
    """Повертає причину зупинки або None."""
    if len(run.steps) >= MAX_STEPS:
        return "steps"
    if time.perf_counter() - started > TIMEOUT:
        return "timeout"
    if _total_tokens(run) >= TOKEN_BUDGET:
        return "tokens"
    return None


def _stop(run: Run, reason: str) -> None:
    run.status = "stopped"
    run.stopped = reason
    run.reply = (
        f"Вибачте, не встигли обробити звернення за відведений ліміт "
        f"({reason}). Спробуйте, будь ласка, пізніше або зверніться до "
        f"оператора підтримки."
    )
    store.save(run)


# ---------------------------------------------------------------------------
# Крок 1 — parse
# ---------------------------------------------------------------------------
PARSE_PROMPT = """Ти розбираєш письмове звернення клієнта магазину «Сузірʼя».

Поверни ЛИШЕ JSON-обʼєкт із полями:
- kind: "return" (повернути товар), "cancel" (скасувати замовлення),
  "status" (питання про стан замовлення) або "other" (усе інше);
- order_id: номер замовлення, якщо клієнт його назвав (лише цифри), інакше null;
- item: назва товару, як клієнт її написав, або null;
- reason: "not_suitable" / "defect" / "wrong_item" / "damaged" — лише
  для kind="return", інакше null. Якщо клієнт не назвав причину, постав null.

Не вигадуй номера замовлення, якщо клієнт його не назвав. Не додумуй
товар, якщо його немає в тексті. Якщо не розумієш звернення — постав
kind="other".

Звернення клієнта:
"""


def _step_parse(run: Run, started: float) -> None:
    t0 = time.perf_counter()
    messages = [{"role": "user", "content": PARSE_PROMPT + run.text}]
    result = llm.complete(messages, schema=schema.request_schema())
    run.elapsed["model"] = run.elapsed.get("model", 0.0) + result["elapsed"]
    run.usage["prompt_tokens"] += result["usage"]["prompt_tokens"]
    run.usage["completion_tokens"] += result["usage"]["completion_tokens"]

    try:
        parsed = schema.validate(result["text"])
    except schema.SchemaError as e:
        _append(run, Step(
            name="parse", kind="model", status="error",
            detail=f"відповідь не за схемою: {e.detail}",
            input=run.text, output=result["text"][:500],
            elapsed=result["elapsed"], usage=result["usage"],
        ))
        run.status = "failed"
        run.stopped = "invalid_parse"
        run.reply = ("Вибачте, не вдалося розібрати звернення. "
                     "Уточніть, будь ласка, про що йдеться.")
        store.save(run)
        raise _Stop()

    run.data["parsed"] = parsed
    run.route = parsed["kind"]
    _append(run, Step(
        name="parse", kind="model", status="ok",
        detail=f"вид={parsed['kind']}",
        input=run.text, output=parsed,
        elapsed=result["elapsed"], usage=result["usage"],
    ))


class _Stop(Exception):
    """Внутрішній сигнал завершити прохід (стан уже збережено)."""


# ---------------------------------------------------------------------------
# Крок 2 — check
# ---------------------------------------------------------------------------
def _step_check(run: Run, started: float) -> None:
    """Перевірити, чого бракує в розібраному зверненні."""
    parsed = run.data.get("parsed") or {}

    if parsed["kind"] == "other":
        # Не зрозуміли — передаємо оператору
        _append(run, Step(
            name="check", kind="code", status="ok",
            detail="вид=other, передаємо оператору",
        ))
        run.status = "handed_off"
        run.reply = ("Ваше звернення передано оператору підтримки. "
                     "Відповідь надійде найближчим часом.")
        store.save(run)
        raise _Stop()

    if not parsed["order_id"]:
        _append(run, Step(
            name="check", kind="code", status="ok",
            detail="номер замовлення не названо — потрібне уточнення",
        ))
        run.status = "needs_info"
        run.reply = ("Уточніть, будь ласка, номер замовлення — "
                     "він складається з кількох цифр і є в листі-підтвердженні.")
        store.save(run)
        raise _Stop()

    _append(run, Step(
        name="check", kind="code", status="ok",
        detail=f"номер={parsed['order_id']}",
    ))


# ---------------------------------------------------------------------------
# Крок 3 — fetch (читання)
# ---------------------------------------------------------------------------
def _step_fetch(run: Run, started: float) -> None:
    ctx = tools.Context(customer_id=run.customer_id)
    parsed = run.data["parsed"]

    # 3.1 — get_order
    t0 = time.perf_counter()
    result = tools.call("get_order", f'{{"order_id": "{parsed["order_id"]}"}}', ctx)
    elapsed = time.perf_counter() - t0
    run.elapsed["tools"] = run.elapsed.get("tools", 0.0) + elapsed
    _append(run, Step(
        name="get_order", kind="tool", status=result.status,
        detail=result.reason,
        input={"order_id": parsed["order_id"]},
        output=result.content,
        elapsed=elapsed,
    ))

    if result.status != "ok":
        # Чуже або неіснуюче — кінцевий стан, не показуємо подробиць
        run.status = "rejected"
        run.reply = ("Не вдалося знайти ваше замовлення з таким номером. "
                     "Перевірте номер у листі-підтвердженні.")
        store.save(run)
        raise _Stop()

    order = result.content
    run.data["order"] = order

    # 3.2 — вибір товару (enum зі складу замовлення), якщо потрібно
    if parsed["kind"] == "return":
        skus = [it["sku"] for it in order.get("items", [])]
        run.data["order_skus"] = skus
        chosen = _pick_item(parsed.get("item"), order)
        if chosen is None:
            run.status = "needs_info"
            run.reply = ("Уточніть, будь ласка, який саме товар із замовлення "
                         "ви хочете повернути: " +
                         ", ".join(it["name"] for it in order["items"]) + ".")
            store.save(run)
            raise _Stop()
        run.data["sku"] = chosen

    # 3.3 — list_returns (для випадку повторного звернення)
    t0 = time.perf_counter()
    returns = tools.call("list_returns", f'{{"order_id": "{parsed["order_id"]}"}}', ctx)
    elapsed = time.perf_counter() - t0
    run.elapsed["tools"] = run.elapsed.get("tools", 0.0) + elapsed
    _append(run, Step(
        name="list_returns", kind="tool", status=returns.status,
        detail=returns.reason,
        input={"order_id": parsed["order_id"]},
        output=returns.content,
        elapsed=elapsed,
    ))
    if returns.status == "ok" and returns.content:
        run.data["existing_returns"] = returns.content


def _pick_item(item_text: Optional[str], order: dict) -> Optional[str]:
    """Обрати артикул із позицій замовлення за текстом клієнта.

    Простий збіг за словами: беремо позицію, у назві якої є хоч одне
    слово з `item_text`. Якщо збігів кілька — повертаємо `None`,
    потрібне уточнення.
    """
    if not item_text:
        # Якщо в замовленні рівно один товар — обираємо його
        items = order.get("items", [])
        if len(items) == 1:
            return items[0]["sku"]
        return None
    words = [w.lower() for w in item_text.split() if len(w) > 2]
    hits = []
    for it in order.get("items", []):
        name = it["name"].lower()
        if any(w in name for w in words):
            hits.append(it["sku"])
    if len(hits) == 1:
        return hits[0]
    return None


# ---------------------------------------------------------------------------
# Крок 4 — route
# ---------------------------------------------------------------------------
def _step_route(run: Run, started: float) -> None:
    parsed = run.data["parsed"]
    kind = parsed["kind"]

    if kind == "status":
        _append(run, Step(name="route", kind="code", status="ok",
                          detail="status — готуємо відповідь без дій"))
        run.data["action"] = None
        return

    if kind == "cancel":
        _append(run, Step(name="route", kind="code", status="ok",
                          detail="cancel — підтвердження оператора"))
        run.data["action"] = {
            "tool": "cancel_order",
            "arguments": {"order_id": parsed["order_id"]},
        }
        return

    if kind == "return":
        # Перевірки правил — до підтвердження
        reason = _infer_reason(parsed.get("reason"), run.text)
        run.data["reason"] = reason

        existing = run.data.get("existing_returns") or []
        same = [r for r in existing if r["sku"] == run.data["sku"]
                and r["reason"] == reason]
        if same:
            _append(run, Step(
                name="route", kind="code", status="ok",
                detail=f"заявка вже існує: {same[0]['return_id']}",
            ))
            run.status = "done"
            run.reply = (f"За цим замовленням уже створено заявку на повернення "
                         f"№{same[0]['return_id']}. Надішліть товар на склад, "
                         f"вказавши номер на упаковці.")
            store.save(run)
            raise _Stop()

        run.data["action"] = {
            "tool": "create_return",
            "arguments": {
                "order_id": parsed["order_id"],
                "sku": run.data["sku"],
                "reason": reason,
                "quantity": 1,
            },
        }
        _append(run, Step(name="route", kind="code", status="ok",
                          detail=f"return — підтвердження оператора, reason={reason}"))
        return


def _infer_reason(model_reason: Optional[str], text: str) -> str:
    """Уточнити причину за словами клієнта.

    Модель може поставити `defect`, коли клієнт каже «не підійшов».
    Тут код узгоджує: якщо в тексті є слова про дефект — defect,
    якщо про «не підійшов» / «не сподобався» — not_suitable.
    """
    t = (text or "").lower()
    defect_markers = ["дефект", "не працює", "зламав", "перестав",
                      "тріснув", "не заряджа", "не вмика"]
    unsuitable_markers = ["не підійшов", "не підійшла", "не підійшли",
                          "не сподобав", "не підходить", "завелик",
                          "замал", "передумав"]

    has_defect = any(m in t for m in defect_markers)
    has_unsuitable = any(m in t for m in unsuitable_markers)

    if has_defect and not has_unsuitable:
        return "defect"
    if has_unsuitable and not has_defect:
        return "not_suitable"
    # Обидва або жодного — довіряємось моделі, інакше not_suitable
    return model_reason or "not_suitable"


# ---------------------------------------------------------------------------
# Крок 5 — prepare_action
# ---------------------------------------------------------------------------
def _step_prepare_action(run: Run, started: float) -> None:
    action = run.data.get("action")
    if not action:
        _append(run, Step(name="prepare_action", kind="code", status="ok",
                          detail="дія не потрібна"))
        return

    order = run.data["order"]
    tool_name = action["tool"]
    args = action["arguments"]

    summary_map = {
        "cancel_order": "Скасувати замовлення",
        "create_return": "Оформити повернення",
    }
    summary = summary_map.get(tool_name, tool_name)

    if tool_name == "create_return":
        sku = args["sku"]
        item = next((it for it in order["items"] if it["sku"] == sku), None)
        if item:
            summary += f" — {item['name']}"

    run.pending = {
        "summary": summary,
        "tool": tool_name,
        "arguments": args,
        "grounds": {
            "text": run.text,
            "route": run.route,
            "order_id": order["order_id"],
        },
    }
    run.status = "awaiting_approval"
    _append(run, Step(
        name="prepare_action", kind="code", status="ok",
        detail=f"чекає підтвердження: {summary}",
    ))
    store.save(run)
    raise _Stop()


# ---------------------------------------------------------------------------
# Крок 6 — execute (після підтвердження)
# ---------------------------------------------------------------------------
def _step_execute(run: Run, started: float) -> None:
    decision = run.decision or {}
    if not decision.get("approve"):
        # Оператор відхилив
        _append(run, Step(
            name="execute", kind="human", status="rejected",
            detail=decision.get("comment") or "оператор відхилив дію",
        ))
        run.status = "rejected"
        run.reply = ("На жаль, оператор не підтвердив цю дію. "
                     "Якщо потрібно — звʼяжіться з підтримкою для деталей.")
        store.save(run)
        raise _Stop()

    action = run.data.get("action")
    if not action:
        _append(run, Step(name="execute", kind="code", status="skipped",
                          detail="дія не потрібна"))
        return

    # Перевірка статусу перед виконанням (захист від подвійного підтвердження)
    if run.data.get("executed"):
        _append(run, Step(name="execute", kind="code", status="ok",
                          detail="дія вже виконана раніше"))
        return

    ctx = tools.Context(customer_id=run.customer_id)
    t0 = time.perf_counter()
    result = tools.call(action["tool"], _json(action["arguments"]), ctx)
    elapsed = time.perf_counter() - t0
    run.elapsed["tools"] = run.elapsed.get("tools", 0.0) + elapsed

    run.data["executed"] = True
    run.data["action_result"] = result.content

    _append(run, Step(
        name=action["tool"], kind="tool", status=result.status,
        detail=result.reason,
        input=action["arguments"],
        output=result.content,
        elapsed=elapsed,
    ))

    if result.status != "ok":
        run.status = "rejected"
        msg = (result.content or {}).get("message", "не вдалося виконати дію")
        run.reply = f"На жаль, {msg}. Спробуйте пізніше або зверніться до оператора."
        store.save(run)
        raise _Stop()


def _json(obj) -> str:
    import json as _j
    return _j.dumps(obj, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Крок 7 — reply
# ---------------------------------------------------------------------------
REPLY_PROMPT = """Ти — помічник магазину «Сузірʼя». Відповідай клієнтові
українською, коротко й по суті, на підставі фактів нижче. Не вигадуй
нічого, чого немає в фактах.

Звернення клієнта: {text}

Вид: {route}
Статус обробки: {status}
{extra}

Пиши відповідь клієнтові. Якщо це повернення — назви номер заявки і що
робити далі. Якщо скасування — підтвердь. Якщо питання про стан — назви
статус і номер відправлення, якщо є.
"""


def _step_reply(run: Run, started: float) -> None:
    order = run.data.get("order")
    extra_parts = []
    if order:
        extra_parts.append(
            f"Замовлення {order['order_id']}, статус: {order.get('status_label')}, "
            f"сума: {order['total']} грн."
        )
        delivery = order.get("delivery") or {}
        if delivery.get("tracking"):
            extra_parts.append(f"Номер відправлення: {delivery['tracking']}.")
        if delivery.get("point"):
            extra_parts.append(f"Пункт: {delivery['point']}.")
    result_action = run.data.get("action_result")
    if result_action:
        if "return_id" in result_action:
            extra_parts.append(
                f"Заявка на повернення: {result_action['return_id']}. "
                f"{result_action.get('next_steps', '')}"
            )
        elif "status_label" in result_action:
            extra_parts.append(
                f"Новий статус замовлення: {result_action['status_label']}."
            )

    prompt = REPLY_PROMPT.format(
        text=run.text, route=run.route or "?",
        status=run.status,
        extra=" ".join(extra_parts),
    )

    t0 = time.perf_counter()
    result = llm.complete([{"role": "user", "content": prompt}])
    run.elapsed["model"] = run.elapsed.get("model", 0.0) + result["elapsed"]
    run.usage["prompt_tokens"] += result["usage"]["prompt_tokens"]
    run.usage["completion_tokens"] += result["usage"]["completion_tokens"]

    reply = result["text"].strip()

    # Перевірка кодом: без службових підрядків
    forbidden = ["internal_note", "card_last4", "customer_id"]
    bad = any(f in reply for f in forbidden)
    if not reply or bad:
        _append(run, Step(
            name="reply", kind="model", status="error",
            detail="відповідь не пройшла перевірку",
            input=prompt, output=reply,
            elapsed=result["elapsed"], usage=result["usage"],
        ))
        # Безпечний шаблон
        reply = ("Ваше звернення оброблено. Якщо потрібні деталі — "
                 "зверніться до оператора підтримки.")
        run.reply = reply
        run.status = "done"
        store.save(run)
        raise _Stop()

    run.reply = reply
    run.status = "done"
    _append(run, Step(
        name="reply", kind="model", status="ok",
        input=prompt, output=reply,
        elapsed=result["elapsed"], usage=result["usage"],
    ))
    store.save(run)


# ---------------------------------------------------------------------------
# Публічні функції
# ---------------------------------------------------------------------------
def start(text: str, customer_id: str) -> Run:
    """Прийняти нове звернення й вести його, доки це можливо без людини."""
    text = (text or "").strip()
    run = _new_run(text, customer_id)
    if not text:
        run.status = "failed"
        run.stopped = "empty"
        run.reply = "Будь ласка, напишіть звернення."
        store.save(run)
        return run

    started = time.perf_counter()
    try:
        _step_parse(run, started)
        _step_check(run, started)
        _step_fetch(run, started)
        _step_route(run, started)
        _step_prepare_action(run, started)
        # Якщо дій немає — відразу reply
        _step_execute(run, started)
        _step_reply(run, started)
    except _Stop:
        pass
    except Exception as e:
        run.status = "failed"
        run.stopped = f"{type(e).__name__}: {e}"
        run.reply = "Внутрішня помилка обробки. Спробуйте пізніше."
        store.save(run)

    return run


def resume(run_id: str, approve: bool, comment: str = "") -> Run:
    """Продовжити звернення після рішення оператора."""
    run = store.load(run_id)
    if run is None:
        raise ValueError(f"звернення {run_id} не знайдено")

    # Дозволені переходи
    if run.status != "awaiting_approval":
        # Друге підтвердження або підтвердження вже завершеного
        return run

    run.decision = {"approve": approve, "comment": comment, "at": _now()}
    run.status = "running"
    store.save(run)

    started = time.perf_counter()
    try:
        _step_execute(run, started)
        _step_reply(run, started)
    except _Stop:
        pass
    except Exception as e:
        run.status = "failed"
        run.stopped = f"{type(e).__name__}: {e}"
        run.reply = "Внутрішня помилка обробки."
        store.save(run)

    return run