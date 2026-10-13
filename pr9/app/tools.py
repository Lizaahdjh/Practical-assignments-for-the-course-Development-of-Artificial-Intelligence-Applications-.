"""Інструменти помічника: описи, перевірка, виконання.

Той самий модуль, що в ПР8, з однією зміною для ПР9: додано `cancel_order`
(у ПР8 ми свідомо не давали скасування моделі, але workflow ПР9 має його
виконувати). Перевірка власника й санітайз однакові для агента й workflow.
"""

import json
import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional

from jsonschema import Draft202012Validator, ValidationError

from shop import service
from shop.service import (
    ShopError, NotFound, InvalidRequest, PolicyViolation, Unavailable,
)

logger = logging.getLogger(__name__)


@dataclass
class Context:
    customer_id: str


@dataclass
class ToolResult:
    status: str
    content: dict | list | str | None = None
    reason: str | None = None
    arguments: dict | None = None


def _pick_order_fields(order: dict) -> dict:
    return {
        "order_id": order["order_id"],
        "created_at": order["created_at"],
        "status": order["status"],
        "status_label": order.get("status_label", order["status"]),
        "items": [
            {"sku": it["sku"], "name": it["name"],
             "quantity": it["quantity"], "price": it["price"]}
            for it in order.get("items", [])
        ],
        "total": order["total"],
        "delivery": {
            "method": order.get("delivery", {}).get("method"),
            "city": order.get("delivery", {}).get("city"),
            "point": order.get("delivery", {}).get("point"),
            "tracking": order.get("delivery", {}).get("tracking"),
            "delivered_at": order.get("delivery", {}).get("delivered_at"),
            "cost": order.get("delivery", {}).get("cost"),
        },
    }


def _pick_list_row(o: dict) -> dict:
    return {"order_id": o["order_id"], "created_at": o["created_at"],
            "status": o["status"], "status_label": o["status_label"],
            "total": o["total"], "items_count": o["items_count"]}


def _pick_product(p: dict) -> dict:
    desc = p.get("description", "")
    for marker in ("Примітка для AI", "AI-асистент", "assistant:"):
        idx = desc.find(marker)
        if idx != -1:
            desc = desc[:idx].strip()
    return {
        "sku": p["sku"], "name": p["name"], "category": p["category"],
        "price": p["price"], "weight_kg": p["weight_kg"],
        "warranty_months": p["warranty_months"], "returnable": p["returnable"],
        "bulky": p["bulky"], "description": desc,
    }


def specs() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "list_orders",
                "description": "Список замовлень поточного клієнта.",
                "parameters": {"type": "object", "properties": {},
                               "additionalProperties": False},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_order",
                "description": ("Деталі замовлення клієнта за номером. "
                                "Номер — лише цифри."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string", "pattern": r"^\d{3,8}$"},
                    },
                    "required": ["order_id"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_products",
                "description": "Пошук товарів у каталозі за словами.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "category": {"type": "string"},
                        "max_price": {"type": "number"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_product",
                "description": "Картка товару за артикулом.",
                "parameters": {
                    "type": "object",
                    "properties": {"sku": {"type": "string"}},
                    "required": ["sku"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_stock",
                "description": "Наявність товару на складі.",
                "parameters": {
                    "type": "object",
                    "properties": {"sku": {"type": "string"}},
                    "required": ["sku"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "delivery_quote",
                "description": ("Вартість і строк доставки. Методи: branch "
                                "(у відділення), courier, pickup."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "method": {"type": "string",
                                   "enum": ["branch", "courier", "pickup"]},
                        "items": {
                            "type": "array", "minItems": 1,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "sku": {"type": "string"},
                                    "quantity": {"type": "integer", "minimum": 1},
                                },
                                "required": ["sku", "quantity"],
                            },
                        },
                    },
                    "required": ["city", "method", "items"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_return",
                "description": ("Створити заявку на повернення. Причини: "
                                "not_suitable, defect, wrong_item, damaged."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string", "pattern": r"^\d{3,8}$"},
                        "sku": {"type": "string"},
                        "reason": {"type": "string",
                                   "enum": ["not_suitable", "defect",
                                            "wrong_item", "damaged"]},
                        "quantity": {"type": "integer", "minimum": 1},
                        "comment": {"type": "string"},
                    },
                    "required": ["order_id", "sku", "reason"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_returns",
                "description": "Список заявок на повернення за замовленням.",
                "parameters": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "additionalProperties": False,
                },
            },
        },
        # ↓↓↓ ДОДАНО ДЛЯ ПР9: скасування замовлення ↓↓↓
        {
            "type": "function",
            "function": {
                "name": "cancel_order",
                "description": ("Скасувати замовлення. Можливо лише якщо "
                                "замовлення в статусі «Готується»."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string", "pattern": r"^\d{3,8}$"},
                    },
                    "required": ["order_id"],
                    "additionalProperties": False,
                },
            },
        },
    ]


_NAME_TO_FUNC: Dict[str, Callable] = {
    "list_orders": lambda ctx: service.list_orders(ctx.customer_id),
    "get_order": lambda ctx, order_id: service.get_order(order_id),
    "search_products": lambda ctx, query, category=None, max_price=None, limit=10:
        service.search_products(query, category, max_price, limit),
    "get_product": lambda ctx, sku: service.get_product(sku),
    "get_stock": lambda ctx, sku: service.get_stock(sku),
    "delivery_quote": lambda ctx, city, method, items:
        service.delivery_quote(city, method, items),
    "create_return": lambda ctx, order_id, sku, reason, quantity=1, comment="":
        service.create_return(order_id, sku, reason, quantity, comment),
    "list_returns": lambda ctx, order_id=None: service.list_returns(order_id),
    # ↓↓↓ ДОДАНО ДЛЯ ПР9 ↓↓↓
    "cancel_order": lambda ctx, order_id: service.cancel_order(order_id),
}


def _build_validators() -> Dict[str, Draft202012Validator]:
    out = {}
    for spec in specs():
        f = spec["function"]
        out[f["name"]] = Draft202012Validator(f["parameters"])
    return out


_VALIDATORS: Optional[Dict[str, Draft202012Validator]] = None


def _validators() -> Dict[str, Draft202012Validator]:
    global _VALIDATORS
    if _VALIDATORS is None:
        _VALIDATORS = _build_validators()
    return _VALIDATORS


def _ensure_own_order(order_id: str, ctx: Context) -> dict:
    order = service.get_order(order_id)
    if order.get("customer_id") != ctx.customer_id:
        raise NotFound(f"замовлення {order_id} не знайдено")
    return order


def _check_content(name: str, args: dict, ctx: Context) -> None:
    if name == "get_order":
        _ensure_own_order(args["order_id"], ctx)
    elif name == "create_return":
        order = _ensure_own_order(args["order_id"], ctx)
        if not any(it["sku"] == args["sku"] for it in order.get("items", [])):
            raise InvalidRequest(f"товару {args['sku']} немає в замовленні")
    # ↓↓↓ ДОДАНО ДЛЯ ПР9: перевірка власника для скасування ↓↓↓
    elif name == "cancel_order":
        _ensure_own_order(args["order_id"], ctx)


def _sanitize_result(name: str, raw: Any) -> Any:
    if name == "list_orders":
        return [_pick_list_row(o) for o in raw]
    if name == "get_order":
        return _pick_order_fields(raw)
    if name == "search_products":
        return [{"sku": p["sku"], "name": p["name"],
                 "category": p["category"], "price": p["price"]} for p in raw]
    if name == "get_product":
        return _pick_product(raw)
    if name == "get_stock":
        return {"sku": raw["sku"], "available": raw["available"],
                "expected_restock": raw.get("expected_restock")}
    if name == "delivery_quote":
        return {"city": raw["city"], "method": raw["method"],
                "cost": raw["cost"], "days": raw["days"],
                "note": raw.get("note", "")}
    if name == "create_return":
        return {"return_id": raw["return_id"], "order_id": raw["order_id"],
                "sku": raw["sku"], "reason": raw["reason"],
                "status": raw["status"], "next_steps": raw["next_steps"]}
    if name == "list_returns":
        return [{"return_id": r["return_id"], "order_id": r["order_id"],
                 "sku": r["sku"], "reason": r["reason"], "status": r["status"]}
                for r in raw]
    # ↓↓↓ ДОДАНО ДЛЯ ПР9 ↓↓↓
    if name == "cancel_order":
        return {"order_id": raw["order_id"],
                "status": raw["status"],
                "status_label": raw.get("status_label", raw["status"])}
    return raw


def call(name: str, raw_arguments: str, ctx: Context) -> ToolResult:
    """Перевірити й виконати виклик."""
    if name not in _NAME_TO_FUNC:
        return ToolResult(status="rejected",
                          reason=f"інструмент {name!r} невідомий",
                          content={"error": "unknown_tool",
                                   "message": f"Інструмент {name!r} не підтримується."})
    try:
        args = json.loads(raw_arguments) if raw_arguments else {}
    except (json.JSONDecodeError, TypeError) as e:
        return ToolResult(status="rejected",
                          reason=f"аргументи не є JSON: {e}",
                          content={"error": "bad_json",
                                   "message": "Аргументи не є JSON."})
    if not isinstance(args, dict):
        return ToolResult(status="rejected", reason="не JSON-обʼєкт",
                          content={"error": "bad_json",
                                   "message": "Аргументи — JSON-обʼєкт."})

    validator = _validators()[name]
    try:
        validator.validate(args)
    except ValidationError as e:
        path = ".".join(str(p) for p in e.absolute_path) or "?"
        return ToolResult(status="rejected",
                          reason=f"аргументи не за схемою: {path}: {e.message}",
                          arguments=args,
                          content={"error": "bad_schema",
                                   "message": f"Аргумент «{path}»: {e.message}"})

    try:
        _check_content(name, args, ctx)
    except NotFound as e:
        return ToolResult(status="rejected", reason=str(e), arguments=args,
                          content={"error": "not_found", "message": str(e)})
    except ShopError as e:
        return ToolResult(status="rejected", reason=str(e), arguments=args,
                          content={"error": e.code, "message": str(e)})

    func = _NAME_TO_FUNC[name]
    try:
        raw = func(ctx, **args)
    except NotFound as e:
        return ToolResult(status="error", reason=str(e), arguments=args,
                          content={"error": "not_found", "message": str(e)})
    except InvalidRequest as e:
        return ToolResult(status="error", reason=str(e), arguments=args,
                          content={"error": "invalid_request", "message": str(e)})
    except PolicyViolation as e:
        return ToolResult(status="error", reason=str(e), arguments=args,
                          content={"error": "policy", "message": str(e)})
    except Unavailable as e:
        return ToolResult(status="error", reason=str(e), arguments=args,
                          content={"error": "unavailable",
                                   "message": "Сервіс тимчасово недоступний."})
    except ShopError as e:
        return ToolResult(status="error", reason=str(e), arguments=args,
                          content={"error": e.code, "message": str(e)})
    except Exception as e:
        logger.exception("Помилка в інструменті %s: %s", name, e)
        return ToolResult(status="error", reason=f"{type(e).__name__}: {e}",
                          arguments=args,
                          content={"error": "internal",
                                   "message": "Внутрішня помилка інструмента."})

    try:
        safe = _sanitize_result(name, raw)
    except Exception:
        safe = raw
    return ToolResult(status="ok", content=safe, arguments=args)