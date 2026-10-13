"""Цикл помічника ПР8: агентський підхід для порівняння з workflow."""

import json
import logging
import os
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv

from . import llm, tools

load_dotenv()
logger = logging.getLogger(__name__)

MAX_ROUNDS = int(os.getenv("TOOL_MAX_ROUNDS", "3"))


@dataclass
class ToolTrace:
    round: int
    name: str
    arguments: str
    status: str
    reason: str | None = None
    result: dict | list | str | None = None
    elapsed: float | None = None


@dataclass
class Answer:
    text: str
    calls: list[ToolTrace] = field(default_factory=list)
    rounds: int = 0
    stopped: str = "answer"
    model: str | None = None
    elapsed: dict = field(default_factory=dict)
    usage: dict | None = None


_FALLBACK = ("Вибачте, не вдалося завершити обробку за обмежений час. "
             "Зверніться, будь ласка, до оператора підтримки.")


def _message_to_dict(msg) -> dict:
    if hasattr(msg, "model_dump"):
        return msg.model_dump(exclude_none=True)
    out = {"role": "assistant", "content": msg.content or ""}
    if getattr(msg, "tool_calls", None):
        out["tool_calls"] = [
            {"id": c.id, "type": "function",
             "function": {"name": c.function.name, "arguments": c.function.arguments}}
            for c in msg.tool_calls
        ]
    return out


def answer(question: str, customer_id: str) -> Answer:
    question = (question or "").strip()
    if not question:
        return Answer(text="Будь ласка, напишіть питання.", stopped="empty",
                      elapsed={"model": 0.0, "tools": 0.0},
                      usage={"prompt_tokens": 0, "completion_tokens": 0})

    ctx = tools.Context(customer_id=customer_id)
    specs = tools.specs()
    messages = llm.build_messages(question)

    traces: list[ToolTrace] = []
    total_prompt = total_completion = 0
    model_time = tools_time = 0.0
    rounds = 0
    model_name: str | None = None

    for round_num in range(1, MAX_ROUNDS + 1):
        rounds = round_num
        try:
            resp = llm.chat(messages, tools=specs, tool_choice="auto")
        except llm.LLMError:
            try:
                resp = llm.chat(messages, tools=[], tool_choice="none")
            except llm.LLMError:
                raise
        model_time += resp["elapsed"]
        model_name = resp["model"]
        total_prompt += resp["usage"]["prompt_tokens"]
        total_completion += resp["usage"]["completion_tokens"]

        msg = resp["message"]
        tool_calls = getattr(msg, "tool_calls", None) or []

        if not tool_calls:
            text = (msg.content or "").strip() or _FALLBACK
            return Answer(
                text=text, calls=traces, rounds=rounds, stopped="answer",
                model=model_name,
                elapsed={"model": round(model_time, 3),
                         "tools": round(tools_time, 3)},
                usage={"prompt_tokens": total_prompt,
                       "completion_tokens": total_completion},
            )

        messages.append(_message_to_dict(msg))

        for call in tool_calls:
            t0 = time.perf_counter()
            result = tools.call(call.function.name, call.function.arguments, ctx)
            elapsed = time.perf_counter() - t0
            tools_time += elapsed

            traces.append(ToolTrace(
                round=round_num, name=call.function.name,
                arguments=call.function.arguments or "",
                status=result.status, reason=result.reason,
                result=result.content, elapsed=round(elapsed, 4),
            ))
            messages.append({
                "role": "tool", "tool_call_id": call.id,
                "content": json.dumps(result.content, ensure_ascii=False)
                           if result.content is not None else "",
            })

    try:
        final = llm.chat(messages, tools=[], tool_choice="none")
        model_time += final["elapsed"]
        total_prompt += final["usage"]["prompt_tokens"]
        total_completion += final["usage"]["completion_tokens"]
        text = (final["message"].content or "").strip() or _FALLBACK
    except llm.LLMError:
        text = _FALLBACK

    return Answer(text=text, calls=traces, rounds=rounds, stopped="limit",
                  model=model_name,
                  elapsed={"model": round(model_time, 3),
                           "tools": round(tools_time, 3)},
                  usage={"prompt_tokens": total_prompt,
                         "completion_tokens": total_completion})