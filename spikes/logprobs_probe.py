"""Phase 0 spike: do logprobs come back through Mellea's OpenAIBackend, and where is the answer?

Usage:
    uv run --env-file .env python spikes/logprobs_probe.py openrouter meta-llama/llama-3.3-70b-instruct
    uv run --env-file .env python spikes/logprobs_probe.py hf            # model from $MODEL_ID

Prints, per probe, the generated text and the top logprobs at each generated position.
The prompt text here is throwaway spike scaffolding, not a prompt config.
"""

import asyncio
import json
import math
import os
import sys
import time

from mellea.backends.model_options import ModelOption
from mellea.backends.openai import OpenAIBackend
from mellea.stdlib.components.chat import Message
from mellea.stdlib.context.chat import ChatContext

SYSTEM = "You answer questions about a state. Reply with only the label of your answer."
PROBES = {
    "noul": (
        "State:\nHelp! My payouts have been failing for 3 days.\n\n"
        "Question: Does this convey urgency?\nYes. Explicitly time-sensitive\n"
        "No. No urgency expressed\n\nAnswer with Yes or No."
    ),
    "choice": (
        "State:\nHelp! My payouts have been failing for 3 days.\n\n"
        "Question: Which team should handle this?\nA. billing: Payments, invoicing, refunds\n"
        "B. technical: Bugs, outages, integrations\nC. sales\n\nAnswer with A, B, or C."
    ),
}


def make_backend(kind: str, model_id: str | None) -> OpenAIBackend:
    if kind == "openrouter":
        return OpenAIBackend(
            model_id=model_id,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            default_extra_body={"provider": {"require_parameters": True}},
        )
    if kind == "hf":
        return OpenAIBackend(
            model_id=model_id or os.environ["MODEL_ID"],
            base_url=os.environ["HF_ENDPOINT_URL"],
            api_key=os.environ["HF_TOKEN"],
        )
    raise SystemExit(f"unknown backend {kind!r}")


def summarize(raw: dict) -> dict:
    choice = raw["choices"][0]
    positions = []
    for pos in (choice.get("logprobs") or {}).get("content") or []:
        top = pos.get("top_logprobs") or []
        positions.append(
            {
                "token": pos["token"],
                "top": [(t["token"], round(math.exp(t["logprob"]), 4)) for t in top],
            }
        )
    return {
        "text": choice["message"]["content"],
        "finish_reason": choice.get("finish_reason"),
        "n_top": len(positions[0]["top"]) if positions else 0,
        "positions": positions,
        "usage": raw.get("usage"),
        "provider": raw.get("provider"),
        "model": raw.get("model"),
    }


async def run(kind: str, model_id: str | None, top_k: int, temperature: float) -> None:
    backend = make_backend(kind, model_id)
    for name, user in PROBES.items():
        ctx = ChatContext().add(Message("system", SYSTEM))
        opts = {
            ModelOption.MAX_NEW_TOKENS: 3,
            ModelOption.TEMPERATURE: temperature,
            "logprobs": True,
            "top_logprobs": top_k,
        }
        t0 = time.perf_counter()
        mot, _ = await backend.generate_from_context(Message("user", user), ctx, model_options=opts)
        await mot.avalue()
        dt = time.perf_counter() - t0
        raw = mot.raw.response
        if not isinstance(raw, dict):
            raw = raw.model_dump()
        out = {"probe": name, "temperature": temperature, "top_k": top_k, "seconds": round(dt, 2)}
        out.update(summarize(raw))
        positions = out.pop("positions")
        usage = out.pop("usage") or {}
        out["tokens"] = (usage.get("prompt_tokens"), usage.get("completion_tokens"))
        out["cached"] = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        print(json.dumps(out))
        for i, pos in enumerate(positions):
            print(f"  pos {i} {pos['token']!r}: {pos['top'][:8]}")


if __name__ == "__main__":
    kind = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else None
    top_k = int(os.environ.get("TOP_K", "20"))
    temp = float(os.environ.get("TEMPERATURE", "0"))
    asyncio.run(run(kind, model, top_k, temp))
