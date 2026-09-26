"""Phase 0 spike: which OpenRouter (model, provider) pairs return usable answer-token logprobs?

For each model, lists the provider endpoints that advertise `top_logprobs`, pins each one
(`provider.order` + `allow_fallbacks: false`), and checks what actually comes back with
max_tokens = 1 and 3.

This is a diagnostic of provider behavior, so it calls OpenRouter's HTTP API directly; the
Mellea path was checked separately (spikes/logprobs_probe.py) and passes the response through
unchanged.

Usage:
    uv run --env-file .env python spikes/openrouter_matrix.py [model ...]
"""

import json
import math
import os
import sys

import httpx

BASE = "https://openrouter.ai/api/v1"
MODELS = [
    "meta-llama/llama-3.3-70b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "google/gemma-3-27b-it",
    "qwen/qwen3-235b-a22b-2507",
    "qwen/qwen3-30b-a3b-instruct-2507",
    "mistralai/mistral-small-3.2-24b-instruct",
    "openai/gpt-4o-mini",
]
MESSAGES = [
    {"role": "system", "content": "Reply with only the label."},
    {
        "role": "user",
        "content": "Which team handles failing payouts?\nA. billing\nB. technical\nC. sales\n"
        "Answer with A, B, or C.",
    },
]
LABELS = {"A", "B", "C"}


def probe(client: httpx.Client, model: str, tag: str, max_tokens: int) -> dict:
    body = {
        "model": model,
        "messages": MESSAGES,
        "max_tokens": max_tokens,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 20,
        "provider": {"order": [tag], "allow_fallbacks": False, "require_parameters": True},
    }
    r = client.post(f"{BASE}/chat/completions", json=body)
    d = r.json()
    if "choices" not in d:
        return {"status": r.status_code, "error": str(d.get("error", d))[:120]}
    choice = d["choices"][0]
    content = (choice.get("logprobs") or {}).get("content") or []
    first = content[0] if content else None
    top: dict[str, float] = {}
    for t in (first or {}).get("top_logprobs", []):  # sum variants (" A", "A") per label
        key = t["token"].strip()
        top[key] = top.get(key, 0.0) + math.exp(t["logprob"])
    usage = d.get("usage") or {}
    return {
        "provider": d.get("provider"),
        "text": choice["message"]["content"],
        "n_positions": len(content),
        "completion_tokens": usage.get("completion_tokens"),
        "first_token": first and first["token"],
        "n_top": len(first["top_logprobs"]) if first else 0,
        "label_mass": round(sum(p for k, p in top.items() if k in LABELS), 4),
    }


def main(models: list[str]) -> None:
    headers = {"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]}
    with httpx.Client(headers=headers, timeout=60) as client:
        for model in models:
            eps = client.get(f"{BASE}/models/{model}/endpoints").json()["data"]["endpoints"]
            for ep in eps:
                tag = ep.get("tag") or ep["provider_name"]
                if "top_logprobs" not in (ep.get("supported_parameters") or []):
                    continue
                for mt in (1, 3):
                    row = {"model": model, "tag": tag, "max_tokens": mt}
                    row.update(probe(client, model, tag, mt))
                    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or MODELS)
