"""Phase 0 spike: Jev through OpenRouter's System One API with the official SDK.

Checks that responses validate against the schema snapshot (via the SDK's generated wire models),
records the returned model id, usage, and headers, compares against `jev_reference` values, checks
the published confidence approximation, and measures variance across repeated calls.

Usage:
    uv run --env-file .env python spikes/jev_openrouter.py [n_cases] [repeats]

Raw responses go to runs/spikes/jev.jsonl (gitignored; never commit Jev responses).
"""

import json
import os
import statistics
import sys
import time
from pathlib import Path

from typesafe_sdk import RetryPolicy, TypeSafeClient
from typesafe_sdk._schemas import models as wire

DOC_EXAMPLE = {
    "id": "api-schema-example",
    "request": {
        "state": "Help! My payouts have been failing for 3 days.",
        "questions": {
            "is_urgent": {
                "type": "noul",
                "instructions": "Does this convey urgency?",
                "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"},
            },
            "department": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {
                    "billing": "Payments, invoicing, refunds",
                    "technical": "Bugs, outages, integrations",
                    "sales": None,
                },
            },
            "frustration": {
                "type": "score",
                "instructions": "How frustrated is the customer?",
                "criteria": ["Calm", "Frustrated", "Very angry"],
            },
        },
    },
}


def confidence(probs: list[float]) -> float:
    n = len(probs)
    return min(1.0, max(0.0, (n * max(probs) - 1) / (n - 1)))


def main(n_cases: int, repeats: int) -> None:
    cases = [DOC_EXAMPLE] + [
        json.loads(line) for line in Path("evals/sanity-v0.jsonl").read_text().splitlines()
    ][:n_cases]
    client = TypeSafeClient(
        base_url="https://openrouter.ai/api",
        api_key=os.environ["OPENROUTER_API_KEY"],
        model="jev-1.13",
        timeout=60,
        retry=RetryPolicy(max_retries=1),
    )
    out = Path("runs/spikes/jev.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        for case in cases:
            req = case["request"]
            values: dict[str, list[float]] = {}
            for i in range(repeats):
                t0 = time.perf_counter()
                resp = client.system_one(req["state"], req["questions"])
                dt = time.perf_counter() - t0
                raw = resp.raw_http_response
                body = raw.json()
                wire.SystemOneResponse.model_validate(body)  # schema-snapshot check
                headers = {
                    k: v
                    for k, v in raw.headers.items()
                    if k.startswith("x-") or k in ("server-timing", "content-type")
                }
                f.write(
                    json.dumps(
                        {
                            "case": case["id"],
                            "i": i,
                            "seconds": dt,
                            "body": body,
                            "headers": headers,
                        }
                    )
                    + "\n"
                )
                for key, ans in body["answers"].items():
                    if ans["type"] == "noul":
                        values.setdefault(key, []).append(ans["noul"])
                    else:
                        probs = list(ans["probabilities"].values())
                        top = max(ans["probabilities"], key=ans["probabilities"].get)
                        values.setdefault(key, []).append(ans["probabilities"][top])
                        values.setdefault(key + ".conf_delta", []).append(
                            ans["confidence"] - confidence(probs)
                        )
                if i == 0:
                    print(
                        f"{case['id']}: model={body['model']} usage={body['usage']} "
                        f"{dt:.2f}s request_id={raw.headers.get('x-typesafe-request-id')} "
                        f"x-headers={sorted(headers)}"
                    )
                    print("   answers:", json.dumps(body["answers"])[:400])
            for key, vals in values.items():
                spread = max(vals) - min(vals)
                ref = (case.get("jev_reference") or {}).get(key)
                print(
                    f"   {key}: mean={statistics.mean(vals):.4f} spread={spread:.4f}"
                    + (f" ref={ref}" if ref is not None else "")
                )


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5, int(sys.argv[2]) if len(sys.argv) > 2 else 3)
