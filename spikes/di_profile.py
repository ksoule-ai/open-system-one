"""Profile Decision Index rows as our server would see them: calls, coverage, prompt tokens.

Renders every request through the profile's prompt config (exactly what the model would get),
applies our limits (options per choice, context length), deduplicates identical questions like
the server does, and estimates prompt tokens from characters with a ratio calibrated on our own
Granite traces (tokens ≈ TOKENS_PER_CHAR · chars + TOKENS_PER_CALL).

Usage:
    uv run python spikes/di_profile.py ROWS.jsonl[.gz] [ROWS2 ...] [--profile oso-granite-3b]

Prints a per-benchmark table and totals; writes runs/spikes/di_profile.json.
"""

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

from open_system_one.config import load_registry
from open_system_one.engine import LimitError, check_limits
from open_system_one.prompts.config import load_prompt
from open_system_one.prompts.render import render_question, render_state
from open_system_one.schema.models import SystemOneRequest

TOKENS_PER_CHAR = 0.2848  # fit on 49 Granite calls (runs/eval traces), 2026-09-26
TOKENS_PER_CALL = 8.2
CONTEXT_LIMIT = 131072  # the endpoint's max_model_len


def tokens(chars: int) -> float:
    return TOKENS_PER_CHAR * chars + TOKENS_PER_CALL


def read_rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows", nargs="+")
    ap.add_argument("--profile", default="oso-granite-3b")
    args = ap.parse_args()

    import os

    os.environ.setdefault("HF_ENDPOINT_URL", "http://unused/v1")
    registry = load_registry("configs/models.yaml")
    profile = registry.resolve(args.profile)
    prompt = load_prompt("configs/prompts", profile.prompt)

    stats = defaultdict(lambda: defaultdict(float))
    for path in args.rows:
        for row in read_rows(path):
            ev = row.get("_evaluation", {})
            bench = (
                ev.get("dataset")
                or ev.get("benchmark")
                or row.get("metadata", {}).get("dataset", "?")
            )
            s = stats[bench]
            s["requests"] += 1
            s["questions"] += len(row["questions"])
            try:
                req = SystemOneRequest.model_validate(
                    {"state": row["state"], "model": profile.name, "questions": row["questions"]}
                )
                check_limits(req, profile, prompt)
            except LimitError:
                s["unsupported_requests"] += 1
                s["unsupported_questions"] += len(row["questions"])
                continue
            except Exception:  # noqa: BLE001 - schema-invalid rows are counted, not fatal
                s["invalid_requests"] += 1
                continue

            seen = set()
            prefix_chars = len(render_state(req.state, prompt)) + len(prompt.system)
            total = cached = 0.0
            too_long = False
            for q in req.questions.values():
                canon = json.dumps(q.root.model_dump(mode="json", exclude_unset=True))
                if canon in seen:
                    continue
                seen.add(canon)
                rendered = render_question(req.state, q.root, prompt)
                t = tokens(sum(len(m["content"]) for m in rendered.messages))
                too_long |= t > CONTEXT_LIMIT
                total += t
                # With prefix caching, every call after the first reuses the system + state prefix.
                cached += (
                    t if len(seen) == 1 else max(t - tokens(prefix_chars) + TOKENS_PER_CALL, 1)
                )
            if too_long:
                s["context_too_long_requests"] += 1
                continue
            s["calls"] += len(seen)
            s["prompt_tokens"] += total
            s["prompt_tokens_after_prefix_cache"] += cached
            s["max_call_tokens"] = max(s["max_call_tokens"], total / max(len(seen), 1))

    keys = [
        "requests",
        "questions",
        "calls",
        "unsupported_requests",
        "context_too_long_requests",
        "invalid_requests",
        "prompt_tokens",
        "prompt_tokens_after_prefix_cache",
    ]
    totals = {k: sum(s[k] for s in stats.values()) for k in keys}
    print(
        f"{'benchmark':28} {'req':>7} {'calls':>8} {'unsup':>6} {'toolong':>7} {'Mtok':>7} {'Mtok(cache)':>11} {'tok/call':>8}"
    )
    for bench, s in sorted(stats.items(), key=lambda kv: -kv[1]["prompt_tokens"]):
        per_call = s["prompt_tokens"] / s["calls"] if s["calls"] else 0
        print(
            f"{bench[:28]:28} {s['requests']:7.0f} {s['calls']:8.0f} {s['unsupported_requests']:6.0f} "
            f"{s['context_too_long_requests']:7.0f} {s['prompt_tokens'] / 1e6:7.1f} "
            f"{s['prompt_tokens_after_prefix_cache'] / 1e6:11.1f} {per_call:8.0f}"
        )
    print("TOTAL", {k: round(v) for k, v in totals.items()})
    out = Path("runs/spikes/di_profile.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"totals": totals, "benchmarks": stats}, indent=1, default=float))


if __name__ == "__main__":
    main()
