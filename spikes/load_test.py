"""Measure one backend replica's throughput on realistic Decision Index traffic.

Samples requests from a Decision Index rows file, renders every question with the profile's
prompt config (exactly what a run would send), and replays the calls through our Mellea client
at rising concurrency. A request's questions are queued back to back, so prefix-cache reuse is
similar to a real run. Reports calls/s, prompt tokens/s (from the backend's usage), latency, and
errors per concurrency level.

Usage:
    uv run --env-file .env python spikes/load_test.py ROWS.jsonl --profile oso-granite-3b-wide \
        [--levels 8 32 64 128] [--seconds 60] [--sample 4000]
"""

import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

from open_system_one.backends import BackendOverloaded, make_client
from open_system_one.config import load_registry
from open_system_one.engine import LimitError, check_limits, request_top_logprobs
from open_system_one.prompts.config import load_prompt
from open_system_one.prompts.render import render_question
from open_system_one.schema.models import SystemOneRequest


def sample_calls(rows_path, n_requests, profile, prompt, seed=20260926):
    # split on \n only: splitlines() also breaks on U+2028 etc. inside JSON strings
    lines = [line for line in Path(rows_path).read_text().split("\n") if line.strip()]
    rng = random.Random(seed)
    calls = []
    for line in rng.sample(lines, n_requests):
        row = json.loads(line)
        req = SystemOneRequest.model_validate(
            {"state": row["state"], "model": profile.name, "questions": row["questions"]}
        )
        try:
            check_limits(req, profile, prompt)
        except LimitError:
            continue
        seen = set()
        for q in req.questions.values():
            canon = json.dumps(q.root.model_dump(mode="json", exclude_unset=True))
            if canon in seen:
                continue
            seen.add(canon)
            r = render_question(req.state, q.root, prompt)
            calls.append((r.messages, r.prefill, request_top_logprobs(profile, len(r.labels))))
    return calls


async def run_level(client, calls, concurrency, seconds):
    queue = list(calls)
    done, tokens, latencies, overloads, errors = 0, 0, [], 0, 0
    deadline = time.monotonic() + seconds
    lock = asyncio.Lock()

    async def worker():
        nonlocal done, tokens, overloads, errors
        while time.monotonic() < deadline:
            async with lock:
                if not queue:
                    queue.extend(calls)
                messages, prefill, top_k = queue.pop(0)
            t = time.perf_counter()
            try:
                result = await client.complete(messages, prefill=prefill, top_logprobs=top_k)
            except BackendOverloaded:
                overloads += 1
                await asyncio.sleep(1)
                continue
            except Exception:  # noqa: BLE001 - counted and reported
                errors += 1
                continue
            latencies.append(time.perf_counter() - t)
            done += 1
            tokens += (result.raw.get("usage") or {}).get("prompt_tokens", 0)

    t0 = time.monotonic()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    elapsed = time.monotonic() - t0
    lat = sorted(latencies)
    return {
        "concurrency": concurrency,
        "calls_per_s": done / elapsed,
        "prompt_tokens_per_s": tokens / elapsed,
        "tokens_per_call": tokens / done if done else 0,
        "p50_ms": statistics.median(lat) * 1000 if lat else None,
        "p95_ms": lat[int(0.95 * (len(lat) - 1))] * 1000 if lat else None,
        "overloads": overloads,
        "errors": errors,
    }


async def main(args):
    registry = load_registry("configs/models.yaml")
    profile = registry.resolve(args.profile)
    prompt = load_prompt("configs/prompts", profile.prompt)
    # The load test sets its own concurrency; lift the client's cap above the highest level.
    profile = profile.model_copy(update={"concurrency": max(args.levels)})
    calls = sample_calls(args.rows, args.sample, profile, prompt)
    print(
        f"{len(calls)} calls from {args.sample} sampled requests; profile {profile.name}",
        flush=True,
    )
    client = make_client(profile)
    await run_level(client, calls[:50], 4, 10)  # warm-up: builds the backend, wakes the endpoint
    results = []
    for level in args.levels:
        r = await run_level(client, calls, level, args.seconds)
        results.append(r)
        print(
            f"c={r['concurrency']:4}  {r['calls_per_s']:6.1f} calls/s  {r['prompt_tokens_per_s']:8.0f} prompt tok/s  "
            f"{r['tokens_per_call']:5.0f} tok/call  p50 {r['p50_ms']:6.0f} ms  p95 {r['p95_ms']:6.0f} ms  "
            f"overloads {r['overloads']}  errors {r['errors']}",
            flush=True,
        )
    out = Path("runs/spikes/load_test.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"profile": profile.name, "results": results}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("--profile", default="oso-granite-3b-wide")
    ap.add_argument("--levels", type=int, nargs="+", default=[8, 32, 64, 128])
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--sample", type=int, default=4000)
    asyncio.run(main(ap.parse_args()))
