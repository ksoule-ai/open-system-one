"""How many model calls per second can one server process push? (All local; no real backend.)

Starts a fake OpenAI-compatible backend that answers after a fixed delay with canned logprobs,
runs our server (one uvicorn process, `open-system-one serve`) against it, and drives it with
concurrent SDK-shaped requests. Reports requests/s, model calls/s, and latency.

Usage:
    uv run python spikes/server_throughput.py [--concurrency 64] [--questions 1] [--seconds 20]
"""

import argparse
import asyncio
import math
import os
import socket
import statistics
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import httpx

FAKE_BACKEND = textwrap.dedent("""
    import asyncio, math, sys
    from fastapi import FastAPI
    import uvicorn
    app = FastAPI()
    TOP = [{"token": t, "logprob": math.log(p)} for t, p in
           [("A", .5), ("B", .2), ("Yes", .1), ("No", .05), ("0", .05), ("1", .05), ("C", .05)]]
    @app.post("/v1/chat/completions")
    async def chat(body: dict):
        await asyncio.sleep(float(sys.argv[2]))
        return {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                "choices": [{"index": 0, "finish_reason": "length",
                             "message": {"role": "assistant", "content": "A"},
                             "logprobs": {"content": [{"token": "A", "logprob": math.log(.5),
                                                       "top_logprobs": TOP}]}}],
                "usage": {"prompt_tokens": 300, "completion_tokens": 1, "total_tokens": 301}}
    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": "fake", "object": "model"}]}
    uvicorn.run(app, port=int(sys.argv[1]), log_level="warning")
""")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for(url: str, headers=None, timeout=30):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            if httpx.get(url, headers=headers, timeout=1).status_code < 500:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    raise RuntimeError(f"{url} did not come up")


async def drive(url: str, key: str, concurrency: int, n_questions: int, seconds: float):
    questions = {
        f"q{i}": {
            "type": "choice",
            "instructions": f"Pick {i}",
            "criteria": {"a": None, "b": None, "c": None},
        }
        for i in range(n_questions)
    }
    latencies, errors = [], 0
    deadline = time.monotonic() + seconds

    async def worker(client):
        nonlocal errors
        i = 0
        while time.monotonic() < deadline:
            i += 1
            t = time.perf_counter()
            r = await client.post(
                "/v1/systemone",
                json={"model": "fake", "state": f"state {i}", "questions": questions},
            )
            if r.status_code == 200:
                latencies.append(time.perf_counter() - t)
            else:
                errors += 1

    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(
        base_url=url, headers={"Authorization": f"Bearer {key}"}, timeout=60, limits=limits
    ) as client:
        t0 = time.monotonic()
        await asyncio.gather(*(worker(client) for _ in range(concurrency)))
        elapsed = time.monotonic() - t0
    return latencies, errors, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--questions", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--backend-delay", type=float, default=0.05, help="fake backend latency (s)")
    ap.add_argument("--server-concurrency", type=int, default=256)
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp())
    fake_port, server_port, key = free_port(), free_port(), "bench-key"
    (tmp / "fake.py").write_text(FAKE_BACKEND)
    (tmp / "models.yaml").write_text(
        textwrap.dedent(f"""
        profiles:
          fake:
            backend: hf_endpoint
            base_url: http://127.0.0.1:{fake_port}/v1
            api_key_env: FAKE_KEY
            model_id: fake
            release_date: "2026-09-26"
            prompt: default@1
            strategy: fanout
            concurrency: {args.server_concurrency}
            max_inflight_requests: 100000
    """)
    )
    env = {**os.environ, "OSO_API_KEY": key, "FAKE_KEY": "x"}
    procs = [
        subprocess.Popen(
            [sys.executable, str(tmp / "fake.py"), str(fake_port), str(args.backend_delay)]
        ),
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "open_system_one.cli",
                "serve",
                "--port",
                str(server_port),
                "--models",
                str(tmp / "models.yaml"),
                "--no-trace",
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ),
    ]
    try:
        wait_for(f"http://127.0.0.1:{fake_port}/v1/models")
        url = f"http://127.0.0.1:{server_port}"
        wait_for(f"{url}/v1/models", headers={"Authorization": f"Bearer {key}"})
        asyncio.run(drive(url, key, 4, args.questions, 2))  # warm up (lazy backend construction)
        lat, errors, elapsed = asyncio.run(
            drive(url, key, args.concurrency, args.questions, args.seconds)
        )
        rps = len(lat) / elapsed
        print(
            f"concurrency={args.concurrency} questions/request={args.questions} backend_delay={args.backend_delay}s: "
            f"{rps:.0f} req/s, {rps * args.questions:.0f} model calls/s, errors={errors}, "
            f"latency p50={statistics.median(lat) * 1000:.0f} ms p95={sorted(lat)[math.floor(0.95 * len(lat))] * 1000:.0f} ms"
        )
    finally:
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
