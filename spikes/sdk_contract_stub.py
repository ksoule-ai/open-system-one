"""Phase 0 spike: point the official typesafe-sdk at a stub server and record what it does.

Runs a throwaway FastAPI stub in-process on a local port and exercises the SDK against it:
paths, sent headers, request-id handling, error classes, and which statuses are retried.

Usage:
    uv run python spikes/sdk_contract_stub.py
"""

import threading
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typesafe_sdk import RetryPolicy, TypeSafeClient, TypeSafeError

PORT = 8765
app = FastAPI()
seen: list[dict] = []

ANSWERS = {
    "n": {"type": "noul", "noul": 0.9},
    "c": {"type": "choice", "choice": "a", "probabilities": {"a": 1, "b": 0}, "confidence": 1},
    "s": {
        "type": "score",
        "score": 1.0,
        "legend": {"0": "low", "1": "high"},
        "probabilities": {"0": 0.0, "1": 1.0},
        "confidence": 1.0,
    },
}


@app.post("/v1/systemone")
async def systemone(request: Request):
    body = await request.json()
    seen.append({"path": request.url.path, "headers": dict(request.headers), "body": body})
    mode = body.get("model")
    rid = {"x-typesafe-request-id": f"req_{len(seen)}"}
    if mode == "unauthorized":
        return JSONResponse({"detail": "Invalid API key"}, 401, headers=rid)
    if mode == "invalid":
        detail = [{"loc": ["body", "state"], "msg": "Field required", "type": "missing"}]
        return JSONResponse({"detail": detail}, 422, headers=rid)
    if mode in ("busy", "overloaded", "broken"):
        status = {"busy": 429, "overloaded": 529, "broken": 500}[mode]
        return JSONResponse({"detail": mode}, status, headers={**rid, "retry-after": "0"})
    if mode == "extra_field":
        return JSONResponse(
            {
                "model": mode,
                "answers": ANSWERS,
                "usage": {"input_tokens": 1, "output_tokens": 1, "x": 1},
            },
            headers=rid,
        )
    return JSONResponse(
        {"model": mode, "answers": ANSWERS, "usage": {"input_tokens": 1, "output_tokens": 1}},
        headers=rid,
    )


@app.get("/v1/models")
async def models(request: Request):
    seen.append({"path": request.url.path, "headers": dict(request.headers)})
    return JSONResponse(
        {"models": [{"name": "stub", "description": "stub", "release_date": "2026-09-26"}]},
        headers={"x-typesafe-request-id": "req_models"},
    )


def main() -> None:
    server = uvicorn.Server(uvicorn.Config(app, port=PORT, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise SystemExit(f"stub server failed to start on port {PORT}")
        time.sleep(0.05)

    client = TypeSafeClient(
        base_url=f"http://127.0.0.1:{PORT}", api_key="k", retry=RetryPolicy(backoff_initial=0)
    )
    questions = {
        "n": {"type": "noul", "instructions": "?"},
        "c": {"type": "choice", "criteria": {"a": None, "b": None}},
        "s": {"type": "score", "criteria": ["low", "high"]},
    }
    for mode in ["ok", "extra_field", "unauthorized", "invalid", "busy", "overloaded", "broken"]:
        before = len(seen)
        try:
            r = client.system_one("state", questions, model=mode)
            result = f"OK request_id={r.request_id} answers={sorted(r.answers)}"
            result += f" score.legend keys={list(r.answers['s'].legend)}"
        except TypeSafeError as e:
            result = f"{type(e).__name__}: {e}"
        print(f"{mode:12} attempts={len(seen) - before}  {result}")
    r = client.models.list()
    print("models.list:", r)
    print("sent headers:", {k: v for k, v in seen[0]["headers"].items() if k != "authorization"})
    print("auth header:", seen[0]["headers"]["authorization"][:9] + "…")
    server.should_exit = True


if __name__ == "__main__":
    main()
