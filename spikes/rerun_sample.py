"""Re-send one dev-set request to several engines and keep every log in one folder.

For each Granite profile the request goes to a local `open-system-one serve` (traces on); Jev goes
through the LiteLLM gateway's /v1/systemone pass-through. Writes to runs/spikes/rerun-<key>/:
  request.json                  the request exactly as the Decision Index kit sends it
  <engine>.response.json        the /v1/systemone response body (+ status, latency, headers)
  traces/                       our server's JSONL traces: rendered messages, raw logprobs
  index.md                      one table: answers, timings, and gateway call ids to search the
                                LiteLLM UI (Logs) with

    set -a; . ./.env; . ../kate-litellm/.env; set +a
    uv run python spikes/rerun_sample.py --key 1816 \
        --profiles oso-granite-3b-wide-litellm oso-granite-micro-cf-v3-litellm --jev
"""

import argparse
import gzip
import json
import os
import secrets
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROWS = Path("evals/decision-index-dev/dev-rows.jsonl.gz")
GATEWAY = "http://localhost:4000"


def find_row(key: str) -> dict:
    with gzip.open(ROWS, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("_dev", {}).get("key") == key:
                return row
    raise SystemExit(f"no dev row with key {key}")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post(url: str, key: str, body: dict) -> dict:
    t0 = time.perf_counter()
    r = httpx.post(url, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=300)
    out = {
        "status": r.status_code,
        "latency_ms": round((time.perf_counter() - t0) * 1000),
        "headers": {k: v for k, v in r.headers.items() if k.startswith(("x-", "server-timing"))},
    }
    try:
        out["body"] = r.json()
    except ValueError:
        out["body"] = r.text
    return out


def run_adapter(key: str, out: Path) -> dict:
    """The hallucination adapter on one RAGTruth sample (inputs as in ragtruth_hallucination.py).

    The call carries a LiteLLM tag (`rerun-<key>`) and its gateway log id is looked up afterwards,
    since the adapter wrapper returns only the parsed records.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    import ragtruth_hallucination as rh

    sample = next((s for s in rh.samples() if s["id"] == key), None)
    if sample is None:
        raise SystemExit(f"{key} is not a RAGTruth dev sample")
    tag = f"rerun-{key}"
    started = time.time()
    result = rh.judge(
        rh.make_backend(), sample, model_options={"extra_body": {"metadata": {"tags": [tag]}}}
    )
    (out / "adapter.inputs.json").write_text(
        json.dumps(
            {k: sample[k] for k in ("question", "document", "response")},
            indent=2,
            ensure_ascii=False,
        )
    )
    logs = httpx.get(
        GATEWAY + "/spend/logs/ui",  # paginated listing; /spend/logs chokes on large ranges
        params={
            "start_date": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(started - 60)),
            "end_date": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time() + 60)),
            "page": 1,
            "page_size": 100,
        },
        headers={"Authorization": f"Bearer {os.environ['LITELLM_MASTER_KEY']}"},
        timeout=60,
    ).json()
    logs = logs.get("data", []) if isinstance(logs, dict) else logs
    ids = [r["request_id"] for r in logs if tag in json.dumps(r.get("request_tags") or [])]
    return {
        "status": "error" if result["error"] else "ok",
        "latency_ms": round(result["seconds"] * 1000),
        "headers": {},
        "gateway_ids": ids,
        "tag": tag,
        "body": {
            "hallucinated": result["pred"],
            "flagged": [
                r["response_text"]
                for r in result["records"]
                if r["faithfulness"] in ("unfaithful", "partial")
            ],
            "records": result["records"],
            "error": result["error"],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True, help="the dev sample's source key (provenance `key`)")
    ap.add_argument("--profiles", nargs="*", default=[])
    ap.add_argument("--jev", action="store_true", help="also send to jev-1.13 via the gateway")
    ap.add_argument(
        "--adapter",
        action="store_true",
        help="also run Granite Switch's hallucination adapter (RAGTruth samples only)",
    )
    args = ap.parse_args()

    row = find_row(args.key)
    request = {"state": row["state"], "questions": row["questions"]}
    out = Path(f"runs/spikes/rerun-{args.key}")
    (out / "traces").mkdir(parents=True, exist_ok=True)
    (out / "request.json").write_text(json.dumps(request, indent=2, ensure_ascii=False))
    meta = {
        "run_id": row["_evaluation"]["run_id"],
        "dataset": row["_evaluation"]["dataset"],
        "expected": row.get("expected"),
    }
    results = {}

    if args.profiles:
        port, key = free_port(), secrets.token_hex(16)
        server = subprocess.Popen(
            [
                "uv",
                "run",
                "open-system-one",
                "serve",
                "--port",
                str(port),
                "--trace-dir",
                str(out / "traces"),
            ],
            env={**os.environ, "OSO_API_KEY": key},
            stdout=(out / "server.log").open("w"),
            stderr=subprocess.STDOUT,
        )
        try:
            base = f"http://127.0.0.1:{port}"
            for _ in range(120):
                try:
                    if (
                        httpx.get(
                            base + "/v1/models",
                            headers={"Authorization": f"Bearer {key}"},
                            timeout=2,
                        ).status_code
                        == 200
                    ):
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.5)
            for profile in args.profiles:
                results[profile] = post(base + "/v1/systemone", key, {"model": profile, **request})
        finally:
            server.terminate()

    if args.jev:
        results["jev-1.13"] = post(
            GATEWAY + "/v1/systemone",
            os.environ["LITELLM_MASTER_KEY"],
            {"model": "jev-1.13", **request},
        )

    if args.adapter:
        results["granite-switch-hallucination-adapter"] = run_adapter(args.key, out)

    for engine, res in results.items():
        (out / f"{engine}.response.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))

    # Gateway call ids per profile, from our traces (the chat-completion id the gateway returns).
    calls = {}
    for path in sorted((out / "traces").glob("**/*.jsonl")):
        for line in path.open():
            t = json.loads(line)
            calls.setdefault(t.get("profile"), []).extend(
                {
                    "gateway_id": (c.get("response") or {}).get("id"),
                    "label_mass": (c.get("reading") or {}).get("label_mass"),
                    "probs": (c.get("reading") or {}).get("probs"),
                }
                for c in t.get("calls", [])
            )

    lines = [
        f"# Rerun of dev sample {args.key}",
        "",
        f"- run id: `{meta['run_id']}` ({meta['dataset']})",
        f"- expected: `{json.dumps(meta['expected'])}`",
        "",
        "| Engine | HTTP | ms | Answers | Gateway call ids (search LiteLLM Logs) |",
        "| --- | --: | --: | --- | --- |",
    ]
    for engine, res in results.items():
        body = res["body"]
        answers = json.dumps(body.get("answers")) if isinstance(body, dict) else str(body)[:200]
        ids = [c["gateway_id"] for c in calls.get(engine, [])]
        if engine == "jev-1.13":
            # Pass-through calls are logged under the gateway's own call id, not OpenRouter's.
            ids = [res["headers"].get("x-litellm-call-id")]
        if "gateway_ids" in res:  # the adapter: verdict + flagged sentences, tagged in LiteLLM
            answers = json.dumps({"hallucinated": body["hallucinated"], "flagged": body["flagged"]})
            ids = res["gateway_ids"] + [f"tag {res['tag']}"]
        lines.append(
            f"| {engine} | {res['status']} | {res['latency_ms']} | `{answers}` | "
            f"{', '.join(f'`{i}`' for i in ids if i)} |"
        )
    (out / "index.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwritten to {out}/")


if __name__ == "__main__":
    sys.exit(main())
