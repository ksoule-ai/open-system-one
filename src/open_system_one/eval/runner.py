"""Run eval cases against one of our profiles and the Jev baseline; write results and a report.

Our server is started in-process on a free local port with its own trace directory, so each
answer can be joined with its trace (label mass, missing labels, timings). Requests go through the
official SDK, so the eval also exercises API compatibility.
"""

import json
import secrets
import socket
import threading
import time
from pathlib import Path
from typing import Any

import uvicorn
from typesafe_sdk import RetryPolicy, TypeSafeAPIError, TypeSafeClient

from open_system_one.config import load_registry
from open_system_one.eval import report, scoring
from open_system_one.eval.jev import JevBaseline
from open_system_one.prompts.config import load_prompt
from open_system_one.server import create_app


def load_cases(path: str | Path, split: str) -> list[dict[str, Any]]:
    cases = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return [c for c in cases if split == "all" or c["split"] == split]


class _Server:
    def __init__(self, app):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning")
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self) -> str:
        self.thread.start()
        deadline = time.monotonic() + 15
        while not self.server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("eval server did not start")
            time.sleep(0.02)
        return f"http://127.0.0.1:{self.port}"

    def __exit__(self, *exc):
        time.sleep(0.2)  # let background trace writes finish
        self.server.should_exit = True
        self.thread.join(10)


def _wait_until_ready(client: TypeSafeClient, target: str, max_wait: float) -> None:
    """Scaled-to-zero endpoints answer 503 (→ our 529) for minutes while they start."""
    t0 = time.monotonic()
    q = {"ready": {"type": "noul", "instructions": "Is this a test?"}}
    while True:
        try:
            client.system_one("ping", q, model=target)
            return
        except TypeSafeAPIError as e:
            if e.status != 529 or time.monotonic() - t0 > max_wait:
                raise
            print(
                f"  backend not ready ({e.status}); waiting… {time.monotonic() - t0:.0f}s",
                flush=True,
            )
            time.sleep(15)


def run_eval(
    cases_path: str,
    target: str,
    split: str = "tune",
    baseline: str | None = "jev-1.13",
    models_path: str = "configs/models.yaml",
    prompts_dir: str = "configs/prompts",
    out_root: str = "runs/eval",
    refresh_jev: bool = False,
    max_wait: float = 600,
) -> Path:
    registry = load_registry(models_path)
    profile = registry.resolve(target)
    if profile is None:
        raise SystemExit(f"unknown target {target!r}")
    prompt = load_prompt(prompts_dir, profile.prompt)
    cases = load_cases(cases_path, split)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(out_root) / f"{stamp}-{profile.name}-{prompt.content_hash}"
    run_dir.mkdir(parents=True)
    trace_dir = run_dir / "traces"

    key = secrets.token_hex(16)
    app = create_app(registry, prompts_dir, [key], trace_dir=trace_dir)
    jev = JevBaseline(baseline, refresh=refresh_jev) if baseline else None
    raw: list[dict[str, Any]] = []

    print(
        f"eval: {len(cases)} {split} cases, target {profile.name} ({prompt.ref} {prompt.content_hash})"
        + (f", baseline {baseline}" if baseline else ""),
        flush=True,
    )
    with _Server(app) as url:
        client = TypeSafeClient(
            base_url=url, api_key=key, timeout=60, retry=RetryPolicy(max_retries=2)
        )
        _wait_until_ready(client, profile.name, max_wait)
        for case in cases:
            req = case["request"]
            entry: dict[str, Any] = {"case": case}
            t0 = time.perf_counter()
            try:
                resp = client.system_one(req["state"], req["questions"], model=profile.name)
                entry["ours"] = resp.raw_http_response.json()
                entry["request_id"] = resp.request_id
            except TypeSafeAPIError as e:
                entry["ours_error"] = f"{e.status} {e}"
                entry["request_id"] = e.request_id
            entry["ours_wall_ms"] = (time.perf_counter() - t0) * 1000
            if jev:
                try:
                    entry["jev"] = jev.answer(req["state"], req["questions"])
                except Exception as e:  # noqa: BLE001 - a Jev failure is recorded, never fatal
                    entry["jev_error"] = f"{type(e).__name__}: {e}"
            raw.append(entry)
            print(
                f"  {case['id']:40} {'ok' if 'ours' in entry else entry['ours_error'][:60]}",
                flush=True,
            )

    traces = {}
    for f in trace_dir.glob("*.jsonl"):
        for line in f.read_text().splitlines():
            t = json.loads(line)
            traces[t["request_id"]] = t

    rows = [row for entry in raw for row in _rows(entry, traces.get(entry.get("request_id")))]
    with (run_dir / "results.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta = {
        "target": profile.name,
        "model_id": profile.model_id,
        "backend": profile.backend,
        "strategy": profile.strategy,
        "prompt": prompt.ref,
        "prompt_hash": prompt.content_hash,
        "baseline": baseline,
        "jev_served_models": sorted({e["jev"]["served_model"] for e in raw if "jev" in e}),
        "jev_cache": {"hits": jev.hits, "misses": jev.misses} if jev else None,
        "cases": str(cases_path),
        "split": split,
        "n_cases": len(cases),
        "time": stamp,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (run_dir / "report.md").write_text(report.render(meta, rows))
    return run_dir


def _rows(entry: dict[str, Any], trace: dict[str, Any] | None) -> list[dict[str, Any]]:
    case = entry["case"]
    calls = {k: c for c in (trace or {}).get("calls", []) for k in c.get("keys", [])}
    ours_answers = (entry.get("ours") or {}).get("answers", {})
    jev_answers = ((entry.get("jev") or {}).get("body") or {}).get("answers", {})
    rows = []
    for key, question in case["request"]["questions"].items():
        expect = case["expect"].get(key)
        row: dict[str, Any] = {
            "case": case["id"],
            "category": case["category"],
            "split": case["split"],
            "question": key,
            "type": question["type"],
            "expect": expect,
            "ours": ours_answers.get(key),
            "ours_error": entry.get("ours_error"),
            "jev": jev_answers.get(key),
            "jev_error": entry.get("jev_error"),
            "jev_reference": (case.get("jev_reference") or {}).get(key),
            "ours_wall_ms": entry["ours_wall_ms"],
            "ours_server_ms": (trace or {}).get("timings_ms", {}).get("total"),
            "jev_seconds": None
            if (entry.get("jev") or {}).get("cached")
            else (entry.get("jev") or {}).get("seconds"),
        }
        reading = (calls.get(key) or {}).get("reading") or {}
        row["label_mass"] = reading.get("label_mass")
        row["missing_labels"] = reading.get("missing")
        if row["ours"] and expect:
            row["ours_score"] = scoring.against_expect(row["ours"], expect)
        if row["jev"] and expect:
            row["jev_score"] = scoring.against_expect(row["jev"], expect)
        if row["ours"] and row["jev"]:
            row["vs_jev"] = scoring.against_jev(row["ours"], row["jev"])
        if row["jev"]:
            row["jev_confidence_delta"] = scoring.jev_confidence_check(row["jev"])
        rows.append(row)
    return rows
