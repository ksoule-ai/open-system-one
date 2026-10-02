"""Run the Decision Index suite against our server in parallel shards, then merge and score.

The kit's runner sends one request at a time, so a full run is sharded: the suite's rows are
dealt round-robin into N files (same benchmark mix in each), N kit `run --engine http` processes
work through them against a few of our server processes, and the shard results are merged into
one results.jsonl scored once by the kit. Nothing about a request changes: every row is sent
unmodified, exactly once, as the kit's http engine sends it.

Run with the kit's Python (it imports decision_index):

    K=external/decision-index/.venv/bin/python
    $K scripts/decision_index_run.py split  --run R --shards 48
    $K scripts/decision_index_run.py launch --run R --profile oso-granite-3b-wide --servers 3
    $K scripts/decision_index_run.py score  --run R

Other rows and targets (e.g. the dev set against Jev on OpenRouter):

    $K scripts/decision_index_run.py split  --run R --rows evals/decision-index-dev/dev-rows.jsonl.gz
    $K scripts/decision_index_run.py launch --run R --base-url https://openrouter.ai/api \
        --profile jev-1.13 --token-env OPENROUTER_API_KEY
    $K scripts/decision_index_run.py score  --run R --rows evals/decision-index-dev/dev-rows.jsonl.gz

`--rows` takes every row of that file (no edition filter); `score --rows` scores them with the
kit's 0.2 scorer in place of the suite. `--base-url` sends to that server instead of starting ours.

`launch` again resumes: finished requests are skipped and errored ones retried (kit behavior).
`launch --engine random` runs the kit's random baseline instead (no servers; a dry run).
Our servers need the model backend's env (HF_ENDPOINT_URL, HF_TOKEN): source .env first.
"""

import argparse
import json
import os
import secrets
import socket
import subprocess
import sys
import time
from pathlib import Path

KIT = Path("external/decision-index")
KIT_PYTHON = KIT / ".venv/bin/python"


def split(args):
    from decision_index.suite.io import Suite

    if args.rows:
        paths, keep = [Path(args.rows)], lambda e: True
    else:
        suite = Suite(Path(args.suite_dir), "0.2")
        paths, keep = suite.row_paths, suite.in_edition
    shard_dir = Path(args.run) / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    files = [
        (shard_dir / f"shard-{i:03d}.jsonl").open("w", encoding="utf-8", newline="\n")
        for i in range(args.shards)
    ]
    n = seen = 0
    for path in paths:
        from decision_index.suite.io import read_jsonl

        for row in read_jsonl(path):
            if keep(row["_evaluation"]):
                seen += 1
                if (seen - 1) % args.every:
                    continue
                files[n % args.shards].write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
    for f in files:
        f.close()
    meta = {
        "rows": n,
        "every": args.every,
        "shards": args.shards,
        "suite_dir": str(args.suite_dir),
        "rows_file": args.rows,
        "edition": "0.2",
    }
    (Path(args.run) / "split.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_ready(url: str, key: str, timeout: float = 60) -> None:
    import httpx

    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            if (
                httpx.get(
                    url + "/v1/models", headers={"Authorization": f"Bearer {key}"}, timeout=2
                ).status_code
                == 200
            ):
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"server at {url} did not start")


def _progress(run: Path, shards: list[Path]) -> tuple[int, dict]:
    import collections

    counts = collections.Counter()
    for s in shards:
        results = run / s.stem / "results.jsonl"
        if results.exists():
            with results.open(encoding="utf-8") as f:
                for line in f:
                    if line.endswith("\n"):
                        counts[json.loads(line)["status"]] += 1
    return sum(counts.values()), dict(counts)


def launch(args):
    run = Path(args.run)
    shards = sorted((run / "shards").glob("shard-*.jsonl"))
    total = json.loads((run / "split.json").read_text())["rows"]
    servers, procs = [], []
    key = secrets.token_hex(16)
    if args.base_url:
        key = os.environ.get(args.token_env, "")
        if not key:
            raise SystemExit(f"{args.token_env} is not set")
    env = {**os.environ, "DECISION_INDEX_API_KEY": key, "OSO_API_KEY": key}
    try:
        urls = [args.base_url] if args.base_url else []
        if args.engine == "http" and not args.base_url:
            for i in range(args.servers):
                port = _free_port()
                log = (run / f"server-{i}.log").open("a")
                servers.append(
                    subprocess.Popen(
                        [
                            "uv",
                            "run",
                            "--project",
                            ".",
                            "open-system-one",
                            "serve",
                            "--port",
                            str(port),
                            "--trace-dir",
                            str(run / "traces" / f"server-{i}"),
                        ],
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                )
                urls.append(f"http://127.0.0.1:{port}")
            for url in urls:
                _wait_ready(url, key)
            print(f"{len(urls)} servers ready: {urls}", flush=True)
        # Every shard runs the kit's own runner in a thread of this process (one blocking
        # request at a time per shard, exactly as `decision_index run` does), which keeps memory
        # flat: ~100 separate interpreters would not fit next to the servers in 8 GB.
        from decision_index.runner import run as kit_run

        os.environ["DECISION_INDEX_API_KEY"] = key
        failures: dict[str, str] = {}

        def shard_worker(i: int, shard: Path) -> None:
            options = {}
            if args.engine == "http":
                options = {"base_url": urls[i % len(urls)], "model": args.profile}
            with (run / f"{shard.stem}.log").open("a", encoding="utf-8") as log:
                try:
                    kit_run(
                        args.engine,
                        options,
                        shard.resolve(),
                        (run / shard.stem).resolve(),
                        compact=True,
                        log=lambda line: print(line, file=log, flush=True),
                    )
                except Exception as exc:  # noqa: BLE001 - reported per shard; rerun resumes it
                    failures[shard.stem] = f"{type(exc).__name__}: {exc}"

        import threading

        for i, shard in enumerate(shards):
            t = threading.Thread(target=shard_worker, args=(i, shard), name=shard.stem, daemon=True)
            t.start()
            procs.append(t)
        print(f"{len(procs)} shard runners started (threads)", flush=True)
        t0 = time.monotonic()
        while any(t.is_alive() for t in procs):
            time.sleep(args.report_every)
            done, counts = _progress(run, shards)
            rate = done / max(time.monotonic() - t0, 1)
            eta = (total - done) / rate / 60 if rate else float("nan")
            alive = sum(t.is_alive() for t in procs)
            print(
                f"[{time.strftime('%H:%M:%S')}] {done}/{total} ({100 * done / total:.1f}%) {counts} "
                f"{rate:.1f} req/s, ETA {eta:.0f} min, {alive} runners alive",
                flush=True,
            )
        done, counts = _progress(run, shards)
        failed = failures
        print(
            json.dumps(
                {
                    "event": "finished",
                    "done": done,
                    "total": total,
                    "counts": counts,
                    "failed_runners": failed,
                    "minutes": round((time.monotonic() - t0) / 60, 1),
                }
            ),
            flush=True,
        )
    finally:
        for p in servers:
            if p.poll() is None:
                p.terminate()


def score(args):
    run = Path(args.run)
    merged_dir = run / "merged"
    merged_dir.mkdir(exist_ok=True)
    merged = merged_dir / "results.jsonl"
    seen, n = set(), 0
    with merged.open("w", encoding="utf-8", newline="\n") as out:
        for results in sorted(run.glob("shard-*/results.jsonl")):
            latest = {}
            with results.open(encoding="utf-8") as f:
                for line in f:
                    if line.endswith("\n"):
                        r = json.loads(line)
                        latest[r["run_id"]] = line  # a resumed run appends retries; keep the last
            for rid, line in latest.items():
                if rid not in seen:
                    seen.add(rid)
                    out.write(line)
                    n += 1
    print(json.dumps({"merged_rows": n}), flush=True)
    if args.rows:
        return score_rows(Path(args.rows), merged, args.name or run.name, merged_dir)
    subprocess.run(
        [
            str(KIT_PYTHON.resolve()),
            "-m",
            "decision_index",
            "score",
            "--results",
            str(merged.resolve()),
            "--suite-dir",
            str(Path(args.suite_dir).resolve()),
            "--engine",
            args.name or run.name,
            "--out",
            str(merged_dir.resolve()),
        ],
        cwd=KIT,
        check=True,
    )


def score_rows(rows: Path, merged: Path, name: str, out: Path):
    """Per-benchmark scores from the kit's 0.2 scorers over `rows` (no composite index: the
    index needs every 0.2 benchmark and the frozen suite). Same steps as the kit's
    `score_run_v02` up to its benchmark summary."""
    import collections

    from decision_index.pipeline import rnd
    from decision_index.scoring import added, index02
    from decision_index.scoring.report import benchmark_summary, load_results
    from decision_index.suite.io import atomic_json, read_jsonl

    results = load_results(merged)
    added_ids = {int(n) for n in index02.spec()["added"]}
    base, extra = [], collections.defaultdict(list)
    for r in read_jsonl(rows):
        n = r["_evaluation"]["catalog_id"]
        (extra[n] if n in added_ids else base).append(r)
    summary = benchmark_summary(None, results, name, rows=base)
    for n, rs in sorted(extra.items()):
        if not any(results.get(r["_evaluation"]["run_id"], {}).get("status") == "ok" for r in rs):
            # the kit's added report crashes with no answered row (mean of an empty generator)
            st = collections.Counter(results.get(r["_evaluation"]["run_id"], {}).get("status", "pending") for r in rs)
            summary["benchmarks"].append({"catalog_id": n, "dataset": rs[0]["_evaluation"]["dataset"], "requests": len(rs), "answered": 0, "unsupported": st["unsupported"], "errors": st["error"], "abstained": st["abstained"], "pending": st["pending"], "metric": "accuracy", "score": None, "median_ms": None})
            continue
        rep = added.report(n, rs, results)
        entry = {k: rep[k] for k in ("catalog_id", "dataset", "requests", "answered", "unsupported", "errors", "abstained", "pending", "metric", "score", "median_ms")}
        entry.update(scored_requests=rep["answered"], detail={"field_accuracy": rep["field_accuracy"], "scored_fields": rep["scored_fields"], "chance_on_rows": rep["chance"]})
        summary["benchmarks"].append(entry)
    summary["edition"] = "0.2"
    summary["rows_file"] = str(rows)
    atomic_json(out / "benchmark-summary.json", summary)
    table = [
        {k: (rnd(v) if isinstance(v, float) else v) for k, v in b.items() if k in ("catalog_id", "dataset", "requests", "answered", "unsupported", "errors", "metric", "score")}
        for b in sorted(summary["benchmarks"], key=lambda b: b["catalog_id"])
    ]
    print(json.dumps({"counts": summary.get("counts"), "benchmarks": table}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("--run", required=True)
    s.add_argument("--shards", type=int, default=48)
    s.add_argument("--every", type=int, default=1, help="keep every Nth row (a pilot subset)")
    s.add_argument("--suite-dir", default=str(KIT / "suite-0.2"))
    s.add_argument("--rows", help="split this rows file instead of the suite")
    s.set_defaults(fn=split)
    lp = sub.add_parser("launch")
    lp.add_argument("--run", required=True)
    lp.add_argument("--engine", default="http", choices=["http", "random"])
    lp.add_argument("--profile", default="oso-granite-3b-wide", help="the request's model")
    lp.add_argument("--base-url", help="send to this server instead of starting ours")
    lp.add_argument("--token-env", default="OPENROUTER_API_KEY", help="bearer key for --base-url")
    lp.add_argument("--servers", type=int, default=5)
    lp.add_argument("--report-every", type=float, default=60)
    lp.set_defaults(fn=launch)
    sc = sub.add_parser("score")
    sc.add_argument("--run", required=True)
    sc.add_argument("--suite-dir", default=str(KIT / "suite-0.2"))
    sc.add_argument("--name")
    sc.add_argument("--rows", help="score against this rows file instead of the suite")
    sc.set_defaults(fn=score)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
