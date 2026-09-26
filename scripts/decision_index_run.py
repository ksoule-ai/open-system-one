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

    suite = Suite(Path(args.suite_dir), "0.2")
    keep = suite.in_edition
    shard_dir = Path(args.run) / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    files = [
        (shard_dir / f"shard-{i:03d}.jsonl").open("w", encoding="utf-8", newline="\n")
        for i in range(args.shards)
    ]
    n = seen = 0
    for path in suite.row_paths:
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
    env = {**os.environ, "DECISION_INDEX_API_KEY": key, "OSO_API_KEY": key}
    try:
        urls = []
        if args.engine == "http":
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


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("--run", required=True)
    s.add_argument("--shards", type=int, default=48)
    s.add_argument("--every", type=int, default=1, help="keep every Nth row (a pilot subset)")
    s.add_argument("--suite-dir", default=str(KIT / "suite-0.2"))
    s.set_defaults(fn=split)
    lp = sub.add_parser("launch")
    lp.add_argument("--run", required=True)
    lp.add_argument("--engine", default="http", choices=["http", "random"])
    lp.add_argument("--profile", default="oso-granite-3b-wide")
    lp.add_argument("--servers", type=int, default=5)
    lp.add_argument("--report-every", type=float, default=60)
    lp.set_defaults(fn=launch)
    sc = sub.add_parser("score")
    sc.add_argument("--run", required=True)
    sc.add_argument("--suite-dir", default=str(KIT / "suite-0.2"))
    sc.add_argument("--name")
    sc.set_defaults(fn=score)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
