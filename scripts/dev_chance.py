"""Chance-corrected dev-set scores: chance levels measured on the dev rows, then every dev run's
per-benchmark score turned into skill the way the Decision Index does it.

Chance per benchmark, on the dev rows themselves (not the suite's constants, which assume the
suite's label balance; e.g. the dev set balances RAGTruth 50/50, the suite has 35% hallucinated):
  default        mean native score of the kit's random engine (uniform choice over the supplied
                 options) over --seeds seeds, scored by the same scorer as the runs (kit 0.2)
  RAGTruth       F1 of always answering "hallucinated" on the dev labels (the 0.2.1 board's rule)
  ForecastBench  the board's baseline rule: skill = clip((0.25 - Brier) / 0.25)
skill = clip((score x coverage - chance) / (1 - chance), 0, 1); coverage = answered / requests,
so unanswered and unsupported requests count as wrong.

Run with the kit's Python (it imports decision_index):
    K=external/decision-index/.venv/bin/python
    $K scripts/dev_chance.py                      # writes runs/di-dev/chance.json, prints tables
"""

import argparse
import contextlib
import gzip
import io
import json
import statistics
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from decision_index_run import score_rows

ROWS = Path("evals/decision-index-dev/dev-rows.jsonl.gz")
RUNS = {
    "3B wide @3": "runs/di-dev/oso-granite-3b-wide",
    "3B 20-opt @3": "runs/di-dev/oso-granite-3b-v3",
    "3B wide @4 (doc)": "runs/di-dev/oso-granite-3b-wide-doc",
    "Micro @3": "runs/di-dev/oso-granite-micro-cf-v3",
    "Micro @1": "runs/di-dev/oso-granite-micro-cf",
    "Jev 1.13": "runs/di-dev/jev-1.13",
}
FORECAST, RAGTRUTH = 48, 59


def summary(results_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        score_rows(ROWS, results_path, "chance", out_dir)
    data = json.loads((out_dir / "benchmark-summary.json").read_text())
    return {b["catalog_id"]: b for b in data["benchmarks"]}


def random_chance(seeds: int) -> dict[int, float]:
    from decision_index.engines.base import RandomEngine

    with gzip.open(ROWS, "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    scores: dict[int, list[float]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for seed in range(seeds):
            engine = RandomEngine(seed=seed)
            path = Path(tmp) / f"random-{seed}.jsonl"
            with path.open("w") as f:
                for r in rows:
                    response, _ = engine(r["state"], r["questions"])
                    f.write(
                        json.dumps(
                            dict(r["_evaluation"])
                            | {
                                "status": "ok",
                                "response": response,
                                "total_wall_ms": 0,
                                "model_request_wall_ms": 0,
                            }
                        )
                        + "\n"
                    )
            for n, b in summary(path, Path(tmp) / f"out-{seed}").items():
                if b["score"] is not None:
                    scores.setdefault(n, []).append(b["score"])
    return {n: statistics.mean(v) for n, v in scores.items()}


def ragtruth_chance() -> float:
    with gzip.open(ROWS, "rt", encoding="utf-8") as f:
        labels = [
            r["expected"]["q"]
            for r in map(json.loads, f)
            if r["_evaluation"]["catalog_id"] == RAGTRUTH
        ]
    p = sum(labels) / len(labels)
    return 2 * p / (1 + p)


def skill(n: int, b: dict, chance: dict[int, float]) -> float | None:
    if b is None or b.get("score") is None or n not in chance and n != FORECAST:
        return None
    coverage = b["answered"] / b["requests"] if b["requests"] else 0.0
    if n == FORECAST:
        return max(0.0, min(1.0, (0.25 - b["score"]) / 0.25)) * coverage
    c = chance[n]
    return max(0.0, min(1.0, (b["score"] * coverage - c) / (1 - c)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default="runs/di-dev/chance.json")
    args = ap.parse_args()
    chance = random_chance(args.seeds)
    chance[RAGTRUTH] = ragtruth_chance()
    chance.pop(FORECAST, None)
    runs = {}
    for label, run in RUNS.items():
        path = Path(run) / "merged/benchmark-summary.json"
        if path.exists():
            runs[label] = {b["catalog_id"]: b for b in json.loads(path.read_text())["benchmarks"]}
    names = {}
    for bs in runs.values():
        names.update({n: b["dataset"] for n, b in bs.items()})
    table = {
        n: {
            "benchmark": names[n],
            "chance": chance.get(n),
            **{label: skill(n, bs.get(n), chance) for label, bs in runs.items()},
        }
        for n in sorted(names)
    }
    Path(args.out).write_text(
        json.dumps({"seeds": args.seeds, "chance": chance, "skill": table}, indent=2)
    )
    labels = list(runs)
    print("| # | Benchmark | chance | " + " | ".join(labels) + " |")
    for n, row in table.items():
        cells = ["–" if row[x] is None else f"{row[x]:.3f}" for x in labels]
        ch = (
            "Brier 0.25"
            if n == FORECAST
            else ("–" if row["chance"] is None else f"{row['chance']:.3f}")
        )
        print(f"| {n} | {row['benchmark'][:32]} | {ch} | " + " | ".join(cells) + " |")
    print(
        "| | mean over benchmarks with a skill | | "
        + " | ".join(
            f"{statistics.mean(v):.3f}"
            if (v := [row[x] for row in table.values() if row[x] is not None])
            else "–"
            for x in labels
        )
        + " |"
    )


if __name__ == "__main__":
    main()
