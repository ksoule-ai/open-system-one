"""Spike: force Granite Switch's hallucination adapter on dev-set benchmarks that ask whether a text
is supported by a source text, and compare with the default@3 and Jev dev runs on the same samples.

Benchmarks and how a converted dev request becomes adapter inputs (oracle routing: fixed per
benchmark here, which a router would have to decide on its own):
  HoVer   documents = state.evidence (one per titled passage), response = state.claim
          hallucinated → NOT_SUPPORTED, else SUPPORTED                               (accuracy)
  NLI4CT  documents = state.primary_trial (+ secondary_trial), response = the statement after the
          first line of the instructions; hallucinated → Contradiction, else Entailment  (macro-F1)
The user turn is the question's instructions (first line), the only question text in the request.

Each sample's per-sentence ratings are kept, and three flag rules are scored afterwards:
  unfaithful | unfaithful+partial | not_faithful (anything but "faithful", NA included)

    set -a; . ./.env; set +a
    uv run --with huggingface_hub python spikes/hallucination_adapter_dev.py [--limit N]
"""

import argparse
import asyncio
import glob
import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ragtruth_hallucination as rh

ROWS = Path("evals/decision-index-dev/dev-rows.jsonl.gz")
OUT = Path("runs/spikes/hallucination-adapter-dev")
BASELINES = {
    "default@3 (3B wide)": "runs/di-dev/oso-granite-3b-wide",
    "default@3 (Micro, Cloudflare)": "runs/di-dev/oso-granite-micro-cf-v3-hover-nli4ct",
    "default@1 (Micro, Cloudflare)": "runs/di-dev/oso-granite-micro-cf",
    "Jev 1.13": "runs/di-dev/jev-1.13",
}
RULES = {
    "unfaithful": {"unfaithful"},
    "unfaithful+partial": {"unfaithful", "partial"},
    "not_faithful": {"unfaithful", "partial", "na"},
}


def trial_text(trial: dict) -> str:
    return f"{trial['id']} ({trial['section']})\n" + "\n".join(trial["text"])


BENCHMARKS = {
    "HoVer claim verification": {
        "metric": "accuracy",
        "options": ("NOT_SUPPORTED", "SUPPORTED"),  # (hallucinated, faithful)
        "inputs": lambda r, q: (
            [rh_document(e["text"], e["title"]) for e in r["state"]["evidence"]],
            r["state"]["claim"],
        ),
    },
    "NLI4CT": {
        "metric": "macro_f1",
        "options": ("Contradiction", "Entailment"),
        "inputs": lambda r, q: (
            [
                rh_document(trial_text(r["state"][k]), k)
                for k in ("primary_trial", "secondary_trial")
                if r["state"].get(k)
            ],
            q["instructions"].split("\n", 1)[1].strip(),
        ),
    },
}


def rh_document(text: str, title: str | None):
    from mellea.stdlib.components.docs.document import Document

    return Document(text, title=title)


def samples(names: list[str]) -> list[dict]:
    out = []
    with gzip.open(ROWS, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            name = r["_evaluation"]["dataset"]
            if name not in names:
                continue
            ((key, q),) = r["questions"].items()
            out.append(
                {
                    "run_id": r["_evaluation"]["run_id"],
                    "benchmark": name,
                    "key": key,
                    "question": q,
                    "row": r,
                    "expected": r["expected"][key],
                }
            )
    return out


def judge(backend, s: dict, user_message: str | None = None) -> dict:
    """`user_message` None: the user turn is the question's first instruction line (first run).
    Otherwise that fixed text, with the instructions ignored."""
    from mellea import ChatContext, Message
    from mellea.stdlib.components.intrinsic import rag

    spec = BENCHMARKS[s["benchmark"]]
    documents, response = spec["inputs"](s["row"], s["question"])
    user = user_message or s["question"]["instructions"].split("\n", 1)[0]
    ctx = ChatContext().add(Message("user", user))
    t0 = time.perf_counter()
    try:
        records, error = rag.flag_hallucinated_content(response, documents, ctx, backend), None
    except Exception as e:  # noqa: BLE001 - recorded per sample
        records, error = [], f"{type(e).__name__}: {e}"
    return {
        "run_id": s["run_id"],
        "benchmark": s["benchmark"],
        "expected": s["expected"],
        "response_checked": response,
        "n_documents": len(documents),
        "ratings": [r.get("faithfulness") for r in records],
        "records": records,
        "error": error,
        "seconds": round(time.perf_counter() - t0, 2),
    }


def predict(result: dict, rule: str) -> str | None:
    if result["error"]:
        return None
    hallucinated, faithful = BENCHMARKS[result["benchmark"]]["options"]
    flagged = any(str(x).lower() in RULES[rule] for x in result["ratings"])
    return hallucinated if flagged else faithful


def metric(name: str, pairs: list[tuple[str | None, str]], options: tuple[str, str]) -> float:
    """Unanswered (None) counts as wrong, as in the Decision Index."""
    if name == "accuracy":
        return sum(p == e for p, e in pairs) / len(pairs)
    f1s = []
    for c in options:
        tp = sum(p == c and e == c for p, e in pairs)
        fp = sum(p == c and e != c for p, e in pairs)
        fn = sum(p != c and e == c for p, e in pairs)
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return sum(f1s) / len(f1s)


def baseline_preds(run_dir: str, run_ids: set[str]) -> dict[str, str | None]:
    preds = {}
    for path in glob.glob(f"{run_dir}/shard-*/results.jsonl"):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r["run_id"] in run_ids:
                    preds[r["run_id"]] = (
                        next(iter(r["response"]["answers"].values()))["choice"]
                        if r["status"] == "ok"
                        else None
                    )
    return preds


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", nargs="*", default=list(BENCHMARKS))
    ap.add_argument("--limit", type=int, help="samples per benchmark")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument(
        "--user-message",
        help="fixed user turn (e.g. 'Tell me about the documents.'); default: the instructions",
    )
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    data = []
    for name in args.benchmarks:
        data += samples([name])[: args.limit]
    backend = rh.make_backend()
    sem = asyncio.Semaphore(args.concurrency)

    async def one(s):
        async with sem:
            return await asyncio.to_thread(judge, backend, s, args.user_message)

    t0 = time.perf_counter()
    results = await asyncio.gather(*(one(s) for s in data))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "results.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    summary = {
        "user_message": args.user_message or "<instructions, first line>",
        "minutes": round((time.perf_counter() - t0) / 60, 1),
        "benchmarks": {},
    }
    for name in args.benchmarks:
        rs = [r for r in results if r["benchmark"] == name]
        spec = BENCHMARKS[name]
        ids = {r["run_id"] for r in rs}
        row = {"n": len(rs), "errors": sum(bool(r["error"]) for r in rs), "metric": spec["metric"]}
        for rule in RULES:
            row[f"adapter[{rule}]"] = round(
                metric(
                    spec["metric"], [(predict(r, rule), r["expected"]) for r in rs], spec["options"]
                ),
                3,
            )
        for label, run_dir in BASELINES.items():
            preds = baseline_preds(run_dir, ids)
            row[label] = round(
                metric(
                    spec["metric"],
                    [(preds.get(r["run_id"]), r["expected"]) for r in rs],
                    spec["options"],
                ),
                3,
            )
        row["expected_counts"] = {o: sum(r["expected"] == o for r in rs) for o in spec["options"]}
        row["median_seconds"] = sorted(r["seconds"] for r in rs)[len(rs) // 2]
        summary["benchmarks"][name] = row
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))
    for r in results:
        if r["error"]:
            print("ERR", r["run_id"], r["error"][:300])


if __name__ == "__main__":
    asyncio.run(main())
