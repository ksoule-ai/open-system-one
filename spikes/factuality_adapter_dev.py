"""Spike: Granite Switch's factuality-detection adapter on RAGTruth, HoVer and NLI4CT (dev set).

Same inputs as the hallucination-adapter runs with the generic user message, so the two adapters
compare like-for-like:
  user       "Tell me about the documents." (--user-message), or with --default-formatting the
             RAGTruth source question / HoVer and NLI4CT first instruction line
  assistant  the response / claim / statement, verbatim
  documents  RAGTruth: the source (passages, article, business data); HoVer: one per evidence
             passage; NLI4CT: primary and secondary trial
Mellea's `guardian.factuality_detection` returns "yes" (the response is factually incorrect) or
"no" for the whole response. "yes" → hallucinated / NOT_SUPPORTED / Contradiction.
Metrics as in the Decision Index: RAGTruth F1 on the hallucinated class, HoVer accuracy, NLI4CT
macro-F1; unanswered samples count as wrong. Calls carry the LiteLLM tag `factuality-dev`.

    set -a; . ./.env; set +a
    uv run --with huggingface_hub python spikes/factuality_adapter_dev.py [--limit N]
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import hallucination_adapter_dev as had
import ragtruth_hallucination as rh

OUT = Path("runs/spikes/factuality-adapter-dev")
RAGTRUTH = "RAGTruth response-level hallucination"
METRIC = {RAGTRUTH: "f1_hallucinated", **{k: v["metric"] for k, v in had.BENCHMARKS.items()}}
OPTIONS = {RAGTRUTH: (True, False), **{k: v["options"] for k, v in had.BENCHMARKS.items()}}


def samples(limit: int | None) -> list[dict]:
    out = []
    prov = {}
    for line in (rh.RAW / "provenance.jsonl").open():
        p = json.loads(line)
        prov[p["key"]] = p["converted_run_ids"][0]
    for s in rh.samples()[:limit]:
        out.append(
            {
                "benchmark": RAGTRUTH,
                "run_id": prov[s["id"]],
                "expected": s["label"],
                "documents": [s["document"]],
                "response": s["response"],
                "default_user": s["question"],  # the source question
            }
        )
    for name, spec in had.BENCHMARKS.items():
        for s in had.samples([name])[:limit]:
            documents, response = spec["inputs"](s["row"], s["question"])
            out.append(
                {
                    "benchmark": name,
                    "run_id": s["run_id"],
                    "expected": s["expected"],
                    "documents": documents,
                    "response": response,
                    "default_user": s["question"]["instructions"].split("\n", 1)[0],
                }
            )
    return out


def judge(backend, s: dict, user_message: str | None) -> dict:
    from mellea import ChatContext, Message
    from mellea.stdlib.components.intrinsic import guardian

    user = user_message or s["default_user"]
    ctx = ChatContext().add(Message("user", user)).add(Message("assistant", s["response"]))
    t0 = time.perf_counter()
    try:
        score = guardian.factuality_detection(
            ctx,
            backend,
            documents=s["documents"],
            model_options={"extra_body": {"metadata": {"tags": ["factuality-dev"]}}},
        )
        error = None
    except Exception as e:  # noqa: BLE001 - recorded per sample
        score, error = None, f"{type(e).__name__}: {e}"
    return {
        "benchmark": s["benchmark"],
        "run_id": s["run_id"],
        "expected": s["expected"],
        "score": score,
        "error": error,
        "seconds": round(time.perf_counter() - t0, 2),
    }


def predict(r: dict):
    """The benchmark's answer for an adapter result; None when unanswered."""
    if r["error"] or str(r["score"]).lower() not in ("yes", "no"):
        return None
    flagged, faithful = OPTIONS[r["benchmark"]]
    return flagged if str(r["score"]).lower() == "yes" else faithful


def f1_hallucinated(pairs) -> float:
    tp = sum(p is True and e for p, e in pairs)
    fp = sum(p is True and not e for p, e in pairs)
    fn = sum(p is not True and e for p, e in pairs)
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def score(name: str, pairs) -> float:
    if METRIC[name] == "f1_hallucinated":
        return f1_hallucinated(pairs)
    return had.metric(METRIC[name], pairs, OPTIONS[name])


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="samples per benchmark")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--user-message", default="Tell me about the documents.")
    ap.add_argument(
        "--default-formatting",
        action="store_true",
        help="user turn = RAGTruth source question / HoVer, NLI4CT first instruction line",
    )
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    if args.default_formatting:
        args.user_message = None
    data = samples(args.limit)
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
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "user_message": args.user_message
        or "<default: RAGTruth source question; HoVer/NLI4CT first instruction line>",
        "minutes": round((time.perf_counter() - t0) / 60, 1),
        "benchmarks": {},
    }
    for name, metric_name in METRIC.items():
        rs = [r for r in results if r["benchmark"] == name]
        if not rs:
            continue
        summary["benchmarks"][name] = {
            "n": len(rs),
            "errors": sum(bool(r["error"]) for r in rs),
            "metric": metric_name,
            "factuality_adapter": round(score(name, [(predict(r), r["expected"]) for r in rs]), 3),
            "scores_seen": sorted({str(r["score"]) for r in rs}),
            "said_yes": sum(str(r["score"]).lower() == "yes" for r in rs),
            "median_seconds": sorted(r["seconds"] for r in rs)[len(rs) // 2],
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))
    for r in results:
        if r["error"]:
            print("ERR", r["run_id"], r["error"][:300])


if __name__ == "__main__":
    asyncio.run(main())
