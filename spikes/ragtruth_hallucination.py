"""Spike: Granite Switch's built-in hallucination-detection adapter on the RAGTruth dev samples.

Runs Mellea's `rag.flag_hallucinated_content` (the `hallucination_detection` LoRA embedded in
ibm-granite/granite-switch-4.1-3b-preview) on each of the 100 RAGTruth dev samples and calls a
response hallucinated when any sentence is flagged. Scored like the Decision Index benchmark:
F1 on the hallucinated class, plus accuracy.

Unlike a /v1/systemone request (state = {prompt, response}), the adapter wants a question, the
source documents, and the response separately. They come from RAGTruth's own source_info record:
  QA        question = source_info.question, document = source_info.passages
  Summary   question = the prompt's instruction line, document = the news text
  Data2txt  question = the prompt's instruction, document = the structured data (as given)

Calls go through the LiteLLM gateway (LITELLM_BASE_URL / LITELLM_API_INFERENCE_KEY) to the HF
vLLM endpoint; the adapter is selected by chat_template_kwargs.adapter_name (Mellea's
EmbeddedBinding). Adapter configs are discovered from the model's Hugging Face repo.

    set -a; . ./.env; set +a
    uv run python spikes/ragtruth_hallucination.py [--limit N] [--concurrency 8]
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

RAW = Path("evals/decision-index-dev/raw/59-RAGTruth")
OUT = Path("runs/spikes/ragtruth-hallucination")
SERVED = "hf/ibm-granite/granite-switch-4.1-3b-preview"  # the gateway's name for the endpoint
REPO = "ibm-granite/granite-switch-4.1-3b-preview"  # where the adapter configs live


def samples() -> list[dict]:
    src = {}
    for line in (RAW / "source_info.subset.jsonl").open():
        s = json.loads(line)
        src[s["source_id"]] = s
    prov = {}
    for line in (RAW / "provenance.jsonl").open():
        p = json.loads(line)
        prov[p["key"]] = p
    out = []
    for line in (RAW / "response.train.subset.jsonl").open():
        r = json.loads(line)
        if r["id"] not in prov:
            continue
        s = src[r["source_id"]]
        info, prompt = s["source_info"], s["prompt"]
        if s["task_type"] == "QA":
            question, document = info["question"], info["passages"]
        elif s["task_type"] == "Summary":
            question, document = prompt.split("\n", 1)[0], info
        else:  # Data2txt
            question = prompt.split("Structured data:", 1)[0].removeprefix("Instruction:\n").strip()
            document = str(info)
        out.append(
            {
                "id": r["id"],
                "task_type": s["task_type"],
                "label": bool(prov[r["id"]]["label"]),  # True = the response hallucinates
                "question": question,
                "document": document,
                "response": r["response"],
            }
        )
    return out


def make_backend(timeout: float = 120):
    from mellea.backends.openai import OpenAIBackend

    return OpenAIBackend(
        model_id=SERVED,
        base_url=os.environ["LITELLM_BASE_URL"],
        api_key=os.environ["LITELLM_API_INFERENCE_KEY"],
        load_embedded_adapters=True,
        adapter_source=REPO,
        timeout=timeout,
        max_retries=2,
    )


def judge(
    backend, s: dict, model_options: dict | None = None, user_message: str | None = None
) -> dict:
    from mellea import ChatContext, Message
    from mellea.stdlib.components.intrinsic import rag

    # user_message None: the question from RAGTruth's source record; else that fixed text.
    ctx = ChatContext().add(Message("user", user_message or s["question"]))
    t0 = time.perf_counter()
    try:
        records = rag.flag_hallucinated_content(
            s["response"], [s["document"]], ctx, backend, model_options=model_options
        )
        error = None
    except Exception as e:  # noqa: BLE001 - recorded per sample
        records, error = [], f"{type(e).__name__}: {e}"
    flagged = [r for r in records if str(r.get("faithfulness", "")).lower() != "faithful"]
    return {
        "id": s["id"],
        "task_type": s["task_type"],
        "label": s["label"],
        "pred": bool(flagged) if error is None else None,
        "n_sentences": len(records),
        "n_flagged": len(flagged),
        "faithfulness_values": sorted({str(r.get("faithfulness")) for r in records}),
        "records": records,
        "error": error,
        "seconds": round(time.perf_counter() - t0, 2),
    }


def report(results: list[dict]) -> dict:
    def stats(rs):
        done = [r for r in rs if r["pred"] is not None]
        tp = sum(r["pred"] and r["label"] for r in done)
        fp = sum(r["pred"] and not r["label"] for r in done)
        fn = sum(not r["pred"] and r["label"] for r in done)
        # Unanswered samples count as wrong, as in the Decision Index (a miss on its true class).
        fn += sum(r["label"] for r in rs if r["pred"] is None)
        fp += sum(not r["label"] for r in rs if r["pred"] is None)
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        acc = sum(r["pred"] == r["label"] for r in done) / len(rs)
        return {
            "n": len(rs),
            "answered": len(done),
            "f1_hallucinated": round(f1, 3),
            "accuracy": round(acc, 3),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    by_task = {
        t: stats([r for r in results if r["task_type"] == t])
        for t in sorted({r["task_type"] for r in results})
    }
    return {"all": stats(results), "by_task": by_task}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--user-message", help="fixed user turn; default: the source question")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    data = samples()[: args.limit]
    backend = make_backend()
    sem = asyncio.Semaphore(args.concurrency)

    async def one(s):
        async with sem:
            return await asyncio.to_thread(judge, backend, s, None, args.user_message)

    t0 = time.perf_counter()
    results = await asyncio.gather(*(one(s) for s in data))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "results.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = report(results) | {
        "user_message": args.user_message or "<source question>",
        "minutes": round((time.perf_counter() - t0) / 60, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))
    for r in results:
        if r["error"]:
            print("ERR", r["id"], r["error"][:300])


if __name__ == "__main__":
    asyncio.run(main())
