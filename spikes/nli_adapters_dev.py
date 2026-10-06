"""Spike: Granite Switch's answerability and factuality-detection adapters on ANLI and ContractNLI
(dev set), recording each adapter's answer and the probability of its "yes" answer, so the scores
can be mapped onto the three-way NLI labels afterwards (spikes/nli_adapters_analyze.py).

Items: ANLI, 100 requests (premise and hypothesis inside the instructions); ContractNLI, 100
requests x 17 hypotheses about one contract (string state), 1,800 items in all. Gold labels are
normalized to entailment / neutral / contradiction (ContractNLI's NotMentioned = neutral).

Adapter inputs (the document is the premise or the contract):
  factuality-detection  user "Tell me about the documents." / assistant = hypothesis, documents on
                        the assistant turn. Output {"score": "yes"|"no"}, yes = factually incorrect.
  answerability         user "Is the following statement true? <hypothesis>", documents on the
                        user turn. Output {"answerability": "answerable"|"unanswerable"};
                        "yes" here means answerable.

Calls go through Mellea exactly as its wrappers do (mfuncs.act on an Intrinsic), with logprobs
and top_logprobs added, so the raw response carries the answer token's distribution. The answer
position is the first generated token whose top-k holds a prefix of both labels; P(label) sums
the top-k tokens that are a prefix of that label only. p_yes = P(yes) / (P(yes) + P(no)).

    set -a; . ./.env; set +a
    uv run --with huggingface_hub python spikes/nli_adapters_dev.py --adapter answerability
"""

import argparse
import concurrent.futures as cf
import gzip
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ragtruth_hallucination as rh

ROWS = Path("evals/decision-index-dev/dev-rows.jsonl.gz")
OUT = Path("runs/spikes/nli-adapters-dev")
NORMALIZE = {
    "entailment": "entailment",
    "neutral": "neutral",
    "notmentioned": "neutral",
    "contradiction": "contradiction",
}
ADAPTERS = {
    # name: (yes label, no label) as the adapter writes them
    "factuality-detection": ("yes", "no"),
    "answerability": ("answerable", "unanswerable"),
}
FACTUALITY_USER = "Tell me about the documents."
ANSWERABILITY_QUESTION = "Is the following statement true? {hypothesis}"


def items(limit: int | None) -> list[dict]:
    out, seen = [], {"ANLI": 0, "ContractNLI": 0}
    with gzip.open(ROWS, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            name = r["_evaluation"]["dataset"]
            if name not in seen or (limit and seen[name] >= limit):
                continue
            seen[name] += 1
            for key, q in r["questions"].items():
                body = q["instructions"].split("\n", 1)[1]
                if name == "ANLI":
                    premise = body.split("Premise:", 1)[1].split("\nHypothesis:", 1)[0].strip()
                    hypothesis = body.split("Hypothesis:", 1)[1].strip()
                    gold = q["criteria"][r["expected"][key]]  # A/B/C → label text
                else:
                    premise, hypothesis, gold = r["state"], body.strip(), r["expected"][key]
                out.append(
                    {
                        "benchmark": name,
                        "run_id": r["_evaluation"]["run_id"],
                        "key": key,
                        "premise": premise,
                        "hypothesis": hypothesis,
                        "gold": NORMALIZE[gold.lower().replace("_", "")],
                    }
                )
    return out


def label_probs(raw: dict, yes: str, no: str) -> dict:
    """The answer position's probabilities for the two labels (see module docstring)."""

    def norm(t):
        return t.strip().strip('"').strip().lower()

    positions = (raw["choices"][0].get("logprobs") or {}).get("content") or []
    for i, pos in enumerate(positions):
        top = [(norm(t["token"]), t["logprob"]) for t in pos["top_logprobs"]]
        # a token "belongs" to a label when it is a non-empty prefix of it and not of the other
        p_yes = sum(
            math.exp(lp) for t, lp in top if t and yes.startswith(t) and not no.startswith(t)
        )
        p_no = sum(
            math.exp(lp) for t, lp in top if t and no.startswith(t) and not yes.startswith(t)
        )
        chosen = norm(pos["token"])
        if (p_yes > 0 and p_no > 0) or (
            chosen and (yes.startswith(chosen) ^ no.startswith(chosen))
        ):
            lp_yes = next(
                (lp for t, lp in top if t and yes.startswith(t) and not no.startswith(t)), None
            )
            lp_no = next(
                (lp for t, lp in top if t and no.startswith(t) and not yes.startswith(t)), None
            )
            return {
                "position": i,
                "token": pos["token"],
                "p_yes_raw": p_yes,
                "p_no_raw": p_no,
                "logprob_yes": lp_yes,
                "logprob_no": lp_no,
                "p_yes": p_yes / (p_yes + p_no) if p_yes + p_no else None,
                "top": pos["top_logprobs"][:8],
            }
    return {"position": None, "p_yes": None}


def judge(backend, item: dict, adapter: str) -> dict:
    from mellea import ChatContext, Message
    from mellea.backends.adapters import AdapterType
    from mellea.backends.model_options import ModelOption
    from mellea.stdlib import functional as mfuncs
    from mellea.stdlib.components.intrinsic.intrinsic import Intrinsic

    if adapter == "factuality-detection":
        ctx = (
            ChatContext()
            .add(Message("user", FACTUALITY_USER))
            .add(Message("assistant", item["hypothesis"], documents=[item["premise"]]))
        )
    else:
        question = ANSWERABILITY_QUESTION.format(hypothesis=item["hypothesis"])
        ctx = ChatContext().add(Message("user", question, documents=[item["premise"]]))
    options = {
        ModelOption.TEMPERATURE: 0.0,
        "logprobs": True,
        "top_logprobs": 20,
        "extra_body": {"metadata": {"tags": [f"nli-{adapter}"]}},
    }
    t0 = time.perf_counter()
    out = {k: item[k] for k in ("benchmark", "run_id", "key", "gold")} | {"adapter": adapter}
    try:
        backend.resolve_adapter(adapter)
        mot, _ = mfuncs.act(
            Intrinsic(adapter, adapter_types=(AdapterType.ALORA, AdapterType.LORA)),
            ctx,
            backend,
            model_options=options,
            tool_calls=True,
            strategy=None,
        )
        raw = mot.raw.response
        raw = raw.model_dump() if hasattr(raw, "model_dump") else raw
        parsed = json.loads(mot.value)
        answer = next(iter(parsed.values())) if isinstance(parsed, dict) else None
        yes, no = ADAPTERS[adapter]
        out |= {
            "answer": answer,
            "answer_yes": str(answer).lower() == yes,
            "gateway_id": raw.get("id"),
            "content": raw["choices"][0]["message"]["content"],
        }
        out |= label_probs(raw, yes, no)
        out["error"] = None
    except Exception as e:  # noqa: BLE001 - recorded per item
        out |= {"answer": None, "p_yes": None, "error": f"{type(e).__name__}: {e}"[:500]}
    out["seconds"] = round(time.perf_counter() - t0, 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", choices=list(ADAPTERS), required=True)
    ap.add_argument("--limit", type=int, help="requests per benchmark")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    data = items(args.limit)
    backend = rh.make_backend()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.adapter}.jsonl"
    # Resume: keep finished items, rerun missing and errored ones.
    kept = {}
    if path.exists():
        with path.open() as f:
            for line in f:
                if line.endswith("\n"):
                    r = json.loads(line)
                    if not r["error"]:
                        kept[(r["run_id"], r["key"])] = r
    data = [it for it in data if (it["run_id"], it["key"]) not in kept]
    print(f"[{args.adapter}] {len(kept)} done before, {len(data)} to run", flush=True)
    with path.open("w") as f:
        for r in kept.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    t0, done = time.perf_counter(), 0
    with path.open("a") as f, cf.ThreadPoolExecutor(args.workers) as pool:
        for r in cf.as_completed([pool.submit(judge, backend, it, args.adapter) for it in data]):
            f.write(json.dumps(r.result(), ensure_ascii=False) + "\n")
            f.flush()
            done += 1
            if done % 100 == 0 or done == len(data):
                print(
                    f"[{args.adapter}] {done}/{len(data)} in {time.perf_counter() - t0:.0f}s",
                    flush=True,
                )
    rows = [json.loads(line) for line in path.open()]
    errors = [r for r in rows if r["error"]]
    print(
        json.dumps(
            {
                "adapter": args.adapter,
                "items": len(rows),
                "errors": len(errors),
                "no_answer_position": sum(r.get("p_yes") is None and not r["error"] for r in rows),
                "minutes": round((time.perf_counter() - t0) / 60, 1),
            }
        )
    )
    for r in errors[:5]:
        print("ERR", r["run_id"], r["key"], r["error"][:200])


if __name__ == "__main__":
    main()
