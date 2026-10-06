"""Spike: Granite Switch's factuality-detection (or hallucination_detection, or guardian-core
with groundedness) adapter on the Decision Index 0.2 suite rows of
RAGTruth (59), HoVer (61) and NLI4CT (42), written as kit results so the kit's scorers score them.

An evaluation, not tuning: the adapter setup ("default user turn") was chosen on the dev set and
is fixed here. Inputs come from the request itself (state + instructions; row metadata is not
read):
  RAGTruth  the prompt is split into the user's question/instruction and its source; task type
            from the prompt's opening (QA "Briefly answer…", Summary "Summarize…", Data2txt
            "Instruction:"). user = question, assistant = state.response, documents = [source]
  HoVer     user = instructions' first line, assistant = state.claim, documents = evidence
  NLI4CT    user = instructions' first line, assistant = the statement, documents = the trials
The adapter answers {"score": "yes"|"no"} (yes = factually incorrect); p = P("yes") from the
answer token's logprobs (spikes/nli_adapters_dev.label_probs). Jev-shaped answers:
  RAGTruth noul = p;  HoVer {NOT_SUPPORTED: p, SUPPORTED: 1-p};  NLI4CT {Contradiction: p,
  Entailment: 1-p}, choice = argmax, confidence by the profile formula. Failures are status error.

    K=external/decision-index/.venv/bin/python   # extract the rows once (see README of the run)
    set -a; . ./.env; set +a
    uv run --with huggingface_hub python spikes/factuality_adapter_suite.py --run R --shard 0 --shards 2
"""

import argparse
import concurrent.futures as cf
import datetime
import gzip
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import nli_adapters_dev as nli
import ragtruth_hallucination as rh

HALLUCINATED = {61: "NOT_SUPPORTED", 42: "Contradiction"}
FAITHFUL = {61: "SUPPORTED", 42: "Entailment"}


def between(text: str, start: str, end: str | None) -> str:
    rest = text.split(start, 1)[1]
    return (rest.split(end, 1)[0] if end and end in rest else rest).strip()


def ragtruth_inputs(state: dict) -> tuple[str, list, str]:
    prompt = state["prompt"]
    if prompt.startswith("Briefly answer"):
        question = prompt.split("\n")[1].strip()
        source = between(prompt, "passages:\n", "\nIn case the passages")
    elif prompt.startswith("Summarize"):
        question, rest = prompt.split("\n", 1)
        source = rest.rsplit("output:", 1)[0].strip()
    elif prompt.startswith("Instruction:"):
        question = between(prompt, "Instruction:\n", "Structured data:")
        source = between(prompt, "Structured data:\n", "\nOverview:")
    else:
        raise ValueError(f"unknown RAGTruth prompt: {prompt[:60]!r}")
    if not question or not source:
        raise ValueError("empty question or source")
    return question, [source], state["response"]


def inputs(row: dict) -> tuple[str, list, str]:
    from mellea.stdlib.components.docs.document import Document

    n = row["_evaluation"]["catalog_id"]
    (q,) = row["questions"].values()
    st = row["state"]
    if n == 59:
        return ragtruth_inputs(st)
    user = q["instructions"].split("\n", 1)[0]
    if n == 61:
        return user, [Document(e["text"], title=e["title"]) for e in st["evidence"]], st["claim"]
    trials = [
        Document(f"{st[k]['id']} ({st[k]['section']})\n" + "\n".join(st[k]["text"]), title=k)
        for k in ("primary_trial", "secondary_trial")
        if st.get(k)
    ]
    return user, trials, q["instructions"].split("\n", 1)[1].strip()


HALLUCINATED_RATINGS = ("unfaithful", "partial")
RATINGS = ("faithful", "unfaithful", "partial", "na")


def hallucination_reading(raw: dict, records: list) -> dict:
    """The hallucination adapter's verdict and a soft score.

    answer_p is the agreed rule, 1.0 if any sentence is rated unfaithful or partial, else 0.0.
    p_yes is soft: at each rating token (top-k holds prefixes of two or more ratings),
    q = P(unfaithful or partial) normalized over the four ratings, and p_yes = 1 - prod(1 - q).
    """

    def norm(t):
        return t.strip().strip('"').strip().lower()

    def owner(t):
        hits = [r for r in RATINGS if t and r.startswith(t)]
        return hits[0] if len(hits) == 1 else None

    qs = []
    for pos in (raw["choices"][0].get("logprobs") or {}).get("content") or []:
        mass = {}
        for t in pos["top_logprobs"]:
            if (o := owner(norm(t["token"]))) is not None:
                mass[o] = mass.get(o, 0.0) + math.exp(t["logprob"])
        if len(mass) >= 2:
            total = sum(mass.values())
            qs.append(sum(mass.get(r, 0.0) for r in HALLUCINATED_RATINGS) / total)
    ratings = [str(r.get("faithfulness")) for r in records]
    flagged = any(r.lower() in HALLUCINATED_RATINGS for r in ratings)
    soft = 1 - math.prod(1 - q for q in qs) if qs and len(qs) == len(ratings) else None
    return {
        "answer_p": 1.0 if flagged else 0.0,
        "p_yes": soft if soft is not None else float(flagged),
        "soft_ok": soft is not None,
        "ratings": ratings,
        "rating_positions": len(qs),
    }


def guardian_kwargs() -> dict:
    """guardian-core with Mellea's built-in groundedness criteria and the default scoring schema
    (the last assistant turn), exactly as guardian.guardian_check(criteria="groundedness")."""
    from mellea.stdlib.components.intrinsic import guardian

    return {
        "criteria": guardian.CRITERIA_BANK["groundedness"],
        "scoring_schema": guardian.SCORING_SCHEMA_BANK["assistant_response"],
    }


def answer(n: int, p: float) -> dict:
    if n == 59:
        return {"type": "noul", "noul": p}
    probs = {HALLUCINATED[n]: p, FAITHFUL[n]: 1 - p}
    choice = max(probs, key=probs.get)
    return {
        "type": "choice",
        "choice": choice,
        "probabilities": probs,
        "confidence": max(0.0, min(1.0, 2 * max(probs.values()) - 1)),
    }


def judge(backend, row: dict, adapter: str = "factuality-detection") -> dict:
    from mellea import ChatContext, Message
    from mellea.backends.adapters import AdapterType
    from mellea.backends.model_options import ModelOption
    from mellea.stdlib import functional as mfuncs
    from mellea.stdlib.components.intrinsic.intrinsic import Intrinsic

    e = row["_evaluation"]
    out = dict(e) | {
        "started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "engine": f"granite-switch-{adapter}",
    }
    t0 = time.perf_counter()
    try:
        user, documents, response = inputs(row)
        ctx = (
            ChatContext()
            .add(Message("user", user))
            .add(Message("assistant", response, documents=documents))
        )
        options = {
            ModelOption.TEMPERATURE: 0.0,
            "logprobs": True,
            "top_logprobs": 20,
            "extra_body": {"metadata": {"tags": [f"{adapter}-suite"]}},
        }
        backend.resolve_adapter(adapter)
        mot, _ = mfuncs.act(
            Intrinsic(
                adapter,
                intrinsic_kwargs=guardian_kwargs() if adapter == "guardian-core" else None,
                adapter_types=(AdapterType.ALORA, AdapterType.LORA),
            ),
            ctx,
            backend,
            model_options=options,
            tool_calls=True,
            strategy=None,
        )
        raw = mot.raw.response
        raw = raw.model_dump() if hasattr(raw, "model_dump") else raw
        if adapter == "hallucination_detection":
            reading = hallucination_reading(raw, json.loads(mot.value))
        elif adapter == "guardian-core":
            # Mellea's result processor turns the yes/no answer into P("yes") (risk detected).
            score = float(json.loads(mot.value)["guardian"]["score"])
            reading = nli.label_probs(raw, "yes", "no") | {"p_yes": score, "mellea_score": score}
        else:
            reading = nli.label_probs(raw, "yes", "no")
        if reading.get("p_yes") is None:
            raise ValueError(f"no answer position in {raw['choices'][0]['message']['content']!r}")
        (key,) = row["questions"]
        out |= {
            "status": "ok",
            "response": {
                "model": f"granite-switch-{adapter}",
                "answers": {
                    key: answer(e["catalog_id"], reading.get("answer_p", reading["p_yes"]))
                },
            },
            "adapter": {
                "label": reading.get("ratings") or json.loads(mot.value),
                "p_yes": reading["p_yes"],
                "logprob_yes": reading.get("logprob_yes"),
                "gateway_id": raw.get("id"),
            },
        }
    except Exception as ex:  # noqa: BLE001 - recorded per row; a rerun retries it
        out |= {"status": "error", "error": f"{type(ex).__name__}: {ex}"[:500]}
    out["completed_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
    out["total_wall_ms"] = out["model_request_wall_ms"] = (time.perf_counter() - t0) * 1000
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory holding rows.jsonl.gz")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--timeout", type=float, default=120, help="seconds per adapter call")
    ap.add_argument("--check-inputs", action="store_true", help="parse every row, call nothing")
    ap.add_argument(
        "--adapter",
        default="factuality-detection",
        choices=["factuality-detection", "hallucination_detection", "guardian-core"],
    )
    args = ap.parse_args()
    run = Path(args.run)
    with gzip.open(run / "rows.jsonl.gz", "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for i, line in enumerate(f) if i % args.shards == args.shard]
    if args.check_inputs:
        bad = []
        for r in rows:
            try:
                inputs(r)
            except Exception as ex:  # noqa: BLE001
                bad.append((r["_evaluation"]["run_id"], str(ex)))
        print(json.dumps({"rows": len(rows), "unparseable": len(bad), "examples": bad[:3]}))
        return
    path = run / f"shard-{args.shard:03d}" / "results.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists():
        with path.open() as f:
            for line in f:
                if line.endswith("\n") and (r := json.loads(line))["status"] == "ok":
                    done.add(r["run_id"])
    todo = [r for r in rows if r["_evaluation"]["run_id"] not in done]
    print(f"[shard {args.shard}] {len(done)} done before, {len(todo)} to run", flush=True)
    backend = rh.make_backend(args.timeout)
    t0, n = time.perf_counter(), 0
    with path.open("a") as f, cf.ThreadPoolExecutor(args.workers) as pool:
        for fut in cf.as_completed([pool.submit(judge, backend, r, args.adapter) for r in todo]):
            f.write(json.dumps(fut.result(), ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if n % 500 == 0 or n == len(todo):
                print(
                    f"[shard {args.shard}] {n}/{len(todo)} in {time.perf_counter() - t0:.0f}s",
                    flush=True,
                )


if __name__ == "__main__":
    main()
