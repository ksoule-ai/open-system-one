"""Spike: guardian-core groundedness on the RAGTruth / HoVer / NLI4CT dev rows with the prompt sent as
raw text, so the activation sequence can be changed.

The prompt is what the chat path would send (spikes/factuality_adapter_suite.py --adapter
guardian-core): the same inputs (user turn, assistant response, documents) and Mellea's guardian
message (groundedness criteria, default scoring schema), rendered with Granite Switch's chat
template exactly as vLLM renders it. --variant:
  as-rendered     unchanged: "<|guardian-core|>guardian>As a judge agent..." (a control)
  no-invocation   the rest of the invocation text dropped: "<|guardian-core|>As a judge agent..."
It goes to /v1/completions through Mellea (OpenAIBackend.generate_from_raw), so no template is
applied on the server. P(yes) sums every case/quote variant of "yes" and "no" in the answer
token's top-k. Results are kit rows; score them with scripts/decision_index_run.py score --rows.

    set -a; . ./.env; set +a
    uv run --with huggingface_hub python spikes/guardian_raw_dev.py --variant no-invocation \
        --template /path/to/chat_template.jinja --run runs/di-dev/guardian-raw-no-invocation
"""

import argparse
import asyncio
import datetime
import gzip
import json
import math
import sys
import time
from pathlib import Path

import jinja2

sys.path.insert(0, str(Path(__file__).parent))
import factuality_adapter_suite as fs
import ragtruth_hallucination as rh

ROWS = Path("runs/di-dev/guardian-groundedness-dev/rows.jsonl.gz")  # dev rows of 59, 61, 42
ACTIVATION = "<|guardian-core|>"
INVOCATION = "<guardian>"
# Mellea's guardian-core judge message (as logged for the chat path), filled with its kwargs.
JUDGE = (
    "<guardian>As a judge agent, your role is to help assess whether the provided text meets the "
    "given judging criteria, utilizing all available information, including conversations, "
    "documents, and tools.\n\n### Criteria: {criteria}\n\n### Scoring Schema: {scoring_schema}"
)


def template(path: str) -> jinja2.Template:
    env = jinja2.Environment(
        trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
    )
    env.filters["tojson"] = (
        lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(
            x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys
        )
    )

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    env.globals["raise_exception"] = raise_exception
    return env.from_string(Path(path).read_text())


def render(tpl: jinja2.Template, row: dict, variant: str) -> str:
    user, documents, response = fs.inputs(row)
    docs = []
    for d in documents:  # Mellea's messages_to_docs: text, then title / doc_id when set
        if isinstance(d, str):
            docs.append({"text": d})
        else:
            doc = {"text": d.text}
            if d.title is not None:
                doc["title"] = d.title
            if d.doc_id is not None:
                doc["doc_id"] = d.doc_id
            docs.append(doc)
    messages = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": response},
        {"role": "user", "content": JUDGE.format(**fs.guardian_kwargs())},
    ]
    text = tpl.render(
        messages=messages,
        documents=docs,
        tools=None,
        add_generation_prompt=True,
        bos_token="<|end_of_text|>",
        eos_token="<|end_of_text|>",
        adapter_name="guardian-core",
    )
    as_rendered = ACTIVATION + INVOCATION[1:]
    assert text.count(as_rendered) == 1, "activation sequence not found exactly once"
    return text if variant == "as-rendered" else text.replace(as_rendered, ACTIVATION)


def p_yes(choice: dict) -> tuple[str | None, float | None]:
    """P(yes) at the answer position: the first generated token that is itself yes or no (any
    case or quoting); the mass of every yes / no variant in its top-k is summed."""

    def norm(t):
        return t.strip().strip('"').strip().lower()

    lp = choice.get("logprobs") or {}
    for tok, top in zip(lp.get("tokens") or [], lp.get("top_logprobs") or []):
        if norm(tok) not in ("yes", "no"):
            continue
        mass = {"yes": 0.0, "no": 0.0}
        for t, v in (top or {}).items():
            if norm(t) in mass:
                mass[norm(t)] += math.exp(v)
        return tok, mass["yes"] / (mass["yes"] + mass["no"])
    return None, None


async def judge(backend, row: dict, text: str, variant: str, sem) -> dict:
    from mellea.backends.model_options import ModelOption
    from mellea.core.base import CBlock
    from mellea.stdlib.context.chat import ChatContext

    e = row["_evaluation"]
    out = dict(e) | {
        "started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "engine": f"guardian-core-raw-{variant}",
    }
    t0 = time.perf_counter()
    async with sem:
        try:
            options = {
                ModelOption.TEMPERATURE: 0.0,
                ModelOption.MAX_NEW_TOKENS: 16,
                "logprobs": 5,
                "extra_body": {"metadata": {"tags": [f"guardian-raw-{variant}"]}},
            }
            (mot,) = await backend.generate_from_raw(
                [CBlock(text)], ChatContext(), model_options=options
            )
            tok, p = p_yes(mot.raw.response)
            if p is None:
                raise ValueError(f"no yes/no answer token in {mot.value!r}")
            (key,) = row["questions"]
            out |= {
                "status": "ok",
                "response": {
                    "model": f"guardian-core-raw-{variant}",
                    "answers": {key: fs.answer(e["catalog_id"], p)},
                },
                "adapter": {"output": mot.value, "answer_token": tok, "p_yes": p},
            }
        except Exception as ex:  # noqa: BLE001 - recorded per row
            out |= {"status": "error", "error": f"{type(ex).__name__}: {ex}"[:500]}
    out["completed_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
    out["total_wall_ms"] = out["model_request_wall_ms"] = (time.perf_counter() - t0) * 1000
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["as-rendered", "no-invocation"], required=True)
    ap.add_argument("--template", required=True, help="Granite Switch chat_template.jinja")
    ap.add_argument("--run", required=True)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    tpl = template(args.template)
    with gzip.open(ROWS, "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    run = Path(args.run)
    (run / "shard-000").mkdir(parents=True, exist_ok=True)
    (run / "rows.jsonl.gz").write_bytes(ROWS.read_bytes())
    backend = rh.make_backend()
    sem = asyncio.Semaphore(args.workers)
    t0 = time.perf_counter()
    results = await asyncio.gather(
        *(judge(backend, r, render(tpl, r, args.variant), args.variant, sem) for r in rows)
    )
    with (run / "shard-000/results.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "variant": args.variant,
                "rows": len(results),
                "errors": sum(r["status"] != "ok" for r in results),
                "seconds": round(time.perf_counter() - t0),
            }
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
