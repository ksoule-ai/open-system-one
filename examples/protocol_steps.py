"""The label-token protocol one step at a time, for wiring it into your own code.

    render  → the question as chat messages, its options under single-token labels
    call    → one generated token with top logprobs (through Mellea, non-streaming)
    read    → each label's probability at the answer position, normalized over the labels
    answer  → the Jev answer object (noul / choice / score)

    uv run --env-file .env python examples/protocol_steps.py --model oso-granite-micro-cf-v3-litellm

examples/decide.py runs the same steps through the engine (dedup, batching strategy, limits, traces).
"""

import argparse
import asyncio
import json

from open_system_one.backends import make_client
from open_system_one.config import load_registry
from open_system_one.engine import request_top_logprobs
from open_system_one.prompts.config import load_prompt
from open_system_one.prompts.render import render_question
from open_system_one.protocol import build_answer, positions_from_response, read_labels
from open_system_one.schema.models import SystemOneRequest

STATE = (
    "Subject: Payouts failing\nOur payouts to the Berlin account have failed three days in a row."
)
QUESTION = {
    "type": "choice",
    "instructions": "Which team should handle this ticket?",
    "criteria": {
        "billing": "Payments, payouts, invoices",
        "technical": "Bugs and outages",
        "sales": "Pricing and new contracts",
    },
}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="oso-latest")
    args = ap.parse_args()
    profile = load_registry("configs/models.yaml").resolve(args.model)
    prompt = load_prompt("configs/prompts", profile.prompt)
    question = (
        SystemOneRequest.model_validate(
            {"model": args.model, "state": STATE, "questions": {"q": QUESTION}}
        )
        .questions["q"]
        .root
    )

    # 1. render: messages, labels (A, B, C), and the option each label stands for
    rendered = render_question(STATE, question, prompt)
    print("labels:", dict(zip(rendered.labels, rendered.options, strict=True)))
    print("user message:\n" + rendered.messages[-1]["content"])

    # 2. call: one token, top logprobs (enough of them that every label can appear)
    result = await make_client(profile).complete(
        rendered.messages,
        prefill=rendered.prefill,
        top_logprobs=request_top_logprobs(profile, len(rendered.labels)),
        documents=rendered.documents or None,
    )

    # 3. read: label probabilities at the answer position, then normalized
    reading = read_labels(
        positions_from_response(result.raw),
        rendered.labels,
        prompt.matching,
        prompt.answer.position,
    )
    print(f"label mass {reading.mass:.3f}, raw {reading.raw}, missing {reading.missing}")

    # 4. answer: the Jev-shaped answer for this question
    print(
        json.dumps(
            build_answer(question.type, rendered.options, rendered.labels, reading), indent=2
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
