"""Answer a Jev-style decision request in-process with the label-token protocol (no server needed).

The same engine as `open-system-one serve`: each question is rendered by the profile's prompt
config with its options under single-token labels, the model generates one token with top
logprobs, and P(option) = P(label) / sum of P(all labels). See docs/logprob-decisions.md.

    uv run --env-file .env python examples/decide.py --model oso-granite-micro-cf-v3-litellm
    uv run --env-file .env python examples/decide.py --model oso-latest --request my_request.json

The model must be a profile (or alias) in configs/models.yaml whose backend env vars are set.
"""

import argparse
import asyncio
import json
from pathlib import Path

from open_system_one.backends import make_client
from open_system_one.config import load_registry
from open_system_one.engine import answer_request
from open_system_one.prompts.config import load_prompt
from open_system_one.schema.models import SystemOneRequest
from open_system_one.tracing import Timer

EXAMPLE = {
    "state": {
        "subject": "Refund not received",
        "body": "I returned the jacket two weeks ago and still have no refund. This is the "
        "third time I'm writing. Fix it today or I'm disputing the charge.",
    },
    "questions": {
        "urgent": {"type": "noul", "instructions": "The customer needs a reply today."},
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this ticket?",
            "criteria": {
                "billing": "Payments, refunds and charges",
                "shipping": None,
                "returns": "Return labels and return status",
            },
        },
        "anger": {
            "type": "score",
            "instructions": "How angry is the customer?",
            "criteria": ["calm", "annoyed", "angry", "furious"],
        },
    },
}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="oso-latest", help="profile or alias in configs/models.yaml")
    ap.add_argument(
        "--request", help="JSON file with state + questions (default: a built-in example)"
    )
    ap.add_argument("--models", default="configs/models.yaml")
    ap.add_argument("--prompts", default="configs/prompts")
    args = ap.parse_args()

    body = json.loads(Path(args.request).read_text()) if args.request else EXAMPLE
    request = SystemOneRequest.model_validate({"model": args.model, **body})
    profile = load_registry(args.models).resolve(args.model)
    if profile is None:
        raise SystemExit(f"unknown model {args.model!r}")
    prompt = load_prompt(args.prompts, profile.prompt)

    trace: dict = {}
    response = await answer_request(request, profile, prompt, make_client(profile), Timer(), trace)

    print(json.dumps(response, indent=2))
    # What the protocol read for each question (the trace the server writes to runs/traces/).
    for call in trace["calls"]:
        reading = call["reading"]
        print(f"\n{', '.join(call['keys'])}: labels {call['labels']} for options {call['options']}")
        print(
            f"  label mass {reading['label_mass']:.3f} (share of the answer token's probability "
            f"on the labels), missing labels {reading['missing']}"
        )
        print(f"  normalized {json.dumps({k: round(v, 3) for k, v in reading['probs'].items()})}")


if __name__ == "__main__":
    asyncio.run(main())
