"""The protocol on real responses recorded from the HF endpoint (Granite 3B on vLLM)."""

import json
from pathlib import Path

import pytest

from open_system_one.prompts.config import MatchingConfig
from open_system_one.protocol import positions_from_response, read_labels

FIXTURE = json.loads(
    (Path(__file__).parent.parent / "fixtures" / "granite_hf_payouts.json").read_text()
)


@pytest.mark.parametrize("call", FIXTURE["calls"], ids=lambda c: "/".join(c["options"]))
def test_recorded_response_reads_cleanly(call):
    positions = positions_from_response(call["response"])
    assert len(positions[0]) == 20  # top_logprobs: 20 honored by vLLM
    reading = read_labels(positions, call["labels"], MatchingConfig())
    assert reading.mass > 0.99 and reading.missing == []
    assert sum(reading.probs.values()) == pytest.approx(1.0)
    generated = call["response"]["choices"][0]["message"]["content"].strip()
    assert max(reading.probs, key=reading.probs.get) == generated
