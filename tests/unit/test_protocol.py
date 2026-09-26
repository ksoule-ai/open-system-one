import math

import pytest

from open_system_one.prompts.config import MatchingConfig
from open_system_one.protocol import (
    ProtocolError,
    build_answer,
    confidence,
    positions_from_response,
    read_labels,
)
from tests.fakes import chat_completion


def pos(**probs):
    return [(t.replace("_", " "), math.log(p)) for t, p in probs.items()]


def test_variants_are_summed_and_normalized():
    # " Yes" and "yes" both count for "Yes"; "The" is ignored.
    position = [
        ("Yes", math.log(0.5)),
        (" Yes", math.log(0.1)),
        ("yes", math.log(0.1)),
        ("No", math.log(0.2)),
        ("The", math.log(0.1)),
    ]
    r = read_labels([position], ["Yes", "No"], MatchingConfig())
    assert r.raw == pytest.approx({"Yes": 0.7, "No": 0.2})
    assert r.mass == pytest.approx(0.9)
    assert r.probs == pytest.approx({"Yes": 0.7 / 0.9, "No": 0.2 / 0.9})
    assert r.missing == []


def test_case_sensitive_matching_ignores_other_cases():
    position = [("Yes", math.log(0.5)), ("yes", math.log(0.3)), ("No", math.log(0.2))]
    r = read_labels([position], ["Yes", "No"], MatchingConfig(case_sensitive=True))
    assert r.raw == pytest.approx({"Yes": 0.5, "No": 0.2})


def test_missing_label_zero_and_floor():
    position = [("A", math.log(0.8)), ("B", math.log(0.1)), ("The", math.log(0.05))]
    zero = read_labels([position], ["A", "B", "C"], MatchingConfig())
    assert zero.missing == ["C"] and zero.probs["C"] == 0.0
    floor = read_labels([position], ["A", "B", "C"], MatchingConfig(missing_label="floor"))
    assert floor.probs["C"] == pytest.approx(0.05 / 0.95)


def test_no_logprobs_or_no_labels_is_an_error():
    with pytest.raises(ProtocolError, match="no logprobs"):
        read_labels([], ["Yes", "No"], MatchingConfig())
    with pytest.raises(ProtocolError, match="no answer label"):
        read_labels([[("The", -0.1)]], ["Yes", "No"], MatchingConfig(missing_label="floor"))


def test_first_label_position_skips_leading_tokens():
    positions = [[("\n", -0.01)], [("B", math.log(0.9)), ("A", math.log(0.1))]]
    with pytest.raises(ProtocolError):
        read_labels(positions, ["A", "B"], MatchingConfig())
    r = read_labels(positions, ["A", "B"], MatchingConfig(), "first_label_position")
    assert r.position == 1 and r.probs["B"] == pytest.approx(0.9)


def test_positions_from_response():
    raw = chat_completion({"A": 0.9, "B": 0.1})
    [position] = positions_from_response(raw)
    assert [t for t, _ in position] == ["A", "B"]
    assert positions_from_response({"choices": [{"message": {}, "logprobs": None}]}) == []


def test_confidence_formula():
    assert confidence([1.0, 0.0, 0.0]) == 1.0
    assert confidence([0.5, 0.5]) == 0.0
    assert confidence([0.86, 0.14, 0.0]) == pytest.approx((3 * 0.86 - 1) / 2)
    assert confidence([1.0]) == 1.0


def test_build_answers():
    r = read_labels([pos(A=0.4, B=0.4, C=0.2)], ["A", "B", "C"], MatchingConfig())
    choice = build_answer("choice", ["billing", "tech", "sales"], ["A", "B", "C"], r)
    assert choice["choice"] == "billing"  # tie → first in criteria order
    assert list(choice["probabilities"]) == ["billing", "tech", "sales"]

    r = read_labels([pos(Yes=0.3, No=0.1)], ["Yes", "No"], MatchingConfig())
    assert build_answer("noul", ["true", "false"], ["Yes", "No"], r) == {
        "type": "noul",
        "noul": pytest.approx(0.75),
    }

    r = read_labels([pos(**{"0": 0.1, "1": 0.6, "2": 0.3})], ["0", "1", "2"], MatchingConfig())
    score = build_answer(
        "score", ["0", "1", "2"], ["0", "1", "2"], r, {"0": "a", "1": "b", "2": "c"}
    )
    assert score["score"] == pytest.approx(0.6 + 2 * 0.3)
    assert score["legend"] == {"0": "a", "1": "b", "2": "c"}
