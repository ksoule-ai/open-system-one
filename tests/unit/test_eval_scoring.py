import pytest

from open_system_one.eval.scoring import against_expect, against_jev, jev_confidence_check

NOUL = {"type": "noul", "noul": 0.8}
CHOICE = {"type": "choice", "choice": "a", "probabilities": {"a": 0.7, "b": 0.3}, "confidence": 0.4}
SCORE = {
    "type": "score",
    "score": 1.2,
    "confidence": 0.5,
    "legend": {"0": "x", "1": "y", "2": "z"},
    "probabilities": {"0": 0.1, "1": 0.6, "2": 0.3},
}


def test_against_expect():
    assert against_expect(NOUL, {"answer": True}) == {
        "correct": True,
        "p_expected": 0.8,
        "brier": pytest.approx(0.04),
    }
    assert against_expect(NOUL, {"answer": False})["correct"] is False
    assert against_expect(NOUL, {"band": [0.2, 0.9]})["correct"] is True
    assert against_expect(CHOICE, {"answer": "b"}) == {
        "correct": False,
        "p_expected": 0.3,
        "brier": None,
    }
    assert against_expect(CHOICE, {"answer_in": ["a", "b"]})["p_expected"] == pytest.approx(1.0)
    assert against_expect(SCORE, {"band": [1.0, 1.5]})["correct"] is True
    assert against_expect(SCORE, {"answer": 2}) == {
        "correct": False,
        "p_expected": 0.3,
        "brier": None,
    }


def test_against_jev():
    jev_noul = {"type": "noul", "noul": 0.3}
    assert against_jev(NOUL, jev_noul) == {"agree": False, "dp": pytest.approx(0.5)}
    jev_choice = {**CHOICE, "probabilities": {"b": 0.1, "a": 0.9}}  # Jev's key order differs
    assert against_jev(CHOICE, jev_choice) == {"agree": True, "dp": pytest.approx(0.2)}
    jev_score = {**SCORE, "score": 1.0, "probabilities": {"0": 0.0, "1": 1.0, "2": 0.0}}
    out = against_jev(SCORE, jev_score)
    assert out["agree"] and out["dscore"] == pytest.approx(0.2)


def test_confidence_check():
    assert jev_confidence_check(NOUL) is None
    assert jev_confidence_check(CHOICE) == pytest.approx(0.0)
