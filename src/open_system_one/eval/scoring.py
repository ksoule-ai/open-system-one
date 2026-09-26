"""Per-question scoring: against the case's `expect`, and against Jev's answer (eval-design.md)."""

from typing import Any

from open_system_one.protocol import confidence


def _argmax(probs: dict[str, float]) -> str:
    return max(probs, key=probs.get)  # first on ties, like the server


def summarize(answer: dict[str, Any]) -> dict[str, Any]:
    """The comparable core of one Jev-shaped answer."""
    if answer["type"] == "noul":
        p = answer["noul"]
        return {"top": "true" if p > 0.5 else "false", "probs": {"true": p, "false": 1 - p}}
    probs = {str(k): v for k, v in answer["probabilities"].items()}
    out = {"top": _argmax(probs), "probs": probs, "confidence": answer.get("confidence")}
    if answer["type"] == "score":
        out["score"] = answer["score"]
    return out


def against_expect(answer: dict[str, Any], expect: dict[str, Any]) -> dict[str, Any]:
    """correct, P(expected), and Brier (nouls with a yes/no expectation)."""
    s = summarize(answer)
    qtype = answer["type"]
    if "band" in expect:
        lo, hi = expect["band"]
        value = answer["noul"] if qtype == "noul" else answer.get("score")
        return {"correct": lo <= value <= hi, "p_expected": None, "brier": None}
    if "answer_in" in expect:
        allowed = [str(a) for a in expect["answer_in"]]
        p = sum(s["probs"].get(a, 0.0) for a in allowed)
        return {"correct": s["top"] in allowed, "p_expected": p, "brier": None}
    want = expect["answer"]
    if qtype == "noul":
        want = "true" if want else "false"
        p = s["probs"][want]
        return {"correct": s["top"] == want, "p_expected": p, "brier": (1 - p) ** 2}
    want = str(want)
    return {"correct": s["top"] == want, "p_expected": s["probs"].get(want, 0.0), "brier": None}


def against_jev(answer: dict[str, Any], jev: dict[str, Any]) -> dict[str, Any]:
    """Argmax agreement and mean |Δp| over options (plus |Δscore| for scores)."""
    a, j = summarize(answer), summarize(jev)
    keys = list(a["probs"])
    out = {
        "agree": a["top"] == j["top"],
        "dp": sum(abs(a["probs"][k] - j["probs"].get(k, 0.0)) for k in keys) / len(keys),
    }
    if answer["type"] == "noul":
        out["dp"] = abs(a["probs"]["true"] - j["probs"]["true"])
    if answer["type"] == "score":
        out["dscore"] = abs(a["score"] - j["score"])
    return out


def jev_confidence_check(jev: dict[str, Any]) -> float | None:
    """|our confidence formula on Jev's probabilities − Jev's confidence| (choice / score only)."""
    if jev["type"] == "noul":
        return None
    return abs(confidence(list(jev["probabilities"].values())) - jev["confidence"])
