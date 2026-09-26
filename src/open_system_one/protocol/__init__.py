"""The label-token probability protocol (CLAUDE.md > Design Decisions > 2).

P(option) = P(option's label token) / Σ P(all label tokens), read from the answer position's
top logprobs. A label's token variants (whitespace, case, per the prompt config) are summed.
"""

import math
from dataclasses import dataclass, field
from typing import Any

from open_system_one.prompts.config import MatchingConfig

# One generated position: its top-k candidates as (token, logprob).
Position = list[tuple[str, float]]


class ProtocolError(Exception):
    """The backend's output can't produce an answer (no logprobs, or no label in the top-k)."""


@dataclass
class LabelReading:
    """What was read for one question, before and after normalization."""

    position: int  # index of the generated position the answer was read at
    raw: dict[str, float]  # label → summed probability as returned (0 when absent)
    missing: list[str]  # labels absent from the top-k
    mass: float  # Σ raw label probabilities before normalization
    probs: dict[str, float] = field(default_factory=dict)  # label → normalized probability


def positions_from_response(raw: dict[str, Any]) -> list[Position]:
    """Top-logprob lists per generated position from an OpenAI-style chat completion."""
    choices = raw.get("choices") or []
    if not choices:
        return []
    content = (choices[0].get("logprobs") or {}).get("content") or []
    return [
        [(t["token"], t["logprob"]) for t in (pos.get("top_logprobs") or [])] for pos in content
    ]


def _norm(token: str, matching: MatchingConfig) -> str:
    if matching.strip_whitespace:
        token = token.strip()
    if not matching.case_sensitive:
        token = token.casefold()
    return token


def _label_probs(
    position: Position, labels: list[str], matching: MatchingConfig
) -> dict[str, float]:
    wanted = {_norm(label, matching): label for label in labels}
    if len(wanted) != len(labels):
        raise ValueError(f"labels collide under the matching rules: {labels}")
    probs = dict.fromkeys(labels, 0.0)
    for token, logprob in position:
        label = wanted.get(_norm(token, matching))
        if label is not None:
            probs[label] += math.exp(logprob)
    return probs


def read_labels(
    positions: list[Position],
    labels: list[str],
    matching: MatchingConfig,
    answer_position: str = "first_token",
) -> LabelReading:
    """Read and normalize label probabilities. Raises ProtocolError when no label is readable."""
    if not positions or not positions[0]:
        raise ProtocolError("backend returned no logprobs")

    index = 0
    if answer_position == "first_label_position":
        for i, pos in enumerate(positions):
            if any(p > 0 for p in _label_probs(pos, labels, matching).values()):
                index = i
                break
    position = positions[index]
    raw = _label_probs(position, labels, matching)
    missing = [label for label, p in raw.items() if p == 0.0]
    mass = sum(raw.values())
    if mass == 0.0:
        seen = [t for t, _ in position[:5]]
        raise ProtocolError(f"no answer label in the top-k at position {index} (top: {seen})")

    filled = dict(raw)
    if missing and matching.missing_label == "floor":
        floor = min(math.exp(lp) for _, lp in position)
        for label in missing:
            filled[label] = floor
    total = sum(filled.values())
    probs = {label: p / total for label, p in filled.items()}
    return LabelReading(position=index, raw=raw, missing=missing, mass=mass, probs=probs)


def confidence(probs: list[float]) -> float:
    """TypeSafe's published approximation: clip((N · max p − 1) / (N − 1), 0, 1)."""
    n = len(probs)
    if n <= 1:
        return 1.0
    return min(1.0, max(0.0, (n * max(probs) - 1) / (n - 1)))


def build_answer(
    qtype: str, options: list[str], labels: list[str], reading: LabelReading, legend=None
):
    """The Jev answer object for one question, from its normalized label probabilities."""
    by_option = {opt: reading.probs[label] for opt, label in zip(options, labels, strict=True)}
    if qtype == "noul":
        return {"type": "noul", "noul": by_option["true"]}
    if qtype == "choice":
        best = max(options, key=lambda o: by_option[o])  # max keeps the first on ties
        return {
            "type": "choice",
            "choice": best,
            "probabilities": by_option,
            "confidence": confidence(list(by_option.values())),
        }
    return {
        "type": "score",
        "score": sum(int(level) * p for level, p in by_option.items()),
        "confidence": confidence(list(by_option.values())),
        "legend": legend,
        "probabilities": by_option,
    }
