"""Answer one System One request: validate limits, render, call the model, apply the protocol."""

import json
import math
import time
from dataclasses import dataclass
from typing import Any

from open_system_one import strategies
from open_system_one.backends import CallResult, ModelClient
from open_system_one.config import Profile
from open_system_one.prompts.config import PromptConfig
from open_system_one.prompts.render import AnyQuestion, RenderedQuestion, render_question
from open_system_one.protocol import (
    LabelReading,
    build_answer,
    positions_from_response,
    read_labels,
)
from open_system_one.schema.models import (
    ChoiceQuestion,
    ScoreQuestion,
    SystemOneRequest,
    SystemOneResponse,
)
from open_system_one.tracing import Timer


class LimitError(Exception):
    """A request that is valid under TypeSafe's schema but exceeds this profile's limits (→ 422)."""

    def __init__(self, errors: list[dict[str, Any]]):
        super().__init__(errors[0]["msg"])
        self.errors = errors


class InvariantError(Exception):
    """Our answer violates an invariant from api-schema.md (→ 500, never a partial answer)."""


@dataclass
class _Unique:
    question: AnyQuestion
    keys: list[str]  # request keys that asked this exact question
    rendered: RenderedQuestion | None = None
    reading: LabelReading | None = None


def _limit_error(key: str, kind: str, q, error_type: str, msg: str, **ctx) -> dict[str, Any]:
    n = len(q.criteria)
    return {
        "type": error_type,
        "loc": ["body", "questions", key, kind, "criteria"],
        "msg": msg,
        "input": q.model_dump(mode="json")["criteria"],
        "ctx": {"field_type": "criteria", "actual_length": n, **ctx},
    }


def check_limits(request: SystemOneRequest, profile: Profile, prompt: PromptConfig) -> None:
    """Option and level limits. Messages use Jev's wording ("options per choice", "a choice needs
    at least two options", "a score takes 2 to 10 levels"), which clients such as the Decision
    Index kit match to classify capacity rejections. The per-profile option cap is the one
    declared deviation from Jev's limits."""
    errors = []
    for key, wrapped in request.questions.items():
        q = wrapped.root
        if isinstance(q, ChoiceQuestion):
            n = len(q.criteria)
            limit = min(profile.max_options, prompt.questions.choice.max_labels())
            if n < 2:
                errors.append(
                    _limit_error(
                        key,
                        "choice",
                        q,
                        "too_short",
                        f"a choice needs at least two options; got {n}",
                        min_length=2,
                    )
                )
            elif n > limit:
                errors.append(
                    _limit_error(
                        key,
                        "choice",
                        q,
                        "too_long",
                        f"This model accepts at most {limit} options per choice; got {n}",
                        max_length=limit,
                    )
                )
        elif isinstance(q, ScoreQuestion):
            n, limit = len(q.criteria), len(prompt.questions.score.labels)
            if n < 2 or n > limit:
                errors.append(
                    _limit_error(
                        key,
                        "score",
                        q,
                        "too_short" if n < 2 else "too_long",
                        f"a score takes 2 to {limit} levels; got {n}",
                        min_length=2,
                        max_length=limit,
                    )
                )
    if errors:
        raise LimitError(errors)


def request_top_logprobs(profile: Profile, n_labels: int) -> int:
    """How many top logprobs to request for a call with `n_labels` answer labels.

    Small questions ask for the profile's base amount; large ones ask for twice their label count
    (so labels still show up when other tokens outrank some of them), capped at the backend's max.
    """
    return min(profile.top_logprobs, max(profile.base_top_logprobs, 2 * n_labels))


def _dedupe(request: SystemOneRequest) -> list[_Unique]:
    uniques: dict[str, _Unique] = {}
    for key, wrapped in request.questions.items():
        q = wrapped.root
        # Insertion-ordered JSON: criteria order matters (it is the option order).
        canon = json.dumps(q.model_dump(mode="json", exclude_unset=True), ensure_ascii=False)
        uniques.setdefault(canon, _Unique(question=q, keys=[])).keys.append(key)
    return list(uniques.values())


def _usage(results: list[CallResult], uniques: list[_Unique]) -> tuple[dict[str, int], bool]:
    input_tokens = output_tokens = 0
    estimated = False
    for result, u in zip(results, uniques, strict=True):
        usage = result.raw.get("usage") or {}
        if "prompt_tokens" in usage and "completion_tokens" in usage:
            input_tokens += usage["prompt_tokens"]
            output_tokens += usage["completion_tokens"]
        else:  # rough fallback: ~4 characters per token
            estimated = True
            input_tokens += sum(len(m["content"]) for m in u.rendered.messages) // 4
            output_tokens += 1
    return {"input_tokens": input_tokens, "output_tokens": output_tokens}, estimated


def _check_invariants(request: SystemOneRequest, answers: dict[str, dict[str, Any]]) -> None:
    if list(answers) != list(request.questions):
        raise InvariantError("answer keys don't match question keys")
    for key, answer in answers.items():
        if answer["type"] != request.questions[key].root.type:
            raise InvariantError(f"answer type mismatch for {key}")
        probs = (
            [answer["noul"]] if answer["type"] == "noul" else list(answer["probabilities"].values())
        )
        if not all(math.isfinite(p) and 0.0 <= p <= 1.0 for p in probs):
            raise InvariantError(f"probability out of range for {key}")
        if answer["type"] != "noul" and abs(sum(probs) - 1.0) > 1e-6:
            raise InvariantError(f"probabilities for {key} sum to {sum(probs)}")


async def answer_request(
    request: SystemOneRequest,
    profile: Profile,
    prompt: PromptConfig,
    client: ModelClient,
    timer: Timer,
    trace: dict[str, Any],
) -> dict[str, Any]:
    """The response body for `request`. Fills `trace` as it goes, so failures are traced too."""
    check_limits(request, profile, prompt)
    uniques = _dedupe(request)
    trace["n_questions"] = len(request.questions)
    trace["n_calls"] = len(uniques)

    t = time.perf_counter()
    for u in uniques:
        u.rendered = render_question(request.state, u.question, prompt)
    timer.add("render", time.perf_counter() - t)

    calls: list[dict[str, Any]] = []
    trace["calls"] = calls

    async def call(u: _Unique) -> CallResult:
        entry: dict[str, Any] = {
            "keys": u.keys,
            "messages": u.rendered.messages,
            "labels": u.rendered.labels,
            "options": u.rendered.options,
            "started_ms": round((time.perf_counter() - t_calls) * 1000, 1),
        }
        calls.append(entry)
        top_k = request_top_logprobs(profile, len(u.rendered.labels))
        entry["top_logprobs"] = top_k
        result = await client.complete(
            u.rendered.messages, prefill=u.rendered.prefill, top_logprobs=top_k
        )
        entry.update(
            queue_wait_ms=round(result.queue_wait * 1000, 1),
            duration_ms=round(result.duration * 1000, 1),
            response=result.raw,
        )
        reading = read_labels(
            positions_from_response(result.raw),
            u.rendered.labels,
            prompt.matching,
            prompt.answer.position,
        )
        entry["reading"] = {
            "position": reading.position,
            "raw": reading.raw,
            "missing": reading.missing,
            "label_mass": reading.mass,
            "probs": reading.probs,
        }
        u.reading = reading
        return result

    t_calls = time.perf_counter()
    results = await strategies.run(profile.strategy, uniques, call)
    timer.add("model", time.perf_counter() - t_calls)

    answers: dict[str, dict[str, Any]] = {}
    for u in uniques:
        legend = None
        if isinstance(u.question, ScoreQuestion):
            legend = {str(i): desc for i, desc in enumerate(u.question.criteria)}
        answer = build_answer(
            u.question.type, u.rendered.options, u.rendered.labels, u.reading, legend
        )
        for key in u.keys:
            answers[key] = answer
    answers = {key: answers[key] for key in request.questions}  # request order
    _check_invariants(request, answers)

    usage, estimated = _usage(results, uniques)
    trace["usage_estimated"] = estimated
    body = {"model": profile.name, "answers": answers, "usage": usage}
    SystemOneResponse.model_validate(body)  # schema check against the snapshot's models
    return body
