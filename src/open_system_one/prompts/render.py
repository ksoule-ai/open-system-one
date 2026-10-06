"""Render a (state, question) pair into chat messages using a PromptConfig.

Template variables (all question types):
    instructions   the question's instructions as text (objects / arrays serialized), or None
    labels         the answer labels in use, in order
Noul adds `yes` / `no` (the criteria text for each side, or None) and `yes_label` / `no_label`.
Choice adds `options`: [{label, key, description}] in criteria order (description None when null).
Score adds `options`: [{label, level, description}], lowest level first.
Filters: `or_join` ("A, B, or C"), plus Jinja's built-ins (e.g. `indent`).

Question keys (the names in the request's `questions` map) are never passed to templates.
"""

import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import jinja2
import yaml

from open_system_one.prompts.config import PromptConfig, StructuredFormat
from open_system_one.schema.models import ChoiceQuestion, NoulQuestion, ScoreQuestion

AnyQuestion = NoulQuestion | ChoiceQuestion | ScoreQuestion


@dataclass(frozen=True)
class RenderedQuestion:
    messages: list[dict[str, str]]  # [{"role", "content"}], ready for the backend
    labels: list[str]  # answer labels, one per option, in option order
    options: list[str]  # what each label stands for: "true"/"false", choice keys, or "0".."n"
    prefill: bool  # the last message is an assistant prefill the model should continue
    # Documents for the chat template (the state, when the config places it there); else empty.
    documents: list[dict[str, str]] = field(default_factory=list)


def serialize(value: Any, fmt: StructuredFormat) -> str:
    """Text for a string / object / array value. Strings pass through unchanged."""
    if isinstance(value, str):
        return value
    if fmt == "yaml":
        return yaml.safe_dump(value, sort_keys=False, allow_unicode=True).strip()
    if fmt == "json":
        return json.dumps(value, indent=2, ensure_ascii=False)
    return json.dumps(value, ensure_ascii=False)


def _or_join(items: list[str]) -> str:
    items = [str(i) for i in items]
    if len(items) <= 1:
        return "".join(items)
    if len(items) == 2:
        return f"{items[0]} or {items[1]}"
    return ", ".join(items[:-1]) + f", or {items[-1]}"


_env = jinja2.Environment(
    undefined=jinja2.StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=False,
    autoescape=False,
)
_env.filters["or_join"] = _or_join


@lru_cache(maxsize=256)
def _template(source: str) -> jinja2.Template:
    return _env.from_string(source)


def _norm(token: str, config: PromptConfig) -> str:
    if config.matching.strip_whitespace:
        token = token.strip()
    if not config.matching.case_sensitive:
        token = token.casefold()
    return token


def choice_labels(keys: list[str], config: PromptConfig) -> list[str]:
    """Labels for a choice question's options.

    Schemes are tried in order: `labels`, `collision_labels`, `extended_labels`. The first one with
    enough labels and no option key that looks like one of its labels wins; if every big-enough
    scheme collides, the first big-enough one is used anyway.
    """
    qc = config.questions.choice
    key_forms = {_norm(key, config) for key in keys}
    schemes = [s for s in (qc.labels, qc.collision_labels, qc.extended_labels) if s]
    big_enough = [s for s in schemes if len(s) >= len(keys)]
    if not big_enough:
        raise ValueError(f"{len(keys)} options but at most {qc.max_labels()} labels")
    for scheme in big_enough:
        if not key_forms & {_norm(label, config) for label in scheme[: len(keys)]}:
            return scheme[: len(keys)]
    return big_enough[0][: len(keys)]


def render_state(state: Any, config: PromptConfig) -> str:
    return _template(config.state.template).render(state=serialize(state, config.state.format))


def render_question(state: Any, question: AnyQuestion, config: PromptConfig) -> RenderedQuestion:
    fmt = config.structured_format
    instructions = None if question.instructions is None else serialize(question.instructions, fmt)

    if isinstance(question, NoulQuestion):
        qc = config.questions.noul
        crit = question.criteria
        yes = None if crit is None or crit.true is None else serialize(crit.true, fmt)
        no = None if crit is None or crit.false is None else serialize(crit.false, fmt)
        labels = list(qc.labels)
        options = ["true", "false"]
        variables = {"yes": yes, "no": no, "yes_label": labels[0], "no_label": labels[1]}
    elif isinstance(question, ChoiceQuestion):
        qc = config.questions.choice
        options = list(question.criteria)
        labels = choice_labels(options, config)
        variables = {
            "options": [
                {
                    "label": label,
                    "key": key,
                    "description": None if desc is None else serialize(desc, fmt),
                }
                for label, (key, desc) in zip(labels, question.criteria.items(), strict=True)
            ]
        }
    else:
        qc = config.questions.score
        n = len(question.criteria)
        if n > len(qc.labels):
            raise ValueError(f"{n} score levels but only {len(qc.labels)} labels")
        labels = qc.labels[:n]
        options = [str(i) for i in range(n)]
        variables = {
            "options": [
                {"label": label, "level": i, "description": serialize(desc, fmt)}
                for i, (label, desc) in enumerate(zip(labels, question.criteria, strict=True))
            ]
        }

    question_text = _template(qc.template).render(
        instructions=instructions, labels=labels, **variables
    )
    blocks = {"state": render_state(state, config), "question": question_text.strip()}
    user = config.block_separator.join(blocks[b] for b in config.user_layout)

    messages = []
    if config.system:
        messages.append({"role": "system", "content": config.system.strip()})
    primer = config.answer.primer
    prefill = bool(primer) and config.answer.primer_mode == "assistant"
    if primer and not prefill:
        user = user + config.block_separator + primer
    messages.append({"role": "user", "content": user})
    if prefill:
        messages.append({"role": "assistant", "content": primer})
    documents = []
    if config.state.placement == "document":
        doc = config.state.document
        documents = [{"doc_id": doc.doc_id, "text": blocks["state"]}]
        if doc.title is not None:
            documents[0]["title"] = doc.title
    return RenderedQuestion(
        messages=messages, labels=labels, options=options, prefill=prefill, documents=documents
    )
