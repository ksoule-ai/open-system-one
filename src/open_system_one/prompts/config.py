"""Prompt configs: `configs/prompts/<name>.yaml`, versioned and content-hashed.

The config owns every piece of prompt text, the label schemes, and the token-matching rules. Code
only renders what the config says (see .claude/rules/prompts.md).
"""

import hashlib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

StructuredFormat = Literal["yaml", "json", "json_compact"]


class StateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    template: str  # Jinja; gets `state` (already serialized to text)
    format: StructuredFormat = "yaml"  # how object / array states are serialized


class QuestionTypeConfig(BaseModel):
    """Rendering and labels for one question type."""

    model_config = ConfigDict(extra="forbid")

    template: str  # Jinja; see prompts/render.py for the variables it gets
    labels: list[str] = Field(min_length=1)
    # Used instead of `labels` when an option key looks like a label (e.g. keys A-E with letter
    # labels). Only choice questions have keys, so only they use it.
    collision_labels: list[str] | None = None
    # Used when a choice has more options than `labels` (or `collision_labels`) can cover, e.g.
    # numbers 1..256 beyond letters A..T. Every label must be a single token for the model.
    extended_labels: list[str] | None = None

    def max_labels(self) -> int:
        return max(
            len(s) for s in (self.labels, self.collision_labels or [], self.extended_labels or [])
        )

    @model_validator(mode="after")
    def _unique(self) -> "QuestionTypeConfig":
        for scheme in (self.labels, self.collision_labels or [], self.extended_labels or []):
            if len(set(scheme)) != len(scheme):
                raise ValueError(f"labels must be unique: {scheme}")
        return self


class QuestionsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    noul: QuestionTypeConfig
    choice: QuestionTypeConfig
    score: QuestionTypeConfig

    @model_validator(mode="after")
    def _noul_two_labels(self) -> "QuestionsConfig":
        if len(self.noul.labels) != 2:
            raise ValueError("noul needs exactly two labels: [yes-label, no-label]")
        return self


class AnswerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Text that ends the prompt right before the answer token. `user`: appended to the user
    # message. `assistant`: sent as a final assistant message the model continues (prefill); only
    # backends that support continuation honor it (vLLM does, OpenRouter doesn't reliably).
    primer: str = ""
    primer_mode: Literal["user", "assistant"] = "user"
    # Where the answer is read: the first generated token, or the first position (within the
    # profile's max_tokens) whose top-k contains a label.
    position: Literal["first_token", "first_label_position"] = "first_token"


class MatchingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strip_whitespace: bool = True  # " A" and "A" both count for label "A"
    case_sensitive: bool = False  # "yes" and "YES" both count for label "Yes"
    missing_label: Literal["zero", "floor"] = "zero"  # probability for labels absent from top-k


class PromptConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", coerce_numbers_to_str=True)

    name: str
    version: str
    description: str = ""
    system: str = ""
    # Order of blocks inside the user message. Keep `state` first so everything up to the question
    # is a prefix shared by all questions in a request.
    user_layout: list[Literal["state", "question"]] = ["state", "question"]
    block_separator: str = "\n\n"
    structured_format: StructuredFormat = (
        "yaml"  # instructions / criteria that are objects or arrays
    )
    state: StateConfig
    questions: QuestionsConfig
    answer: AnswerConfig = AnswerConfig()
    matching: MatchingConfig = MatchingConfig()

    # Set by the loader, not the file.
    content_hash: str = ""

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"


def load_prompt(prompts_dir: str | Path, ref: str) -> PromptConfig:
    """Load `<prompts_dir>/<name>@<version>.yaml` (or `<name>.yaml`) and check its version."""
    name, version = ref.split("@", 1)
    path = Path(prompts_dir) / f"{ref}.yaml"
    if not path.exists():
        path = Path(prompts_dir) / f"{name}.yaml"
    raw = path.read_bytes()
    config = PromptConfig(**yaml.safe_load(raw))
    if config.name != name or str(config.version) != version:
        raise ValueError(
            f"{path} is {config.name}@{config.version}, but {ref} was requested; "
            "a changed prompt needs a new version"
        )
    config.content_hash = hashlib.sha256(raw).hexdigest()[:12]
    return config
