"""Model profiles: `configs/models.yaml` → validated `Profile` objects plus aliases."""

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

Strategy = Literal["sequential", "fanout", "warm_fanout"]
_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class Profile(BaseModel):
    """One servable model: backend, model id, prompt config, strategy, and limits."""

    model_config = ConfigDict(extra="forbid")

    name: str
    backend: Literal["hf_endpoint", "openrouter", "ollama"]
    model_id: str
    base_url: str | None = None
    api_key_env: str | None = None  # name of the env var holding the key; never the key itself
    description: str = ""
    release_date: str  # YYYY-MM-DD, shown by GET /v1/models
    prompt: str  # "<name>@<version>", a file in configs/prompts/
    strategy: Strategy = "warm_fanout"
    top_logprobs: int = Field(20, ge=1)
    max_options: int = Field(20, ge=1)
    max_tokens: int = Field(1, ge=1)
    temperature: float = 0.0
    concurrency: int = Field(4, ge=1)  # simultaneous model calls
    max_inflight_requests: int = Field(64, ge=1)  # requests above this get 429
    timeout: float = 30.0  # seconds per model call
    extra_body: dict[str, Any] = Field(default_factory=dict)  # merged into every call

    @model_validator(mode="after")
    def _check_limits(self) -> "Profile":
        if self.max_options > self.top_logprobs:
            raise ValueError(
                f"profile {self.name}: max_options ({self.max_options}) must be <= "
                f"top_logprobs ({self.top_logprobs})"
            )
        if "@" not in self.prompt:
            raise ValueError(f"profile {self.name}: prompt must be '<name>@<version>'")
        return self

    @property
    def prompt_name(self) -> str:
        return self.prompt.split("@", 1)[0]

    @property
    def prompt_version(self) -> str:
        return self.prompt.split("@", 1)[1]


class Registry(BaseModel):
    """All profiles and aliases from one models.yaml."""

    profiles: dict[str, Profile]
    aliases: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_aliases(self) -> "Registry":
        for alias, target in self.aliases.items():
            if target not in self.profiles:
                raise ValueError(f"alias {alias} points at unknown profile {target}")
            if alias in self.profiles:
                raise ValueError(f"alias {alias} shadows a profile")
        return self

    def resolve(self, model: str) -> Profile | None:
        """The profile a request's `model` names (directly or via an alias), or None."""
        return self.profiles.get(self.aliases.get(model, model))


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_REF.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_registry(path: str | Path) -> Registry:
    """Load models.yaml, expanding `${VAR}` references from the environment."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    profiles = {
        name: Profile(name=name, **_expand_env(body))
        for name, body in (data.get("profiles") or {}).items()
    }
    return Registry(profiles=profiles, aliases=data.get("aliases") or {})
