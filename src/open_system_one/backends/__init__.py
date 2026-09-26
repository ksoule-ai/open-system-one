"""Profile → model client. All model calls go through Mellea (see .claude/rules/mellea.md)."""

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Protocol

from open_system_one.config import Profile

log = logging.getLogger(__name__)


class BackendOverloaded(Exception):
    """The backend is rate-limited, overloaded, or not ready (→ 529 with retry-after)."""

    def __init__(self, message: str, retry_after: float = 1.0):
        super().__init__(message)
        self.retry_after = retry_after


class BackendError(Exception):
    """Any other backend failure (→ 500)."""


@dataclass
class CallResult:
    raw: dict[str, Any]  # the backend's full chat-completion response
    queue_wait: float  # seconds waiting for a concurrency slot
    duration: float  # seconds for the call itself


class ModelClient(Protocol):
    async def complete(self, messages: list[dict[str, str]], *, prefill: bool) -> CallResult: ...


class MelleaClient:
    """An OpenAI-compatible endpoint (HF Inference Endpoint / vLLM, OpenRouter) via Mellea.

    Non-streaming, so `mot.raw.response` keeps the logprobs (Mellea's streaming merge drops them).
    The Mellea backend is built lazily: its constructor probes the server synchronously.
    """

    def __init__(self, profile: Profile):
        self.profile = profile
        self._semaphore = asyncio.Semaphore(profile.concurrency)
        self._backend = None
        self._lock = asyncio.Lock()

    async def _get_backend(self):
        if self._backend is None:
            async with self._lock:
                if self._backend is None:
                    self._backend = await asyncio.to_thread(self._build_backend)
        return self._backend

    def _build_backend(self):
        from mellea.backends.openai import OpenAIBackend

        p = self.profile
        api_key = os.environ.get(p.api_key_env, "") if p.api_key_env else ""
        if not api_key:
            raise BackendError(f"profile {p.name}: env var {p.api_key_env} is not set")
        if not p.base_url:
            raise BackendError(f"profile {p.name}: base_url is empty")
        # Retries belong to the caller (the SDK retries 429 / 5xx); fail fast here.
        return OpenAIBackend(
            model_id=p.model_id,
            base_url=p.base_url,
            api_key=api_key,
            timeout=p.timeout,
            max_retries=0,
        )

    def _model_options(self, prefill: bool) -> dict[str, Any]:
        from mellea.backends.model_options import ModelOption

        p = self.profile
        extra_body = dict(p.extra_body)
        if prefill and p.backend == "hf_endpoint":
            # vLLM continues the final assistant message instead of starting a new turn.
            extra_body.update({"continue_final_message": True, "add_generation_prompt": False})
        options: dict[str, Any] = {
            ModelOption.MAX_NEW_TOKENS: p.max_tokens,
            ModelOption.TEMPERATURE: p.temperature,
            "logprobs": True,
            "top_logprobs": p.top_logprobs,
        }
        if extra_body:
            options["extra_body"] = extra_body
        return options

    async def complete(self, messages: list[dict[str, str]], *, prefill: bool) -> CallResult:
        import openai
        from mellea.stdlib.components.chat import Message
        from mellea.stdlib.context.chat import ChatContext

        backend = await self._get_backend()
        ctx = ChatContext()
        for m in messages[:-1]:
            ctx = ctx.add(Message(m["role"], m["content"]))
        action = Message(messages[-1]["role"], messages[-1]["content"])

        t0 = time.perf_counter()
        async with self._semaphore:
            t1 = time.perf_counter()
            try:
                mot, _ = await backend.generate_from_context(
                    action, ctx, model_options=self._model_options(prefill)
                )
                await mot.avalue()
            except openai.APIStatusError as e:
                if e.status_code in (429, 503, 529):
                    raise BackendOverloaded(
                        f"backend returned {e.status_code}", _retry_after(e.response)
                    ) from e
                raise BackendError(f"backend returned {e.status_code}: {e.message}") from e
            except openai.APITimeoutError as e:
                raise BackendOverloaded("backend call timed out") from e
            except openai.APIConnectionError as e:
                raise BackendError(f"could not reach backend: {e}") from e
            t2 = time.perf_counter()

        raw = mot.raw.response
        if not isinstance(raw, dict):
            raw = raw.model_dump()
        return CallResult(raw=raw, queue_wait=t1 - t0, duration=t2 - t1)


def _retry_after(response) -> float:
    try:
        return max(0.0, float(response.headers.get("retry-after", "1")))
    except (TypeError, ValueError):
        return 1.0


def make_client(profile: Profile) -> ModelClient:
    if profile.backend in ("hf_endpoint", "openrouter"):
        return MelleaClient(profile)
    raise BackendError(f"backend {profile.backend!r} is not implemented yet")
