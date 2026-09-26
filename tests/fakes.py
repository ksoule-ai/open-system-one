"""A fake model client that returns OpenAI-shaped responses with chosen top logprobs."""

import math
from collections.abc import Callable

from open_system_one.backends import BackendOverloaded, CallResult

# Top-k at the answer position: label variants for every default label scheme, plus noise.
DEFAULT_TOP = {
    "Yes": 0.60,
    " Yes": 0.05,
    "No": 0.25,
    "A": 0.50,
    "B": 0.30,
    "C": 0.10,
    "1": 0.15,
    "2": 0.03,
    "The": 0.02,
}


def chat_completion(top: dict[str, float], usage: tuple[int, int] | None = (100, 1)) -> dict:
    token = max(top, key=top.get) if top else ""
    content = (
        [
            {
                "token": token,
                "logprob": math.log(top[token]) if top else 0.0,
                "top_logprobs": [{"token": t, "logprob": math.log(p)} for t, p in top.items()],
            }
        ]
        if top
        else []
    )
    raw = {
        "choices": [
            {"message": {"role": "assistant", "content": token}, "logprobs": {"content": content}}
        ]
    }
    if usage:
        raw["usage"] = {"prompt_tokens": usage[0], "completion_tokens": usage[1]}
    return raw


class FakeClient:
    def __init__(
        self,
        top: dict[str, float] | Callable[[list[dict]], dict] = DEFAULT_TOP,
        overloaded: bool = False,
    ):
        self.top = top
        self.overloaded = overloaded
        self.calls: list[list[dict]] = []

    async def complete(self, messages, *, prefill):
        self.calls.append(messages)
        if self.overloaded:
            raise BackendOverloaded("fake overload", retry_after=2)
        top = self.top(messages) if callable(self.top) else self.top
        return CallResult(raw=chat_completion(top), queue_wait=0.0, duration=0.001)
