"""Jev baseline through OpenRouter's System One API, cached on disk (never committed).

The SDK is constructed explicitly with OpenRouter's base URL and key, never from TYPESAFE_* env
vars (those point the SDK at our own server). OpenRouter sends no x-typesafe-request-id, so the
response's request_id is not read.
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from typesafe_sdk import RetryPolicy, TypeSafeClient

OPENROUTER_BASE_URL = "https://openrouter.ai/api"


def request_hash(model: str, state: Any, questions: dict[str, Any]) -> str:
    # Insertion order is kept (no sort_keys): criteria order is part of the question.
    canon = json.dumps({"model": model, "state": state, "questions": questions}, ensure_ascii=False)
    return hashlib.sha256(canon.encode()).hexdigest()[:20]


class JevBaseline:
    def __init__(
        self,
        model: str = "jev-1.13",
        cache_dir: str | Path = "runs/jev-cache",
        refresh: bool = False,
    ):
        self.model = model
        self.cache_dir = Path(cache_dir) / model
        self.refresh = refresh
        self._client: TypeSafeClient | None = None
        self.hits = self.misses = 0

    def _sdk(self) -> TypeSafeClient:
        if self._client is None:
            key = os.environ.get("OPENROUTER_API_KEY", "")
            if not key:
                raise RuntimeError("OPENROUTER_API_KEY is not set (needed for the Jev baseline)")
            self._client = TypeSafeClient(
                base_url=OPENROUTER_BASE_URL,
                api_key=key,
                model=self.model,
                timeout=60,
                retry=RetryPolicy(max_retries=3),
            )
        return self._client

    def answer(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        """{"body", "served_model", "seconds", "cached"} for this request."""
        path = self.cache_dir / f"{request_hash(self.model, state, questions)}.json"
        if path.exists() and not self.refresh:
            self.hits += 1
            return {**json.loads(path.read_text()), "cached": True}
        self.misses += 1
        t0 = time.perf_counter()
        resp = self._sdk().system_one(state, questions)
        entry = {
            "request": {"model": self.model, "state": state, "questions": questions},
            "body": resp.raw_http_response.json(),
            "served_model": resp.model,
            "seconds": time.perf_counter() - t0,
            "time": time.time(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entry, ensure_ascii=False))
        return {**entry, "cached": False}
