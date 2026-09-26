"""Batching strategies: the order in which a request's per-question calls are issued.

Concurrency is capped by the model client (profile `concurrency`), not here.
- sequential:  one call at a time (baseline).
- fanout:      all calls at once.
- warm_fanout: the first call alone, so it fills the backend's prefix cache with the shared
               system + state prefix, then the rest at once.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


async def run(strategy: str, items: list[T], call: Callable[[T], Awaitable[R]]) -> list[R]:
    """Apply `call` to every item under `strategy`; results keep the order of `items`."""
    if strategy == "sequential" or len(items) <= 1:
        return [await call(item) for item in items]
    if strategy == "fanout":
        return list(await asyncio.gather(*(call(item) for item in items)))
    if strategy == "warm_fanout":
        first = await call(items[0])
        rest = await asyncio.gather(*(call(item) for item in items[1:]))
        return [first, *rest]
    raise ValueError(f"unknown strategy {strategy!r}")
