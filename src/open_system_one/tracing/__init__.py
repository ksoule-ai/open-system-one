"""Per-request traces: one JSON line per request in `<dir>/<YYYY-MM-DD>.jsonl`.

A trace holds everything needed to inspect or replay a request (rendered messages, raw backend
responses with logprobs, label readings, timings, usage). Writes happen after the response is
sent, so they don't add to measured latency.
"""

import json
import threading
import time
from pathlib import Path
from typing import Any


class TraceWriter:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{time.strftime('%Y-%m-%d')}.jsonl"
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock, path.open("a") as f:
            f.write(line + "\n")


class Timer:
    """Named wall-clock spans in milliseconds, for traces and the Server-Timing header."""

    def __init__(self) -> None:
        self.spans: dict[str, float] = {}

    def add(self, name: str, seconds: float) -> None:
        self.spans[name] = self.spans.get(name, 0.0) + seconds * 1000

    def server_timing(self) -> str:
        return ", ".join(f"{name};dur={ms:.1f}" for name, ms in self.spans.items())
