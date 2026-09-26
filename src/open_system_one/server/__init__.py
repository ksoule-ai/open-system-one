"""The Jev-compatible HTTP API: POST /v1/systemone, GET /v1/models (see api-schema.md)."""

import hmac
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.background import BackgroundTask
from starlette.datastructures import MutableHeaders

from open_system_one.backends import BackendError, BackendOverloaded, ModelClient, make_client
from open_system_one.config import Registry
from open_system_one.engine import InvariantError, LimitError, answer_request
from open_system_one.prompts.config import PromptConfig, load_prompt
from open_system_one.protocol import ProtocolError
from open_system_one.schema.models import SystemOneRequest
from open_system_one.tracing import Timer, TraceWriter

log = logging.getLogger(__name__)
REQUEST_ID_HEADER = "x-typesafe-request-id"


class RequestIdMiddleware:
    """Gives every request an id (`request.state.request_id`) and returns it as a header."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request_id = "req_" + uuid.uuid4().hex
        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        state["received_at"] = time.perf_counter()

        async def send_with_id(message):
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message).append(REQUEST_ID_HEADER, request_id)
            await send(message)

        await self.app(scope, receive, send_with_id)


class AuthError(Exception):
    pass


def _error(status: int, detail: Any, headers: dict[str, str] | None = None, background=None):
    return JSONResponse(
        {"detail": detail}, status_code=status, headers=headers, background=background
    )


def create_app(
    registry: Registry,
    prompts_dir: str | Path,
    api_keys: list[str],
    trace_dir: str | Path | None = None,
    clients: dict[str, ModelClient] | None = None,
) -> FastAPI:
    """Build the app. `clients` overrides the model client per profile (tests use fakes)."""
    if not api_keys:
        raise ValueError("at least one API key is required (set OSO_API_KEY)")
    prompts: dict[str, PromptConfig] = {
        name: load_prompt(prompts_dir, p.prompt) for name, p in registry.profiles.items()
    }
    clients = dict(clients or {})
    tracer = TraceWriter(trace_dir) if trace_dir else None
    inflight = dict.fromkeys(registry.profiles, 0)

    app = FastAPI(title="open-system-one", docs_url=None, redoc_url=None)
    app.add_middleware(RequestIdMiddleware)

    @app.exception_handler(AuthError)
    async def _auth_error(request: Request, exc: AuthError):
        return _error(401, str(exc), {"www-authenticate": "Bearer"})

    async def require_key(request: Request) -> None:
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise AuthError("Missing bearer API key")
        if not any(hmac.compare_digest(token.strip(), key) for key in api_keys):
            raise AuthError("Invalid API key")

    def client_for(name: str) -> ModelClient:
        if name not in clients:
            clients[name] = make_client(registry.profiles[name])
        return clients[name]

    @app.get("/v1/models", dependencies=[Depends(require_key)])
    async def list_models():
        models = [
            {"name": p.name, "description": p.description, "release_date": p.release_date}
            for p in registry.profiles.values()
        ]
        for alias, target in registry.aliases.items():
            p = registry.profiles[target]
            models.append(
                {
                    "name": alias,
                    "description": f"Alias for {target}. {p.description}".strip(),
                    "release_date": p.release_date,
                }
            )
        return {"models": models}

    @app.post("/v1/systemone", dependencies=[Depends(require_key)])
    async def systemone(body: SystemOneRequest, request: Request):
        request_id = request.state.request_id
        profile = registry.resolve(body.model)
        if profile is None:
            detail = [
                {
                    "type": "value_error",
                    "loc": ["body", "model"],
                    "msg": f"Unknown model '{body.model}'. Use GET /v1/models to list models.",
                    "input": body.model,
                }
            ]
            return JSONResponse({"detail": detail}, status_code=422)

        if inflight[profile.name] >= profile.max_inflight_requests:
            return _error(429, "Too many concurrent requests for this model", {"retry-after": "1"})

        prompt = prompts[profile.name]
        timer = Timer()
        trace: dict[str, Any] = {
            "request_id": request_id,
            "time": time.time(),
            "model": body.model,
            "profile": profile.name,
            "backend": profile.backend,
            "model_id": profile.model_id,
            "prompt": prompt.ref,
            "prompt_hash": prompt.content_hash,
            "strategy": profile.strategy,
            "request": body.model_dump(mode="json", exclude_unset=True),
        }

        def finish(status: int, payload: dict[str, Any] | None, error: str | None = None):
            timer.add("total", time.perf_counter() - request.state.received_at)
            trace.update(status=status, error=error, timings_ms=timer.spans, response=payload)
            task = BackgroundTask(tracer.write, trace) if tracer else None
            return task, {"server-timing": timer.server_timing()}

        inflight[profile.name] += 1
        try:
            result = await answer_request(
                body, profile, prompt, client_for(profile.name), timer, trace
            )
        except LimitError as e:
            task, headers = finish(422, None, str(e))
            return JSONResponse({"detail": e.errors}, 422, headers=headers, background=task)
        except BackendOverloaded as e:
            task, headers = finish(529, None, str(e))
            headers["retry-after"] = str(max(1, round(e.retry_after)))
            return _error(529, "Model backend is overloaded; retry later", headers, task)
        except (ProtocolError, InvariantError, BackendError) as e:
            log.warning("request %s failed: %s", request_id, e)
            task, headers = finish(500, None, f"{type(e).__name__}: {e}")
            return _error(500, f"{type(e).__name__}: {e}", headers, task)
        except Exception as e:  # never leak a traceback; keep the request id on the response
            log.exception("request %s failed", request_id)
            task, headers = finish(500, None, f"{type(e).__name__}: {e}")
            return _error(500, "Internal server error", headers, task)
        finally:
            inflight[profile.name] -= 1

        task, headers = finish(200, result)
        return JSONResponse(result, headers=headers, background=task)

    return app
