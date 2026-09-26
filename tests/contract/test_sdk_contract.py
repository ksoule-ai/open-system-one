"""The official typesafe-sdk against our server (real HTTP on a local port, fake model backends)."""

import socket
import threading
import time

import pytest
import uvicorn
from typesafe_sdk import (
    RetryPolicy,
    TypeSafeAuthenticationError,
    TypeSafeClient,
    TypeSafeInternalServerError,
    TypeSafeUnprocessableEntityError,
)

from open_system_one.config import Profile, Registry
from open_system_one.server import create_app
from tests.fakes import FakeClient

KEY = "test-key"
QUESTIONS = {
    "is_urgent": {
        "type": "noul",
        "instructions": "Does this convey urgency?",
        "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"},
    },
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": "Payments", "technical": "Bugs", "sales": None},
    },
    "frustration": {
        "type": "score",
        "instructions": "How frustrated is the customer?",
        "criteria": ["Calm", "Frustrated", "Very angry"],
    },
}


def profile(name: str, **kw) -> Profile:
    base = {
        "backend": "hf_endpoint",
        "model_id": "fake",
        "release_date": "2026-09-26",
        "prompt": "default@1",
        "description": f"{name} (fake backend)",
        "max_options": 3,
        "top_logprobs": 20,
    }
    return Profile(name=name, **{**base, **kw})


@pytest.fixture(scope="module")
def server():
    clients = {
        "oso-fake": FakeClient(),
        "oso-overloaded": FakeClient(overloaded=True),
        "oso-nolabels": FakeClient(top={"The": 0.9, "Answer": 0.1}),
    }
    registry = Registry(
        profiles={name: profile(name) for name in clients},
        aliases={"oso-latest": "oso-fake"},
    )
    app = create_app(registry, "configs/prompts", [KEY], trace_dir=None, clients=clients)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not srv.started:
        assert time.monotonic() < deadline, "server did not start"
        time.sleep(0.02)
    yield f"http://127.0.0.1:{port}", clients
    srv.should_exit = True
    thread.join(5)


def sdk(url, key=KEY, **kw):
    return TypeSafeClient(base_url=url, api_key=key, retry=RetryPolicy(max_retries=0), **kw)


def test_answers_parse_into_sdk_models(server):
    url, clients = server
    resp = sdk(url).system_one("Help! Payouts failing for 3 days.", QUESTIONS, model="oso-latest")
    assert resp.model == "oso-fake"  # aliases resolve to their profile
    assert resp.request_id.startswith("req_")
    assert list(resp.answers) == list(QUESTIONS)
    assert resp.answers["is_urgent"].noul == pytest.approx(0.65 / 0.9)
    dept = resp.answers["department"]
    assert dept.choice == "billing" and list(dept.probabilities) == [
        "billing",
        "technical",
        "sales",
    ]
    assert sum(dept.probabilities.values()) == pytest.approx(1.0)
    score = resp.answers["frustration"]
    assert score.legend == {0: "Calm", 1: "Frustrated", 2: "Very angry"}
    assert score.score == pytest.approx(sum(k * p for k, p in score.probabilities.items()))
    assert resp.usage.input_tokens == 300 and resp.usage.output_tokens == 3
    assert "total;dur=" in resp.raw_http_response.headers["server-timing"]
    # Question keys never reach the model.
    sent = str(clients["oso-fake"].calls[-3:])
    assert "is_urgent" not in sent and "department" not in sent and "frustration" not in sent


def test_identical_questions_are_deduplicated(server):
    url, clients = server
    before = len(clients["oso-fake"].calls)
    q = QUESTIONS["is_urgent"]
    resp = sdk(url).system_one("state", {"a": q, "b": q}, model="oso-fake")
    assert resp.answers["a"] == resp.answers["b"]
    assert len(clients["oso-fake"].calls) == before + 1


def test_models_list(server):
    url, _ = server
    names = [m.name for m in sdk(url).models.list().models]
    assert names == ["oso-fake", "oso-overloaded", "oso-nolabels", "oso-latest"]


def test_bad_key_is_401(server):
    url, _ = server
    with pytest.raises(TypeSafeAuthenticationError) as e:
        sdk(url, key="wrong").system_one("s", QUESTIONS, model="oso-fake")
    assert e.value.request_id.startswith("req_")
    with pytest.raises(TypeSafeAuthenticationError):
        sdk(url, key="wrong").models.list()


def test_unknown_model_is_422(server):
    url, _ = server
    with pytest.raises(TypeSafeUnprocessableEntityError) as e:
        sdk(url).system_one("s", QUESTIONS, model="jev-latest")
    assert e.value.body["detail"][0]["loc"] == ["body", "model"]


def test_schema_violation_is_fastapi_422(server):
    url, _ = server
    bad = {"q": {"type": "choice", "instructions": "no criteria"}}
    with pytest.raises(TypeSafeUnprocessableEntityError) as e:
        # The SDK validates questions client-side; extra_body replaces them after that check.
        sdk(url).system_one("s", QUESTIONS, model="oso-fake", extra_body={"questions": bad})
    detail = e.value.body["detail"]
    assert detail and {"loc", "msg", "type"} <= set(detail[0])


def test_too_many_options_is_422(server):
    url, _ = server
    q = {"type": "choice", "criteria": {k: None for k in "wxyz"}}
    with pytest.raises(TypeSafeUnprocessableEntityError) as e:
        sdk(url).system_one("s", {"q": q}, model="oso-fake")
    [err] = e.value.body["detail"]
    assert err["loc"] == ["body", "questions", "q", "choice", "criteria"]
    assert err["ctx"]["max_length"] == 3


def test_backend_overload_is_529_with_retry_after(server):
    url, _ = server
    with pytest.raises(TypeSafeInternalServerError) as e:
        sdk(url).system_one("s", QUESTIONS, model="oso-overloaded")
    assert e.value.status == 529 and e.value.headers["retry-after"] == "2"


def test_no_labels_is_500_not_a_guess(server):
    url, _ = server
    with pytest.raises(TypeSafeInternalServerError) as e:
        sdk(url).system_one("s", QUESTIONS, model="oso-nolabels")
    assert e.value.status == 500 and "no answer label" in e.value.body["detail"]
    assert e.value.request_id.startswith("req_")
