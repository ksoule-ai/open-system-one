import pytest

from open_system_one.config import load_registry
from open_system_one.prompts.config import load_prompt
from open_system_one.prompts.render import render_question
from open_system_one.schema.models import SystemOneRequest


@pytest.fixture(scope="module")
def cfg():
    return load_prompt("configs/prompts", "default@1")


def questions(qs, state="Help! My payouts have been failing for 3 days."):
    req = SystemOneRequest.model_validate({"state": state, "model": "m", "questions": qs})
    return {k: q.root for k, q in req.questions.items()}


def test_prompt_version_must_match(cfg):
    assert cfg.ref == "default@1" and len(cfg.content_hash) == 12
    with pytest.raises(FileNotFoundError):
        load_prompt("configs/prompts", "nonexistent@1")


def test_question_keys_never_reach_the_prompt(cfg):
    qs = questions(
        {
            "secret_key_1": {"type": "noul", "instructions": "Urgent?"},
            "secret_key_2": {"type": "choice", "criteria": {"billing": None, "tech": "Bugs"}},
            "secret_key_3": {"type": "score", "criteria": ["low", "high"]},
        }
    )
    for q in qs.values():
        text = str(render_question("s", q, cfg).messages)
        assert "secret_key" not in text


def test_shared_prefix_across_questions(cfg):
    qs = questions(
        {
            "a": {"type": "noul", "instructions": "Urgent?"},
            "b": {"type": "choice", "instructions": "Team?", "criteria": {"billing": None}},
        },
        state={"subject": "Payouts", "tags": ["urgent"]},
    )
    rendered = [
        render_question({"subject": "Payouts", "tags": ["urgent"]}, q, cfg) for q in qs.values()
    ]
    system_a, user_a = rendered[0].messages
    system_b, user_b = rendered[1].messages
    assert system_a == system_b
    prefix = user_a["content"].split("Question:")[0]
    assert user_b["content"].startswith(prefix) and "subject: Payouts" in prefix


def test_choice_rendering_and_label_collision(cfg):
    qs = questions(
        {
            "dept": {"type": "choice", "criteria": {"billing": "Payments", "sales": None}},
            "letters": {"type": "choice", "criteria": {"A": None, "b": None}},
        }
    )
    dept = render_question("s", qs["dept"], cfg)
    assert dept.labels == ["A", "B"] and dept.options == ["billing", "sales"]
    assert "A. billing: Payments\nB. sales\n" in dept.messages[-1]["content"]
    letters = render_question("s", qs["letters"], cfg)
    assert letters.labels == ["1", "2"]  # keys look like letter labels → numbers


def test_score_and_noul_rendering(cfg):
    qs = questions(
        {
            "s": {"type": "score", "criteria": ["Calm", {"level": "angry"}]},
            "n": {"type": "noul", "instructions": "Urgent?", "criteria": {"true": "Deadline"}},
        }
    )
    score = render_question("x", qs["s"], cfg)
    assert (
        score.labels == ["0", "1"] and "0. Calm\n1. level: angry\n" in score.messages[-1]["content"]
    )
    noul = render_question("x", qs["n"], cfg)
    assert "Answer Yes if: Deadline" in noul.messages[-1]["content"]
    assert "Answer No if" not in noul.messages[-1]["content"]


def test_models_yaml_loads(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT_URL", "https://example.test/v1")
    reg = load_registry("configs/models.yaml")
    assert reg.resolve("oso-latest").name == "oso-granite-3b"
    assert reg.profiles["oso-granite-3b"].base_url == "https://example.test/v1"
    assert reg.resolve("nope") is None


def test_extended_labels_cover_large_choices():
    v1 = load_prompt("configs/prompts", "default@1")
    v2 = load_prompt("configs/prompts", "default@2")
    small = questions({"q": {"type": "choice", "criteria": {"billing": None, "sales": None}}})["q"]
    # Up to 20 options, v2 renders exactly like v1 (letters).
    assert render_question("s", small, v2).messages == render_question("s", small, v1).messages
    big = questions(
        {"q": {"type": "choice", "criteria": {f"option_{i}": f"intent {i}" for i in range(151)}}}
    )["q"]
    rendered = render_question("s", big, v2)
    assert rendered.labels == [str(i) for i in range(1, 152)]
    assert "151. option_150: intent 150" in rendered.messages[-1]["content"]
    with pytest.raises(ValueError, match="at most 20 labels"):
        render_question("s", big, v1)


def test_request_top_logprobs():
    from open_system_one.config import Profile
    from open_system_one.engine import request_top_logprobs

    p = Profile(
        name="p",
        backend="hf_endpoint",
        model_id="m",
        release_date="2026-09-26",
        prompt="default@2",
        top_logprobs=256,
        max_options=200,
    )
    assert request_top_logprobs(p, 3) == 20
    assert request_top_logprobs(p, 77) == 154
    assert request_top_logprobs(p, 151) == 256


def test_state_as_document_leaves_the_user_message_to_the_question():
    v3 = load_prompt("configs/prompts", "default@3")
    v4 = load_prompt("configs/prompts", "default@4")
    state = {"subject": "Payouts", "tags": ["urgent"]}
    q = questions({"k": {"type": "noul", "instructions": "Urgent?"}}, state=state)["k"]
    in_user, as_doc = render_question(state, q, v3), render_question(state, q, v4)
    assert in_user.documents == [] and "subject: Payouts" in in_user.messages[-1]["content"]
    assert as_doc.documents == [{"doc_id": "state", "text": "subject: Payouts\ntags:\n- urgent"}]
    assert "Payouts" not in str(as_doc.messages)
    assert [m["role"] for m in as_doc.messages] == ["user"]  # the template supplies the system turn
    assert as_doc.messages[-1]["content"].startswith("Question: Urgent?")
    assert as_doc.labels == in_user.labels
