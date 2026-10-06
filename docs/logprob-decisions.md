# Answering decision requests from log-probabilities

This guide explains how open-system-one turns any chat model that returns token log-probabilities into a
decision engine with TypeSafe's Jev API: `noul` (yes/no), `choice` and `score` questions, each answered
with a probability for every option. It covers the method, three ways to use it, how to configure prompts
and models, how to read a yes/no probability from a Granite Switch adapter, and the pitfalls we hit.

Results from using it are in [`results/`](../results/README.md).

## The method in one paragraph

Each question is rendered as a chat prompt in which every option sits under a short label that is a
single token for the model: `Yes` / `No` for a `noul`, `A`, `B`, `C`… (or `1`…`200` for long lists) for a
`choice`, and `0`…`9` for a `score`. The model generates **one token** with its top log-probabilities. The
probability of each option is the probability of its label token, divided by the total over all labels:

```
P(option) = P(label token) / Σ P(every label token)
```

The answer is then built in Jev's shape: `noul` = P(Yes); `choice` = the most likely key plus every key's
probability; `score` = the expected level (Σ level × P(level)) plus every level's probability. No
fine-tuning and no output parsing: the decision is read straight off the next-token distribution.

```
request (state + questions)
  └─ per question ─► render (prompt config) ─► 1 token + top logprobs (Mellea) ─► read labels ─► Jev answer
```

The same `state` is the shared prefix of every question in a request, so the backend's prefix cache is
reused across questions (see `strategies/`).

## What a backend must support

| Requirement | Why | Notes |
| --- | --- | --- |
| `logprobs` + `top_logprobs` on chat completions, non-streaming | the label probabilities come from the answer token's top-k | Mellea 0.8.0's streaming merge drops logprobs; we call non-streaming |
| `top_logprobs` ≥ the number of options | every label must be readable | OpenAI and OpenRouter cap at 20; vLLM's default is 20, raised with `--max-logprobs` (our endpoint: 256) |
| the answer at the first generated token | we read one position | OpenRouter returns at most one logprob position; keep `max_tokens: 1` |
| temperature 0 (or 1) | raw log-probabilities | some providers apply temperature to logprobs at other values |
| no reasoning before the answer | reasoning models emit thinking tokens first | disable reasoning or don't use those models |

On OpenRouter, pin the provider (`provider.order` + `allow_fallbacks: false`) to one verified to return
logprobs: `require_parameters` does not stop silent drops. Phase 0 results per provider are in
[`spikes/FINDINGS.md`](../spikes/FINDINGS.md).

## Three ways to use it

### 1. Run the server, call it like Jev

```bash
uv run --env-file .env open-system-one serve --port 8080
```

Any Jev client works, including TypeSafe's official SDK:

```bash
TYPESAFE_BASE_URL=http://localhost:8080 TYPESAFE_API_KEY=$OSO_API_KEY python your_app.py
```

or plain HTTP:

```bash
curl -s localhost:8080/v1/systemone -H "Authorization: Bearer $OSO_API_KEY" -H "Content-Type: application/json" -d '{
  "model": "oso-latest",
  "state": "I returned the jacket two weeks ago and still have no refund.",
  "questions": {
    "urgent": {"type": "noul", "instructions": "The customer needs a reply today."},
    "team": {"type": "choice", "criteria": {"billing": "Refunds and charges", "shipping": null}}
  }
}'
```

`model` names a profile or alias in [`configs/models.yaml`](../configs/models.yaml). Each request writes a
trace to `runs/traces/` with the rendered messages, the raw response with logprobs, the label reading
(label mass, missing labels, normalized probabilities) and timings.

### 2. In-process, without the server

[`examples/decide.py`](../examples/decide.py) answers a whole request with the server's engine and prints
what the protocol read for each question:

```bash
uv run --env-file .env python examples/decide.py --model oso-granite-micro-cf-v3-litellm
uv run --env-file .env python examples/decide.py --model oso-latest --request my_request.json
```

### 3. Step by step, inside your own code

[`examples/protocol_steps.py`](../examples/protocol_steps.py) runs the four steps explicitly:

```python
rendered = render_question(state, question, prompt)                 # messages, labels, options
result = await make_client(profile).complete(rendered.messages, prefill=rendered.prefill,
    top_logprobs=request_top_logprobs(profile, len(rendered.labels)), documents=rendered.documents or None)
reading = read_labels(positions_from_response(result.raw), rendered.labels,
                      prompt.matching, prompt.answer.position)       # label mass, raw, normalized
answer = build_answer(question.type, rendered.options, rendered.labels, reading)
```

`make_client` is a thin wrapper over Mellea's `OpenAIBackend` (all model calls go through Mellea, per
[`.claude/rules/mellea.md`](../.claude/rules/mellea.md)). For a `score` question, pass the legend (level →
description) as the last argument of `build_answer`, as `engine.py` does.

## Prompts: how a question becomes labels

Prompts live in versioned configs, [`configs/prompts/<name>@<version>.yaml`](../configs/prompts/), never
in code. Every response trace records the prompt's name, version and content hash. The knobs:

| Knob | What it controls | `default@3` |
| --- | --- | --- |
| `system` | system text | "You are a precise classifier…" |
| `user_layout` | order of blocks in the user message | `[state, question]` (state first: shared prefix) |
| `state.template`, `state.format` | how the state is shown; objects and arrays as YAML or JSON | `<state>…</state>`, YAML |
| `state.placement` | `user` (a block of the user message) or `document` (the chat template's `documents` field) | `user` |
| `questions.<type>.template` | Jinja template per question type | instructions, labeled options, "Reply with only the label…" |
| `questions.choice.labels` / `collision_labels` / `extended_labels` | label schemes, tried in order; collision labels are used when keys look like labels (keys `A`–`E`) | `A`–`T`, then `1`–`20`, then `1`–`256` |
| option line | which of label, key and description appear | the description when there is one, else the key |
| `answer.primer`, `answer.primer_mode` | text right before the answer token; `assistant` = prefill (vLLM only) | none |
| `answer.position` | `first_token`, or the first position whose top-k contains labels | `first_token` |
| `matching` | strip whitespace, case-insensitive, missing labels count as 0 (`zero`) or the smallest returned probability (`floor`) | strip, case-insensitive, `zero` |

Question keys are never sent to the model. Criteria are rendered as given, in their order.

`default@4` sends the state as a chat-template document instead of a user block. On Granite Switch it
scored worse on the dev set (mean chance-corrected skill 0.255 against 0.303 for `default@3`); see
[`results/dev-set.md`](../results/dev-set.md).

## Models: adding a profile

A profile in `configs/models.yaml`:

```yaml
  my-profile:
    backend: hf_endpoint              # hf_endpoint (vLLM) | openrouter
    base_url: ${LITELLM_BASE_URL}     # any OpenAI-compatible /v1 base; ${VAR} is read from the environment
    api_key_env: LITELLM_API_INFERENCE_KEY   # name of the env var holding the key, never the key
    model_id: hf/ibm-granite/granite-switch-4.1-3b-preview
    prompt: default@3
    strategy: warm_fanout             # sequential | fanout | warm_fanout
    top_logprobs: 256                 # the most the backend allows per call
    base_top_logprobs: 20             # what small questions ask for (large ones ask for 2 × options)
    max_options: 200                  # ≤ top_logprobs; requests above it get a 422
    max_tokens: 1
    temperature: 0
    concurrency: 64                   # simultaneous model calls per server
    max_inflight_requests: 256        # requests above it get a 429
    timeout: 60
    release_date: "2026-10-02"
    description: ...
    extra_body:                       # merged into every call (e.g. OpenRouter provider pinning)
      provider: {order: [cloudflare], allow_fallbacks: false}
```

Profiles in use (October 2026), all through the local LiteLLM gateway (`LITELLM_BASE_URL`,
`LITELLM_API_INFERENCE_KEY`):

| Profile | Model | Options per choice | Prompt |
| --- | --- | --: | --- |
| `oso-granite-3b-wide-litellm` | Granite Switch 4.1 3B preview on our HF endpoint (vLLM) | 200 | `default@3` |
| `oso-granite-3b-v3-litellm` | same | 20 | `default@3` |
| `oso-granite-3b-wide-doc-litellm` | same | 200 | `default@4` |
| `oso-granite-micro-cf-v3-litellm` | Granite 4.0 Micro via OpenRouter, pinned to Cloudflare | 20 | `default@3` |
| `oso-granite-micro-cf-litellm` | same | 20 | `default@1` |

The older direct profiles (`oso-granite-3b`, `oso-granite-3b-wide`, `oso-granite-micro-cf`) need
`HF_ENDPOINT_URL` / `HF_TOKEN` / `OPENROUTER_API_KEY` in `.env`.

Before trusting a new model, check that it returns usable logprobs: send one easy question with
`logprobs: true, top_logprobs: 20, max_tokens: 1` and look at the top candidates; then run
`examples/decide.py` and check that **label mass** is close to 1.

## Reading a yes/no probability from a Granite Switch adapter

Granite Switch embeds adapters (hallucination detection, factuality detection, guardian, answerability,
requirement check, …) that answer with JSON such as `{"score": "yes"}`. The same idea gives a probability:
request logprobs on the adapter call and read P(yes) / (P(yes) + P(no)) at the answer token. Mellea's
adapter functions (`guardian.guardian_check`, `rag.flag_hallucinated_content`, …) return only the parsed
answer, so we call the same path one level down, `mfuncs.act` on an `Intrinsic`, with `logprobs` and
`top_logprobs` in the model options, and read `mot.raw.response`.
`guardian.guardian_check` already returns P("yes") as its score. Code:
[`spikes/factuality_adapter_suite.py`](../spikes/factuality_adapter_suite.py) (`judge`),
[`spikes/nli_adapters_dev.py`](../spikes/nli_adapters_dev.py) (`label_probs`), and for raw-text requests
[`spikes/guardian_raw_dev.py`](../spikes/guardian_raw_dev.py).

Lessons from doing this:

- **Read the position where the model actually wrote the answer.** The first position whose top-k merely
  contains "yes" can be an earlier token (for example the `"score"` key), which gives false 1.0 scores.
- **Sum every variant of a label** (`yes`, `Yes`, `YES`, `"yes`): don't let one overwrite another.
- **Multi-token labels** (answerability's `answerable` / `unanswerable`) are decided at their first
  differing token (`answer` against `un`): sum the top-k tokens that are a prefix of only one label.
- **Adapters are activated by the chat template**, from `chat_template_kwargs.adapter_name`. aLoRA adapters
  swap the first token of their invocation text (`<guardian>` becomes `<|guardian-core|>guardian>`); LoRA
  adapters put their token at the very start of the prompt. The template is
  `chat_template.jinja` in the model's Hugging Face repo.
- **The hallucination adapter writes a verdict and an explanation per sentence** (slow: tens of seconds
  per long response under load). Its labels are usable; its explanations often describe the wrong
  sentence.

## Evaluating

- **Dev set first.** [`evals/decision-index-dev/`](../evals/decision-index-dev/README.md) has 34
  Decision Index benchmarks from data the scored suite does not use. Run it with
  [`scripts/decision_index_run.py`](../scripts/decision_index_run.py) (`split`, `launch`, `score --rows`).
- **Chance-correct.** Raw scores are not comparable across benchmarks or across differently balanced
  sets. [`scripts/dev_chance.py`](../scripts/dev_chance.py) measures chance on the dev rows (the kit's
  random engine over 20 seeds; RAGTruth = always answering "hallucinated"; ForecastBench against Brier
  0.25) and turns every dev run into skill = clip((score × coverage − chance) / (1 − chance), 0, 1).
- **Noise.** Two runs of the same model and prompt differed by up to about 0.02 per benchmark on the dev
  set (the vLLM endpoint is not fully deterministic at temperature 0). With 100 samples per benchmark,
  differences under about 0.05 are within sampling noise.
- **Then the suite**, once, with a fixed configuration: the Decision Index house rules forbid tuning on
  it and per-benchmark prompt switching. Rescore with the kit's newer editions without rerunning
  (`score --edition 0.2.1`).

## Pitfalls we hit

| Symptom | Cause | Fix |
| --- | --- | --- |
| OpenRouter returns 404 "No endpoints found" | Mellea sends `"tools": null`, which OpenRouter treats as a tool request and so excludes providers without tool support | pick a provider that accepts it (Cloudflare does for Granite 4.0 Micro); a workaround that strips the field is still pending |
| Logprobs silently missing on OpenRouter | provider ignores the parameter | pin a verified provider |
| Label mass well below 1 | the model wants to start with something else ("The", whitespace, reasoning) | end the question with "Reply with only the label…"; check `answer.position` |
| Choices over 20 options rejected | `top_logprobs` cap | a backend with a higher cap (vLLM `--max-logprobs`) and `extended_labels` |
| RAGTruth looks good on the dev set, poor on the suite | the dev set is balanced 50/50; the suite has 35% hallucinated | chance-correct against always answering "hallucinated" on each set |
| Adapter run fails with timeouts or truncated JSON | long outputs under load; malformed JSON at temperature 0 repeats on retry | longer timeout, lower concurrency; count the rest as wrong |
| The gateway log shows no documents in the messages | documents travel in the request's `documents` field and are written into the system turn by the chat template on the server | look at the request's `documents` field, or render the template locally |
