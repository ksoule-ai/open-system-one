# Phase 0 findings (2026-09-26)

Ollama spikes are deferred. Raw outputs are in `runs/spikes/` (gitignored). Scripts are in this
directory and rerun with `uv run --env-file .env python spikes/<script>.py`.

## 1. Schema snapshot
- Downloaded `https://api.typesafe.ai/openapi.json` (TypeSafe API `version: 0.2.0`) as
  `typesafe-openapi-2026-09-26.json`. **Not yet in `schemas/`**: the `Edit(./schemas/**)` deny rule
  in `.claude/settings.json` also blocks Claude from creating the file there, so it needs a manual copy.
- Models generated with `datamodel-codegen` (pydantic v2) are **semantically identical** to the
  models bundled in `typesafe-sdk==0.7.1` (`typesafe_sdk/_schemas/models.py`): the JSON schemas of all
  16 models match.
- Differences between the snapshot and `api-schema.md`:
  - `instructions` is optional and nullable for all three question types; noul `criteria.true` and
    `criteria.false` may be null.
  - Score `criteria` has `minItems: 1` but no upper limit. The "2–10 levels" and "up to 255 options"
    limits are not in the spec (they may be enforced server-side; to verify against Jev).
  - 401 / 429 / 529 are not declared in the spec (only 200 and 422).

## 2. Mellea logprobs (`OpenAIBackend`, non-streaming)
- `logprobs` and `top_logprobs` pass straight through `model_options`. Mellea filters options against
  `openai`'s `chat.completions.create` signature, and both are real parameters.
- `mot.raw.response` holds the full `ChatCompletion.model_dump()` with `choices[0].logprobs`.
- `ModelOption.MAX_NEW_TOKENS` is sent as `max_completion_tokens`. OpenRouter and vLLM accept it.
- **Bug that affects us: Mellea always sends `"tools": null`.** OpenRouter treats that as a tool-use
  request and drops every provider without tool support (for Llama 3.3 70B: SambaNova, Parasail,
  Cloudflare, Vertex) → 404 "No endpoints found". The same request without the key succeeds. We need a
  thin, documented workaround (e.g. strip `tools: null` in a request hook on the OpenAI client Mellea
  builds, which accepts `http_client=` via `**kwargs`) and an upstream issue.
- `default_extra_body={"provider": {...}}` is the way to pass OpenRouter routing options.

## 3. Backend capability matrix
### OpenRouter (`spikes/openrouter_matrix.py`, provider pinned via `order` + `allow_fallbacks: false`)
- **OpenRouter returns at most one logprob position**, even when two tokens are generated. For many
  providers that one position is the **last** token (e.g. `<|eot_id|>`), not the answer.
  → **`max_tokens` must be 1 on OpenRouter**, and the answer is always read at the first generated
  token. The "first position with labels within `max_tokens`" mode can't work there.
- **`require_parameters: true` does not stop silent drops.** Cloudflare advertises `top_logprobs` for
  Llama 3.3 70B but returns no logprobs. The endpoints API's `supported_parameters` is not reliable.
  → Profiles must **pin providers** (`order` + `allow_fallbacks: false`) to ones verified here, and the
  server must still treat missing logprobs as a 500.
- With `max_tokens: 1`, `top_logprobs: 20`, usable (label mass ≈ 1 on an easy question):

  | Model | Verified providers | Broken / unavailable |
  | --- | --- | --- |
  | meta-llama/llama-3.3-70b-instruct | novita, akashml, parasail, sambanova, coreweave | cloudflare (no logprobs); deepinfra, groq, together, vertex (not advertised) |
  | meta-llama/llama-3.1-8b-instruct | novita, coreweave | cloudflare, deepinfra, groq |
  | google/gemma-3-27b-it | parasail | others not advertised |
  | qwen/qwen3-235b-a22b-2507 | gmicloud, parasail, streamlake, google-vertex/us-south1 | alibaba (400) |
  | qwen/qwen3-30b-a3b-instruct-2507 | streamlake | alibaba (400) |
  | mistralai/mistral-small-3.2-24b-instruct | parasail (429 upstream during test; worked earlier) | — |
  | openai/gpt-4o-mini | openai, azure | — |

- `top_logprobs` cap: 20 (OpenAI rejects 21 with 400). `max_options` ≤ 20 holds.
- **Temperature:** Novita and GMICloud apply temperature to logprobs (T=0.7 and T=2 change them), but
  T=0 and T=1 give identical raw values. SambaNova and OpenAI ignore temperature for logprobs.
  → Use T=0 (or 1): raw distribution on every provider tested.
- **Prefix caching** (~2.4k-token prefix, 3 questions): GMICloud, OpenAI, and CoreWeave report
  `cached_tokens` from the second call on (≈95% of the prompt). Novita and SambaNova report 0.
- Upstream 429s ("temporarily rate-limited upstream") are common; they should map to our 529 with
  `retry-after`.
- Latency: ~0.2–1.1 s per single-token call.

### Granite 4.0 Micro on OpenRouter (added 2026-09-26)
- `ibm-granite/granite-4.0-h-micro` (Granite 4.0 Micro, 3B) is served only by **Cloudflare**, which
  advertises `logprobs` / `top_logprobs` but not `tools`. For this model Cloudflare **does** return
  logprobs (20 top, answer at the first token, label mass ≈ 1), unlike Llama 3.3 70B on Cloudflare.
- Mellea's `"tools": null` doesn't get Cloudflare filtered out here, so the profile works through
  the normal Mellea path without the postponed workaround.
- No `cached_tokens` reported, so `warm_fanout` gains nothing: 8 requests × 3 questions, p50 929 ms
  (warm_fanout) vs **405 ms (fanout)**. Profile `oso-granite-micro-cf` uses `fanout`.
- Granite Micro on the payouts example: noul 1.0, choice `technical` (0.9999), score 1.97; the HF
  Granite Switch 3B gave score 1.0 on the same request.

### HF Inference Endpoint (vLLM, `ibm-granite/granite-switch-4.1-3b-preview`)
- `HF_ENDPOINT_URL` already ends in `/v1`; `/v1/models` works with `HF_TOKEN`.
- Full per-position logprobs (all generated tokens), 20 top logprobs, `>20` → 400
  ("max allowed: 20", vLLM `max_logprobs` default; raisable server-side).
- Logprobs are raw (identical at T=0, 1, 2).
- Reports `cached_tokens` (vLLM automatic prefix caching is on).
- ~0.1–0.5 s per call.

## 4. Answer position
- With an explicit "Answer with A, B, or C" line, all tested models put the label in the first token
  (label mass ≈ 1).
- **Without that line**, Granite and gpt-4o-mini start with "The" (label mass ≈ 0.01 and ≈ 0.15);
  Qwen3-235B still answers `B`. The answer-format instruction has to be part of the prompt config.
- **Assistant prefill** (`Answer:` as a final assistant message):
  - vLLM honors it with `extra_body={"continue_final_message": true, "add_generation_prompt": false}`;
    the answer token then carries a leading space (`" B"`), so token matching must strip whitespace.
  - OpenRouter doesn't reliably continue it: Qwen/GMICloud split mass between `B` and a new `Answer`;
    gpt-4o-mini treats it as a prior turn. → Keep the primer in the user message by default and make
    prefill a per-backend config option.
- Label variants seen in the top-k: `A`/` A`, `Yes`/` Yes`/`yes`/`YES`. Summing variants matters.

## 5. SDK contract (`spikes/sdk_contract_stub.py`)
- Paths `/v1/systemone`, `/v1/models`. Sends `Authorization: Bearer`, `Accept` / `Content-Type:
  application/json`, `User-Agent` and `X-TypeSafe-SDK: typesafe-sdk/0.7.1`, `X-TypeSafe-Runtime`,
  and `X-TypeSafe-Retry-Count` on retries.
- Reads `x-typesafe-request-id` (`resp.request_id` raises if absent) and `retry-after` /
  `retry-after-ms`.
- Error classes: 401 → `TypeSafeAuthenticationError`, 422 → `TypeSafeUnprocessableEntityError`
  (message built from FastAPI's `detail[].loc/msg`), 429 → `TypeSafeRateLimitError`, 5xx →
  `TypeSafeInternalServerError`.
- **Retries by default: 408, 429, and every 5xx**, 2 retries, 30 s budget. So our 500 for "no usable
  logprobs" gets retried twice. That's acceptable (it can be transient on OpenRouter) but costs
  latency and money on permanent failures.
- Default timeout **10 s** per HTTP operation; slow backends or big fan-outs must stay under it.
- Response models are `strict=True` and `extra="ignore"`: extra fields are tolerated, ints are
  accepted for floats, and score `legend` / `probabilities` string keys become ints.

## 6. Jev via OpenRouter (`spikes/jev_openrouter.py`)
- The official SDK with `base_url="https://openrouter.ai/api"` and `model="jev-1.13"` works. Returned
  model id: **`typesafe/jev-1.13-20260917`** (use it in the cache key).
- Responses validate against the snapshot's `SystemOneResponse`.
- **OpenRouter differences from TypeSafe:** `usage` carries an extra `cost` field (≈ $1.2e-5 per
  single-noul request); no `x-typesafe-request-id` header (OpenRouter sends `x-generation-id` and
  `x-provider-name`), so the baseline client must not read `resp.request_id`.
- Jev rounds probabilities to 2 decimals and emits exact zeros as integers (`0`). Its `probabilities`
  key order isn't `criteria` order.
- Matches the published `jev_reference` values within 0.04; spread across 3 repeats ≤ 0.04.
- **Confidence formula** clip((N·max p − 1)/(N − 1), 0, 1) matches Jev's `confidence` within ±0.01
  (rounding) on the choice and score questions tested.
- Usage: ~280 input / 20 output tokens for a single noul; 0.14–0.33 s per call.

## Environment notes
- In the Claude Code sandbox: set `UV_CACHE_DIR=$TMPDIR/uv-cache`; `.env` and `.env.*` (including
  `.env.example`) are read-denied, so anything using keys runs outside the sandbox. Local port
  binding is allowed (`sandbox.network.allowLocalBinding: true`), so the server and contract tests
  run inside it.
- `.env` has `MODEL_ID` and `HF_TOKEN_INFERENCE`, which aren't in `.env.example`; `OSO_API_KEY` isn't
  set yet.

# Later spikes (2026-10-02 to 2026-10-06)

Write-ups with all numbers are in [`results/`](../results/README.md); the method is documented in
[`docs/logprob-decisions.md`](../docs/logprob-decisions.md). This section records what each new script is
for and the facts we learned about the stack.

| Script | What it does |
| --- | --- |
| `ragtruth_hallucination.py` | Hallucination adapter on the 100 RAGTruth dev samples (inputs from RAGTruth's source records); `make_backend()` is the shared Mellea backend for the adapter spikes |
| `hallucination_adapter_dev.py` | Hallucination adapter on HoVer and NLI4CT dev; `--user-message` |
| `factuality_adapter_dev.py` | Factuality adapter on RAGTruth, HoVer, NLI4CT dev; `--default-formatting` |
| `factuality_adapter_suite.py` | Any of factuality / hallucination / guardian-core on suite or dev rows, writing kit-format results scored by the kit; reads P("yes") from logprobs; resumable, sharded |
| `nli_adapters_dev.py`, `nli_adapters_analyze.py` | Answerability and factuality on ANLI and ContractNLI with P("yes"); threshold mapping onto three classes |
| `guardian_raw_dev.py` | Guardian groundedness sent as raw text via Mellea's completions path, to change the activation sequence |
| `rerun_sample.py` | Re-send one dev sample to several engines and the adapter, saving request, responses, traces and gateway log ids |

Facts about the stack:

- **LiteLLM gateway.** All model calls since 2026-10-02 go through a local LiteLLM gateway
  (`LITELLM_BASE_URL`, `LITELLM_API_INFERENCE_KEY`; config in the separate `kate-litellm` repo). It passes
  `logprobs` / `top_logprobs` (256 on the HF endpoint), `documents`, `chat_template_kwargs` and
  `structured_outputs` through to vLLM, and OpenRouter's `provider` routing through to OpenRouter. Jev is
  reached through a pass-through route `POST /v1/systemone` → `https://openrouter.ai/api/v1/systemone`;
  per-key pass-through permissions are a LiteLLM premium feature, so that route is used with the master
  key. Log lookup: `/spend/logs?request_id=…` and the paginated `/spend/logs/ui`.
- **Mellea 0.8.0 talks to OpenAI-compatible servers with chat requests** (`chat.completions.create`),
  including adapter calls; the completions endpoint is used only by `generate_from_raw`. So the model's
  chat template, applied by vLLM, writes the final prompt (system turn with documents, adapter
  activation tokens).
- **Granite chat templates** take documents as a `documents` field and write them into the system turn
  with fixed wording ("You are a helpful assistant with access to the following documents…"). A message
  with `role: document` is ignored. OpenRouter (Cloudflare) drops the `documents` field.
- **The vLLM endpoint has no tool calling enabled**: `tool_choice: auto` and `required` both need
  `--enable-auto-tool-choice --tool-call-parser …` in the container args.
- **Adapter activation** is driven by `chat_template_kwargs.adapter_name` in the chat template
  (`chat_template.jinja` in the model repo): LoRA adapters put their token at the start of the prompt;
  aLoRA adapters replace the first token of their invocation text (`<guardian>` → `<|guardian-core|>` +
  `guardian>`). Mellea downloads adapter configs from the model repo, which needs `huggingface_hub`
  (`uv run --with huggingface_hub`; it is Mellea's `switch` extra, not a project dependency).
- **The HF endpoint scales to zero and is sometimes paused.** A paused endpoint answers 400 "The endpoint
  is paused"; a waking one answers 503 for about 3 minutes. Long batch jobs should wait for a 200 first and
  retry errored rows.
