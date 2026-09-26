# Project: open-system-one

## Thesis
**You can build your own System One model with Mellea.** Take any chat LLM, expose a state and a typed
question to it through a configurable prompt, read the probability of each option's answer token, and
return a Jev-shaped answer. No fine-tuning, no special model: the decision layer is a Mellea program.

The deliverable is an HTTP API that **exactly mimics TypeSafe's Jev API** (`POST /v1/systemone`,
`GET /v1/models`), so existing Jev clients — including TypeSafe's official SDKs pointed at our base URL —
work unchanged. Behind it, any of three backends: an OpenRouter model, a custom Hugging Face Inference
Endpoint, or a local Ollama model.

**Probability protocol (the one estimator):** for each question, options are shown under single-token
labels; the model generates one answer token; P(option) = P(option's label token) / Σ P(all label
tokens).

Goals:
- Jev-compatible API: same request, response, errors, and headers.
- Easily configurable prompt structure — how state, instructions, and options are exposed — because
  prompt tuning is expected.
- A lightweight eval for prompt tuning and sanity checks, run against our endpoint and a cached Jev
  baseline with identical requests.
- End-to-end latency tracking, and batching strategies that maximize prefix-cache reuse.

Non-goals (MVP): fine-tuning, Granite Switch or adapters, calibration beyond the raw protocol, and the
Decision Index (a future goal: its kit's `http` engine targets `/v1/systemone`, so our server is a
drop-in once it's ready).

This project should stay neutral: prefer configurability and rich, un-aggregated data capture over baking
in any particular prompt or backend.

## Stack
- Python 3.11+ (Mellea 0.8 requires it), managed with uv (`.venv/`)
- **Mellea 0.8.0** — backend abstraction, prompt components, async calls. The API shifts between 0.x
  releases; see `.claude/rules/mellea.md`.
- **FastAPI + uvicorn** — the API server. TypeSafe's own 422 body is FastAPI's `HTTPValidationError`
  shape, so FastAPI's default validation errors already match.
- **Schemas:** TypeSafe's OpenAPI spec (`https://api.typesafe.ai/openapi.json`), snapshotted and dated in
  `schemas/`, is the source of truth; request/response models are generated from it.
- **`typesafe-sdk`** (official, MIT) — dev dependency for contract tests against our server and as the
  client for the Jev baseline.
- **Jev baseline via OpenRouter**, not TypeSafe directly: OpenRouter's System One API
  (`POST https://openrouter.ai/api/v1/systemone`) implements TypeSafe's request and response shapes and
  accepts the official SDK with `base_url="https://openrouter.ai/api"` and an OpenRouter key. The same
  `OPENROUTER_API_KEY` serves our OpenRouter chat-completions profiles and the Jev baseline; they are
  different OpenRouter endpoints.
- Backends, all through Mellea:
  - **OpenRouter** — `OpenAIBackend` with OpenRouter's base URL (or the LiteLLM backend; chosen in Phase 0).
  - **HF Inference Endpoint** — `OpenAIBackend` with the endpoint's `/v1` base URL (vLLM or TGI).
  - **Ollama** — `OllamaModelBackend`, which in 0.8.0 passes `logprobs` / `top_logprobs` through.

## Design Decisions

### 1. API server (exact Jev mimic)
- **Endpoints:** `POST /v1/systemone` and `GET /v1/models`, bearer-token auth
  (`Authorization: Bearer <key>`), JSON bodies. Every response carries an `x-typesafe-request-id` header
  (the SDK reads it).
- **Request/response:** exactly TypeSafe's schema (`api-schema.md`): `state` (string | object | array),
  `model`, `questions` (noul / choice / score); response `model`, `answers`, `usage`. Choice and score
  answers carry `confidence`.
- **Errors:** 401 (bad key), 422 (validation, FastAPI `HTTPValidationError` body), 429 / 529 (backpressure
  and backend overload), matching TypeSafe's table.
- **`model` → model profile.** The request's `model` names a profile in `configs/models.yaml` (backend,
  model id, prompt config, batching strategy, limits). `GET /v1/models` lists the profiles in TypeSafe's
  shape (`name`, `description`, `release_date`). An alias such as `oso-latest` can point at a profile.
- **Compatibility is tested, not assumed:** contract tests run the official `typesafe-sdk` against our
  server with `TYPESAFE_BASE_URL` set to it.
- **Question keys are never sent to the model**, matching Jev's documented behavior.

### 2. Probability protocol (label-token probabilities)
Per question:
1. **Label the options** with single-token labels from the prompt config (e.g. `A`, `B`, `C` for
   choices; `Yes` / `No` for nouls; `0` … `9` for score levels, which fits Jev's 10-level cap).
   Option keys are shown alongside descriptions but are not the answer tokens, since keys are often
   multi-token.
2. **Generate one answer token** with `logprobs` and `top_logprobs` requested (`max_tokens` from config,
   usually 1). Streaming is off: Mellea 0.8.0's OpenAI backend drops logprobs when merging streamed
   chunks, while the non-streaming raw response keeps them.
3. **Read label probabilities** from the answer position's top logprobs. Token variants are matched per
   config (leading whitespace, case), and a label's variants are summed.
4. **Normalize:** P(option) = P(label) / Σ P(all labels).
5. **Fill the Jev answer:**
   - **noul:** `noul` = P(Yes).
   - **choice:** `choice` = argmax key (ties → first in `criteria` order), `probabilities` for every key.
   - **score:** `probabilities` per level (`"0"`, `"1"`, …), `score` = Σ level × P(level), and `legend`
     mapping level → description.
   - **confidence** (choice, score) = clip((N · max p − 1) / (N − 1), 0, 1). TypeSafe publishes this only
     as an approximation of its own formula; the eval checks it against Jev's returned values.

Constraints and diagnostics:
- **Label mass** — Σ P(label tokens) before normalization — is logged per question. Low mass means the
  model wanted to answer something else; it is the main prompt-tuning signal.
- **Missing labels.** A label absent from the returned top-k gets probability per config (`zero`, or
  `floor` = the smallest returned probability). Recorded per question.
- **Option-count cap (decided: keep it simple).** Every label must be readable from the top-k, and
  backends cap `top_logprobs` (often 20). Each profile declares `max_options` (≤ its `top_logprobs`) and
  requests above it get a 422. This is deliberately smaller than Jev's 255 and is the one declared
  deviation from Jev's limits.
- **Answer position.** Some models emit whitespace, a preamble, or reasoning before the answer. Config
  controls where the answer is read: the first generated token, or the first position (within
  `max_tokens`) whose top-k contains labels. Reasoning must be disabled or the model excluded.
- **Temperature** is recorded per profile, since backends differ on whether logprobs reflect it.

### 3. Prompt configuration
Prompts live in versioned config files (`configs/prompts/<name>.yaml`), never in code. A prompt config
defines:
- **Layout order** — by default system, then state, then question, so everything up to the question is a
  shared prefix across all questions in a request (see section 5).
- **System text.**
- **State rendering** — header, and how objects and arrays are serialized (pretty JSON, compact JSON, or
  YAML).
- **Question template** (Jinja, rendered into Mellea `Message` components) — how instructions, options, and
  criteria appear, per question type.
- **Label scheme** per question type, token-matching rules, and the missing-label policy.
- **Answer primer** — text that ends the prompt just before the answer token (e.g. `Answer:`).

Every run and response trace records the prompt config name, version, and content hash. Changing a
prompt means a new version, and comparisons are always between named versions.

### 3a. Criteria handling
`criteria` is how a Jev caller defines what each answer means, and it is where most prompt engineering is
expected. Its shape differs by question type, and every rendering choice below is a prompt-config knob,
never code.

| Type | `criteria` shape (Jev) | What it means | Rendered as |
| --- | --- | --- | --- |
| `noul` | optional `{true, false}`; either side may be missing; each string / object / array | what a yes and a no mean | a description under the Yes and No labels, or nothing if absent |
| `choice` | map key → description (string / object / array / null) | what each option means; null = the key is self-explanatory | one line or block per option: label, key, description |
| `score` | ordered array of level descriptions | position = level, starting at 0 | levels in order under digit labels, lowest first |

Config knobs (per question type, in the prompt config):
- **Option line format** — which of label, key, and description appear, and how (e.g.
  `{label}. {key}: {description}`, or `{label}. {description}` to hide opaque keys).
- **Null descriptions** — show the key alone (default), or the key with a fixed phrase.
- **Key display** — show keys, hide them, or show them only when there is no description. Keys may be
  meaningful words (`billing`) or opaque ids (`record_18`); the eval covers both.
- **Structured descriptions** (objects / arrays) — serialized as YAML, pretty JSON, or compact JSON.
  Field names are kept, since Jev's docs have callers refer to data by field name in backticks.
- **Noul criteria** — how `true` / `false` descriptions are attached to the Yes / No labels, what to show
  when only one side is given, and whether the instructions or the criteria come first.
- **Score framing** — an optional line stating that levels are ordered from lowest to highest.
- **Criteria preamble** — an optional sentence telling the model the descriptions define the answers.
- **Label collisions** — when option keys look like labels (keys `A`–`E` with letter labels), use the
  config's alternate label scheme (e.g. numbers). The rule is generic and declared in config; it never
  looks at request content beyond the keys.

Rules that don't change:
- Criteria are passed through faithfully: no rewording, reordering, or dropping options, and no
  truncation. Option order in the rendering is `criteria` order.
- Contradictions between instructions and criteria (Jev's docs warn against a noul whose `true` means
  "no") are rendered as given, never "fixed". The eval records how the model behaves.
- Criteria are per question, so they render after the state and never break the shared prefix.

### 4. Backends and model profiles
`configs/models.yaml` defines profiles:

```yaml
profiles:
  oso-local-llama:
    backend: ollama                # ollama | openrouter | hf_endpoint
    model_id: llama3.1:8b
    prompt: default@1
    strategy: warm_fanout
    top_logprobs: 20
    max_options: 20
    concurrency: 4                 # match OLLAMA_NUM_PARALLEL
  oso-openrouter-llama70b:
    backend: openrouter
    model_id: meta-llama/llama-3.3-70b-instruct
    provider: {require_parameters: true}   # only route to providers that honor logprobs
    prompt: default@1
    strategy: warm_fanout
  oso-hf:
    backend: hf_endpoint
    base_url: ${HF_ENDPOINT_URL}
    prompt: default@1
    strategy: warm_fanout
aliases:
  oso-latest: oso-local-llama
```

Model ids above are illustrative. Whether a given model returns usable logprobs is a per-profile fact
established in Phase 0 and recorded in the profile.

### 5. Batching and cache reuse
All questions in a request share the rendered prefix (system + state). Strategies, set per profile:
- **`sequential`** — one question at a time. Baseline for measurement.
- **`fanout`** — all questions concurrently, up to the profile's concurrency limit.
- **`warm_fanout`** (default) — the first question alone, then the rest concurrently. The first call
  populates the backend's prefix cache (vLLM automatic prefix caching, Ollama's KV cache, provider-side
  prompt caching on OpenRouter) before the others need it.
- **`batched`** (experimental) — every question in one prompt and one generation, with answers read at
  successive answer positions. Fewest tokens, but later answers can be conditioned on earlier ones, which
  Jev explicitly avoids.

Also: identical questions within a request are deduplicated; cached-token counts are recorded wherever a
backend reports them.

### 6. Latency and tracing
- **End-to-end latency** per request, measured in the server from receipt to response, plus per model
  call (queue wait and call duration), render time, and strategy.
- Each request writes a trace to `runs/traces/*.jsonl`: request id, profile, prompt hash, rendered
  messages, raw backend response (including logprobs), label mass, timings, and usage. This replaces a
  separate gateway for payload inspection and doesn't distort latency.
- Timing summaries are available in an optional `Server-Timing` response header, which adds no fields to
  the JSON body.

### 7. Lightweight eval
A small, fast harness for prompt tuning and sanity checks (details in `eval-design.md`):
- **Cases** in Jev request format with expected answers (`evals/sanity-v0.jsonl`), each tagged `tune`
  or `holdout`.
- **Targets:** our server (any profile) and Jev via OpenRouter's System One API, pinned to `jev-1.13`.
  Jev responses are cached to disk by request hash and the model id OpenRouter returns, so re-running the
  eval doesn't re-spend OpenRouter credits.
- **Metrics:** argmax accuracy, probability on the expected answer, Brier, agreement with Jev (argmax
  agreement, mean |Δp|), label mass, invariance checks (option order, key renaming, single- vs.
  multi-question packing), a check of the confidence formula against Jev's own values, and latency
  percentiles and cache stats.
- Output: `runs/eval/<run>/results.jsonl` plus a Markdown report keyed by profile and prompt hash.

## Roadmap (after MVP)
1. **`batched` strategy** hardened, if the eval shows it's safe.
2. **Decision Index** — run the finished endpoint through the kit's `http` engine. Its house rules (one
   fixed prompt for every benchmark, no tuning on the suite) apply.
3. **Post-hoc calibration** per profile (temperature scaling on labeled data), optional.
4. **Guided decoding** on backends that support it, to force the answer into the label set.

## Files
```
open-system-one/
├── CLAUDE.md                  # this file
├── api-schema.md              # the Jev API as implemented here
├── eval-design.md             # eval cases, metrics, comparison with Jev
├── README.md
├── .claude/
│   ├── settings.json
│   └── rules/
│       ├── mellea.md          # installed Mellea source is API truth; all model calls through Mellea
│       ├── api-compat.md      # never deviate from TypeSafe's schema; contract tests with the SDK
│       └── prompts.md         # prompts only in config; version + hash; tune on `tune` cases only
├── pyproject.toml
├── uv.lock
├── .env.example
├── .gitignore
├── schemas/
│   └── typesafe-openapi-<date>.json   # snapshot of TypeSafe's OpenAPI spec
├── configs/
│   ├── models.yaml            # model profiles + aliases
│   └── prompts/
│       └── default.yaml       # prompt config (versioned)
├── evals/
│   └── sanity-v0.jsonl        # eval cases
├── runs/                      # traces, eval results, Jev cache (gitignored)
├── src/open_system_one/
│   ├── server/                # FastAPI app, auth, errors, request ids
│   ├── schema/                # generated TypeSafe models
│   ├── prompts/               # prompt config loading + Mellea components
│   ├── protocol/              # label-token probability protocol
│   ├── backends/              # profile → Mellea backend
│   ├── strategies/            # sequential / fanout / warm_fanout / batched
│   ├── tracing/               # timings + JSONL traces
│   └── eval/                  # eval runner, Jev client + cache, report
└── tests/
    ├── contract/              # official typesafe-sdk against our server
    └── fixtures/              # recorded backend responses for offline tests
```

## Commands
Target interface; exact flags are finalized during implementation. `.env`-dependent commands need
`--env-file` (uv does not load `.env` automatically).

```
uv sync
cp .env.example .env

# Serve the API
uv run --env-file .env open-system-one serve --port 8080

# Use it with the official SDK
TYPESAFE_BASE_URL=http://localhost:8080 TYPESAFE_API_KEY=$OSO_API_KEY python my_app.py
# (TYPESAFE_API_KEY here is our own OSO_API_KEY; the project never uses a TypeSafe key)

# Eval: our profile vs. Jev on the sanity set
uv run --env-file .env open-system-one eval --cases evals/sanity-v0.jsonl \
    --target oso-local-llama --baseline jev-1.13 --split tune

# Tests (offline, from fixtures)
uv run pytest
```

## Environment
- `OSO_API_KEY` — bearer key(s) our server accepts.
- `OPENROUTER_API_KEY` — OpenRouter profiles **and** the Jev baseline (OpenRouter's System One API).
- `HF_TOKEN`, `HF_ENDPOINT_URL` — HF Inference Endpoint profiles.
- `OLLAMA_HOST` — default `http://localhost:11434`.
- Unit tests and CI run offline against fixtures; only integration tests hit live backends or Jev.

## Setup status / next steps
Status (2026-09-26): Phase 0 done except Ollama. MVP steps 1–2 and most of 3–4 are built and
running live on the HF endpoint (`oso-granite-3b`, alias `oso-latest`) and on OpenRouter
(`oso-granite-micro-cf`: Granite 4.0 Micro 3B pinned to Cloudflare, `fanout`): FastAPI server, bearer auth,
request ids, 401/422/429/529/500 mapping, prompt config `default@1`, the label-token protocol,
`sequential` / `fanout` / `warm_fanout`, dedup, JSONL traces, `Server-Timing`. 26 offline tests
(unit, SDK contract, and a recorded Granite fixture) pass. Not built yet: OpenRouter profiles on
providers without tool support that get filtered by `tools: null` (workaround pending), Ollama, `batched`, the eval harness.

Inside the Claude Code sandbox, uv can't write `~/.cache/uv`; set `UV_CACHE_DIR=$TMPDIR/uv-cache`.

Phase 0 results so far (2026-09-26; details in `spikes/FINDINGS.md`). Spikes 1–6 are done for
OpenRouter, the HF endpoint, the SDK, and Jev; Ollama is pending.
- **OpenRouter returns one logprob position at most**, often the *last* token → `max_tokens: 1` and
  first-token answer position on OpenRouter profiles.
- **Pin OpenRouter providers** (`provider.order` + `allow_fallbacks: false`) to ones verified in the
  matrix; `require_parameters` doesn't stop silent logprob drops (Cloudflare).
- **Mellea 0.8.0 sends `"tools": null`**, which makes OpenRouter drop providers without tool support;
  needs a thin workaround before OpenRouter profiles can use SambaNova/Parasail. **Postponed**
  (2026-09-26): the MVP targets the HF endpoint first; revisit when OpenRouter profiles are added.
- `top_logprobs` cap is 20 on OpenRouter and on vLLM (default). Temperature 0 gives raw logprobs on
  every provider tested.
- Assistant prefill works on vLLM (`continue_final_message`), not reliably on OpenRouter; answer
  tokens may carry a leading space.
- The SDK retries 429 and all 5xx (including our 500) twice by default; its default timeout is 10 s.
- Jev via OpenRouter returns `typesafe/jev-1.13-20260917`, adds `usage.cost`, and sends no
  `x-typesafe-request-id`; the confidence formula matches Jev within rounding.
- The schema snapshot must be copied into `schemas/` by hand (the deny rule blocks Claude creating it).

Phase 0 — spikes (findings go in `spikes/FINDINGS.md`, then back into this file):
1. **Schema snapshot.** Download TypeSafe's `openapi.json` into `schemas/`, dated; generate models; diff
   against the SDK's bundled models.
2. **Mellea logprobs, per backend.** Which call path returns the raw response with logprobs (non-streaming
   `OpenAIBackend` `mot.raw.response`; `OllamaModelBackend` logprobs), and that `logprobs`,
   `top_logprobs`, and `max_tokens` pass through model options.
3. **Backend capability matrix.** For each backend and a few candidate models: logprobs returned or not,
   `top_logprobs` cap, whether logprobs reflect temperature, prefix caching behavior, and concurrency
   limits. OpenRouter: confirm `provider.require_parameters` stops silent drops.
4. **Answer position.** How chat templates and models start their reply (whitespace, preambles,
   reasoning); whether an answer primer or assistant prefill is honored.
5. **SDK contract.** Point the official `typesafe-sdk` at a stub server; confirm paths, headers, error
   handling, and retries.
6. **Jev via OpenRouter.** Call `jev-1.13` through OpenRouter's System One API with the official SDK;
   confirm the request/response shapes match the schema snapshot, record the returned model id and
   pricing, and measure response variance across repeated calls. (The SDK's `models.list()` doesn't work
   against OpenRouter, which returns its own Models API shape; the baseline doesn't need it.)

Then (MVP):
1. Schema models + FastAPI server with stub answers; SDK contract tests passing.
2. Prompt config loader + default prompt; protocol on the HF endpoint (Granite 3B) with `sequential`.
3. `fanout` and `warm_fanout`; OpenRouter backend (after the `tools: null` workaround); Ollama later.
4. Tracing and latency.
5. Eval harness with the Jev cache; first tuning pass on `tune` cases; report on `holdout`.

## Cautions
- **Logprob availability varies.** Many hosted models or providers don't return logprobs, or return
  fewer than requested. A profile is valid only after the Phase 0 capability check.
- **Reasoning models** put thinking tokens before the answer; disable reasoning or don't use them.
- **Don't overfit the prompt to the eval.** Tune on `tune` cases, report on `holdout`, and keep holdout
  results out of tuning decisions.
- **Jev responses are cached locally** and never committed; check OpenRouter's and TypeSafe's terms
  before publishing comparisons.
- **Pin the Jev baseline** to `jev-1.13`, not `jev-latest`, so a new Jev release doesn't silently shift
  the comparison. Moving the pin is a deliberate change that invalidates the cache.
- The confidence formula is TypeSafe's published approximation; treat mismatches with Jev as findings.
- Not affiliated with TypeSafe AI.

## Future Goals
- Decision Index run via the kit's `http` engine.
- Use the endpoint as cheap in-loop validators in Mellea IVR loops.
- Model routing as a first application.
