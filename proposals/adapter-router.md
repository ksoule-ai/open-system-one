# Proposal: an adapter router behind `/v1/systemone`

Status: draft for review (2026-10-04). Nothing here is built. Experiments since then
([`results/adapters.md`](../results/adapters.md)) found that each adapter helps on at most one of the
benchmarks it fits (factuality: RAGTruth; guardian groundedness: HoVer; answerability with factuality:
ContractNLI) and is worse than `default@3` elsewhere.

## Summary

A new kind of model profile, `oso-granite-router`, answers each question of a Decisions API request in
one of four ways, chosen by Granite itself through a tool call:

| Tool | What runs | Answers |
| --- | --- | --- |
| `hallucination_check` | Granite Switch's built-in `hallucination_detection` adapter | 1 if any sentence of the response is flagged, else 0 |
| `guardian_check` | the `guardian-core` adapter | the adapter's risk score, 0 to 1 |
| `requirement_check` | the `requirement-check` adapter | the adapter's score, 0 to 1 |
| `decision_program` | today's System One path: prompt `default@3`, label-token probabilities | full Jev answer, any question type |

`decision_program` is also the fallback for anything the router can't or shouldn't send to an
adapter. The request and response stay exactly Jev-shaped; the routing is visible only in traces and
an optional header.

The whole thing runs on the one Hugging Face endpoint we already use: the Granite Switch 4.1 3B
model is the router (with no adapter active), hosts the three adapters, and is the model behind
`decision_program`.

## Two facts that shape the design

1. **Our vLLM endpoint can't take tool calls yet.** A test request with tools came back with
   `"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set`, and
   `tool_choice: "required"` fails the same way. Tool calling needs two container args added to the
   endpoint and a restart, as we did for `--max-logprobs 256`. The parser to use for Granite 4.x
   (`hermes` or `granite`) has to be checked in the spike: the switch model's chat template asks for
   JSON inside `<tool_call></tool_call>` tags.
2. **Almost no dev-set questions are `noul`.** Only RAGTruth (100) and PhishNChips (600) use `noul`.
   Most yes/no questions are written as two-option choices: HoVer (`SUPPORTED` / `NOT_SUPPORTED`),
   NLI4CT, BFCL, ToolRet, BRIGHT, ACOS, CLadder and others. If adapters only answer `noul`, they
   will barely be used. The proposal therefore lets adapters answer two-option choices too, with the
   router saying which option is the "yes" side (see step 4).

## Flow for one request

```
POST /v1/systemone  (Jev request, model = oso-granite-router)
  │
  ├─ 1. validate + dedupe questions        (same code and limits as today)
  │
  ├─ 2. route each unique question         (one tool call to Granite, no adapter active)
  │       input:  the question's instructions and criteria, plus the state's field names
  │       output: one tool call with arguments
  │
  ├─ 3. check the call                     (code, no model)
  │       wrong question type, unknown field, bad arguments, no call → decision_program
  │
  ├─ 4. run the tool                       (adapter via Mellea, or the existing default@3 path)
  │
  └─ 5. build the Jev answer               (same response builder and invariants as today)
```

### 1. Input

Unchanged: the Jev request is validated against the schema and the profile's limits, and identical
questions are deduplicated.

### 2. Routing call

One call per unique question to the switch model with no adapter selected, temperature 0, and the
four tools from `configs/routers/<name>.yaml`.

**What the router sees.** You asked for "just the instruction". I propose a little more, for review:

| Router input | Why |
| --- | --- |
| `instructions` | the question itself |
| `criteria` | often carries the meaning (RAGTruth's `true` = "contains content that is not supported…") and the option keys a two-option mapping needs |
| the state's field names (not values) | the adapters need to know which field is the response and which is the context; e.g. RAGTruth's `prompt` / `response`, HoVer's `claim` / `evidence` |

The state's content is never shown to the router, which keeps the call short (no 2,000-token
article) and makes the routing decision depend only on the question. A string state has no fields,
so the hallucination tool can't be used for it (no way to separate context from response).

**Routing cache.** Routing depends only on the question definition and the state's field names, so
the decision can be cached under that key. On the dev set, 36,661 questions are only 10,152 distinct
question definitions (ACOS: 22,692 → 402; ContractNLI: 1,700 → 17; PhishNChips: 900 → 9), so a
cache cuts routing calls by about 72%. The cache is per server process and recorded in traces.

### 3. Checking the tool call

Code, not the model, decides whether the call can run. Anything that fails a check goes to
`decision_program` and the reason is traced:

- the tool is an adapter but the question is a `score` question or a choice with more than two options;
- a field name in the arguments isn't in the state;
- for a two-option choice, `yes_option` isn't one of the two keys;
- the arguments don't match the tool's schema, there is no tool call, or there is more than one.

### 4. Tools

All model calls go through Mellea (`OpenAIBackend` with `load_embedded_adapters=True`, as in
`spikes/ragtruth_hallucination.py`). The tool descriptions and the router's system text are prompt
text, so they live in the router config, versioned and hashed like prompt configs.

| Tool | Arguments from the router | Mellea call | Raw result |
| --- | --- | --- | --- |
| `hallucination_check` | `response_field`, `context_fields`, optional `question_field`, `yes_option` | `rag.flag_hallucinated_content(response, documents, ctx, backend)` | per-sentence `faithful` / `unfaithful` / `partial` / `NA` |
| `guardian_check` | `criteria` (a bank key such as `harm`, `jailbreak`, `groundedness`, or custom text), `target_field`, `yes_option` | `guardian.guardian_check(ctx, backend, criteria)` | risk score 0–1 |
| `requirement_check` | `requirement` (text), `target_field`, `yes_option` | `core.requirement_check(ctx, backend, requirement)` | score 0–1 |
| `decision_program` | none | today's `render_question` + label-token protocol with `default@3` | full answer |

How the adapter inputs are built from the state:

- **hallucination:** the context fields become the documents, the response field is the response,
  and the question field (if any) is the user turn. The instructions are not used as the user turn,
  since they ask *about* the response rather than being the question it answered.
- **guardian and requirement check:** the target field is put in the assistant turn of a short chat,
  which is what both adapters judge by default. For requirement check, the requirement text is the
  router's restatement of the question.

**Polarity.** Each adapter result is a probability that "the bad thing is present" (hallucination,
risk) or "the requirement is met". For a `noul`, the router's arguments are written so that a yes
answer means the same thing (the `criteria` for guardian, the `requirement` text). For a two-option
choice, `yes_option` names the key that gets the probability.

### 5. Building the answer

| Question type | Answer from an adapter score `s` |
| --- | --- |
| `noul` | `noul = s` |
| two-option `choice` | `probabilities = {yes_option: s, other: 1 − s}`, `choice` = argmax (ties → first in `criteria` order), `confidence` by today's formula |
| anything else | not routed to adapters |

For hallucination, `s` is 0 or 1 as you specified: 1 if any sentence is flagged. With a hard 0/1,
`confidence` is always 1 and Brier-style metrics see no uncertainty; see the open questions.

The response goes through the same invariant checks as today (keys match, probabilities in range and
summing to 1). Nothing is added to the body. Routing details go to:

- the trace: tool chosen, arguments, raw router output, cache hit or miss, check failures, adapter
  records, and timings per stage;
- an optional `x-oso-routes` header with a compact summary (e.g. `q1=hallucination_check;q2=decision_program`);
- `Server-Timing`, with `route` and `adapter` stages added.

## Configuration

`configs/models.yaml`:

```yaml
  oso-granite-router:
    backend: hf_endpoint
    base_url: ${LITELLM_BASE_URL}
    api_key_env: LITELLM_API_INFERENCE_KEY
    model_id: hf/ibm-granite/granite-switch-4.1-3b-preview
    adapter_source: ibm-granite/granite-switch-4.1-3b-preview   # where Mellea finds adapter configs
    router: adapter-router@1        # configs/routers/adapter-router@1.yaml
    fallback: oso-granite-3b-wide   # profile used by decision_program (prompt default@3)
    adapter_timeout: 120
    adapter_concurrency: 8
```

`configs/routers/adapter-router@1.yaml` holds the router's system text, the four tool definitions
(names, descriptions, argument schemas), which question types each tool accepts, and the
hallucination flag rule (which sentence ratings count as "flagged"). Like prompts: every change is a
new version, and traces record name, version and content hash.

## What stays the same

- The Jev request and response schema, status codes and errors. The contract tests with the
  official SDK run against the router profile too.
- The option cap, which is the fallback profile's (200 with `oso-granite-3b-wide`), so no new
  declared deviation.
- Question keys are never sent to any model, router included.
- `decision_program` is byte-for-byte today's `default@3` path, so any score change comes from routing.

## Latency and cost

- **Routing:** one short generation (about 50 output tokens for a tool call) per uncached question.
- **Hallucination adapter:** slow. It writes a JSON verdict and explanation for every sentence; one
  RAGTruth sample took 24.5 s, and the 100-sample dev run averaged about 35 s per sample at 8 at a
  time. That is far past the official SDK's 10 s default timeout, so callers would need to raise it.
  The Decision Index kit's 600 s timeout is fine.
- **Guardian and requirement check:** these return a single short JSON score, so they should be
  close to one `default@3` call. Not measured yet.
- Everything runs on the existing endpoint; no new model or provider.

## How we'd evaluate it (dev set only)

1. **Routing table.** For every dev benchmark, the share of questions sent to each tool, plus every
   check failure. This shows routing behavior before any scores, and is cheap with the cache (about
   10,000 calls).
2. **Scores against the baseline.** A full dev run of the router profile against the existing
   `oso-granite-3b-wide` + `default@3` run, per benchmark. Differences under about 0.02 are noise (we
   measured that between two runs of the same model and prompt).
3. **Oracle routing.** For the benchmarks where an adapter clearly applies (RAGTruth →
   hallucination; HoVer and NLI4CT are candidates), force the tool and compare. This separates "the
   router chose badly" from "the adapter is worse than `default@3`". We already have one data point:
   on RAGTruth the hallucination adapter scores F1 0.608–0.674, depending on the flag rule, against
   0.301 for `default@3`.
4. **Latency.** p50 and p95 per tool and per request.

Decision Index house rules apply if this ever runs on the suite: one fixed router config for every
benchmark, tuned on the dev set only. The router choosing a tool from the request itself is part of
the engine, not per-benchmark prompt switching, but that reading should be checked against the
kit's rules before a suite run.

## Phases

0. **Spike (about a day).** Add the tool-calling args to the endpoint; confirm the parser; run the
   router alone over the dev set's 10,152 distinct questions and produce the routing table. Measure
   guardian and requirement-check latency on a handful of samples.
1. **Build.** Router config loader, routing cache, call checks, the three adapter tools through
   Mellea, answer mapping, traces and header; unit tests with a fake router; contract tests on the
   new profile.
2. **Evaluate.** Steps 1–4 above on the dev set.
3. **Decide** whether to keep it, and whether `oso-latest` should point at it.

## Alternative if tool calling doesn't work well

Route with our own protocol instead of a tool call: ask the switch model a choice question over the
four tools and read the label-token probabilities. That needs no endpoint change, gives a
probability per route (useful for a "only route when confident" threshold), and costs one token. It
can't fill in arguments, though, so field names and polarity would come from a second step. I'd only
switch to this if the spike shows tool calls are unreliable or slow.

## Open questions for you

1. **Router input:** instructions only, as you described, or instructions + criteria + state field
   names, as proposed? Without field names the hallucination tool can't know which field is the
   response.
2. **Two-option choices:** should adapters answer them (needed for HoVer, NLI4CT, BFCL and most of
   the dev set's binary questions), or `noul` only?
3. **Hallucination output:** hard 0/1 as you specified, or a soft score such as the share of
   flagged sentences? 0/1 means `confidence` is always 1 and probability-based metrics (Brier) get
   no gradation.
4. **Hallucination flag rule:** count `unfaithful` only, `unfaithful` + `partial` (F1 0.608 on
   RAGTruth dev), or anything not `faithful` (0.674, but `NA` sentences such as "Unable to answer…"
   then count as hallucinated)?
5. **Guardian scope:** `guardian-core` only, or also `policy-guardrails` and `factuality-detection`,
   which the switch model also hosts?
6. **Endpoint change:** OK to add `--enable-auto-tool-choice --tool-call-parser <parser>` to the
   endpoint's container args and restart it?
7. **Router model:** the switch model with no adapter (proposed, one endpoint), or a separate base
   Granite model?
