# API Schema

## Goal
Implement TypeSafe's System One API exactly, so any Jev client works against our server by changing only
its base URL.

**Source of truth:** TypeSafe's OpenAPI spec (`https://api.typesafe.ai/openapi.json`), snapshotted into
`schemas/typesafe-openapi-<date>.json`. Request/response models are generated from the snapshot (the same
way TypeSafe's own SDK generates its models). Where this doc and the snapshot disagree, the snapshot
wins and this doc is corrected. Refreshing the snapshot is a deliberate, dated change.

## Endpoints
| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/systemone` | Answer typed questions about a state |
| `GET` | `/v1/models` | List available model names |

- **Auth:** `Authorization: Bearer <key>`; keys come from `OSO_API_KEY`.
- **Headers:** JSON in, JSON out. Every response sets `x-typesafe-request-id`, which the SDK reads. An
  optional `Server-Timing` header carries latency summaries without changing the JSON body.

## Request: `POST /v1/systemone`
```jsonc
{
  "state": "Help! My payouts have been failing for 3 days.",   // string | object | array
  "model": "oso-latest",                                        // a profile name or alias
  "questions": {                                                // ≥ 1; keys are yours, never sent to the model
    "is_urgent":  {"type": "noul", "instructions": "Does this convey urgency?",
                   "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"}},
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"billing": "Payments, invoicing, refunds",
                                "technical": "Bugs, outages, integrations",
                                "sales": null}},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]}
  }
}
```

| Type | `instructions` | `criteria` |
| --- | --- | --- |
| `noul` | string, object, or array | optional `{true, false}`, each string / object / array |
| `choice` | string, object, or array | required map option → description (string / object / array / null); up to 255 options |
| `score` | string, object, or array | required ordered array of level descriptions; 2–10 levels |

Our server additionally enforces the profile's **`max_options`** (at most its `top_logprobs`, often 20)
for choices, and returns 422 above it. This is a deliberate, permanent simplification — smaller than
Jev's 255 — and the one declared deviation from Jev's limits.

## Response
```jsonc
{
  "model": "oso-local-llama",          // the profile that answered (aliases resolve to their profile)
  "answers": {
    "is_urgent":  {"type": "noul", "noul": 0.95},
    "department": {"type": "choice", "choice": "billing",
                   "probabilities": {"billing": 0.88, "technical": 0.12, "sales": 0.0},
                   "confidence": 0.81},
    "frustration": {"type": "score", "score": 1.05,
                    "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
                    "probabilities": {"0": 0.0, "1": 0.95, "2": 0.05},
                    "confidence": 0.92}
  },
  "usage": {"input_tokens": 296, "output_tokens": 20}
}
```

How each field is computed (protocol in `CLAUDE.md` > Design Decisions > 2):
- **noul:** `noul` = P(Yes label) after normalization over the Yes/No labels. No `confidence`, as in Jev.
- **choice:** `probabilities` covers every `criteria` key; `choice` is the argmax (ties → first key in
  `criteria` order); `confidence` = clip((N · max p − 1) / (N − 1), 0, 1).
- **score:** levels are indexed from 0 in `criteria` order; `probabilities` and `legend` are keyed by the
  level index as a string; `score` = Σ i · P(i); `confidence` uses the same formula over the levels.
- **usage:** input and output tokens summed over every model call made for the request (all questions,
  all strategies). Where a backend doesn't report usage, the server counts tokens itself and the trace
  marks the counts as estimated.

Invariants checked before returning (a violation is our bug and returns 500, never a partial answer):
- `answers` has exactly the request's question keys, and each `type` matches its question.
- Probabilities are finite, in [0, 1], and sum to 1 within 1e-6 (after rounding for output).

## `GET /v1/models`
```json
{"models": [{"name": "oso-local-llama", "description": "llama3.1:8b via Ollama, prompt default@1",
             "release_date": "2026-09-26"}]}
```
One entry per profile and per alias. `release_date` is the profile's config date.

## Errors
Status codes match TypeSafe's API reference:

| Status | When | Body |
| --- | --- | --- |
| `401` | Missing or invalid bearer key | `{"detail": "..."}` |
| `422` | Schema validation failure, unknown `model`, too many options for the profile | FastAPI `HTTPValidationError`: `{"detail": [{"loc", "msg", "type", "input", "ctx"}]}` |
| `429` | Server-side concurrency limit reached | `{"detail": "..."}`, with `retry-after` |
| `529` | Backend overloaded or rate-limited upstream | `{"detail": "..."}`, with `retry-after` |
| `500` | Backend returned no usable logprobs, invariant violated, other internal errors | `{"detail": "..."}` |

- The SDK retries 429 and 529 with backoff by default, so both carry `retry-after`.
- A backend that returns no logprobs, or none of the labels, is a 500 — never a guessed answer.
- Exact error bodies are checked against the SDK in the contract tests.

## Compatibility tests
- **Official SDK:** `typesafe-sdk` pointed at our server (`TYPESAFE_BASE_URL`) runs every example request
  from TypeSafe's docs; responses must parse into the SDK's own response models.
- **Schema:** our generated models round-trip every Jev response in the eval cache.
- **Keys are private:** renaming question keys must not change any answer (also an eval invariance check).
