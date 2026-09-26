# Eval Design (lightweight)

## Goal
A fast harness for **prompt tuning** and **sanity checks**: send the same Jev-format requests to our
server and to Jev, and compare. It is not a benchmark; the Decision Index is the future benchmark.

## Cases
`evals/sanity-v0.jsonl`, one case per line:

```jsonc
{
  "id": "noul-refund-clear",
  "category": "noul/clear",
  "split": "tune",                         // tune | holdout
  "request": {"state": "...", "questions": {...}},   // Jev request minus "model"
  "expect": {
    "refund": {"answer": true}             // noul: true / false, or {"band": [0.2, 0.8]} for ambiguous
    // choice: {"answer": "billing"} or {"answer_in": ["billing", "account"]}
    // score:  {"answer": 2} or {"band": [1.5, 2.5]}
  },
  "jev_reference": {"refund": 0.99}        // optional: a value published in TypeSafe's docs (jev-1.13.0)
}
```

- `expect` is our label, written by hand. `jev_reference` values come only from TypeSafe's published
  docs and are informational (Jev's probabilities vary slightly between calls).
- Roughly two thirds `tune`, one third `holdout`, balanced by category.

## Targets
- **Our server:** any profile, via the official SDK (so the eval also exercises API compatibility).
- **Jev:** through **OpenRouter's System One API**, using the official SDK constructed explicitly with
  `base_url="https://openrouter.ai/api"` and `api_key=OPENROUTER_API_KEY` (never from `TYPESAFE_*`
  environment variables, which point the SDK at our own server during contract tests). The model is
  pinned to `jev-1.13`; OpenRouter maps it to `typesafe/jev-1.13` and returns the served model id.
- Jev responses are cached in `runs/jev-cache/` by (request hash, returned model id). Re-runs hit the
  cache unless `--refresh-jev`, so repeated tuning runs don't re-spend OpenRouter credits.

## Invariance transforms
Generated automatically from each case, not written by hand:
- **Option order:** permute choice `criteria` order; the probability for each key should not move much.
- **Key renaming:** rename question keys; answers must be identical, since keys never reach the model.
- **Packing:** each question alone vs. all questions in one request; answers should match within
  tolerance for every strategy except `batched`.
- **Repeat:** the same request twice; measures run-to-run variance (and cache effects).
- **Criteria ablation:** for nouls with `criteria`, the same request without them. Jev's docs advise
  trying both, so the delta is measured for our server and for Jev.
- **Key opacity:** choice keys replaced with opaque ids (`opt_1`, `opt_2`, …) while descriptions stay;
  measures how much the model leans on key names versus descriptions.

## Metrics
Per case and category, for each target:
- **Correctness:** argmax matches `expect`; for bands, value inside the band.
- **P(expected):** probability assigned to the expected answer; **Brier** for nouls.
- **Agreement with Jev:** argmax agreement; mean |Δp| over options (or over `noul`); score |Δ|.
- **Confidence check:** our confidence formula applied to *Jev's* probabilities vs. Jev's returned
  `confidence`. This tests the formula, not our model.
- **Label mass:** Σ P(label tokens) before normalization; low mass flags prompt problems.
- **Invariance deltas:** max probability shift under each transform.
- **By criteria shape:** every metric is also sliced by criteria shape — none, string, null, object /
  array, opaque keys, label-colliding keys, one-sided noul criteria, contradictory — since that's where
  prompt changes are expected to matter most.
- **Latency:** p50 / p95 end-to-end, per strategy; model calls per request; cached-token share where
  reported.

## Report
`runs/eval/<timestamp>-<profile>-<prompt-hash>/`: `results.jsonl` (per case, per target, raw answers and
metrics) and `report.md` (tables by category; worst cases by |Δp| and by label mass; latency summary).
Two runs can be diffed to compare prompt versions.

## Prompt-tuning protocol
1. Change the prompt config (new version).
2. Run on `tune` cases; compare against the previous version and Jev.
3. Repeat. Only when a version is final, run `holdout` once and report it.
Holdout results never feed back into tuning.
