# Decision Index runs

Scored with the [decision-index kit](https://github.com/apolinario/decision-index) (commit `19ad28e`),
edition 0.2 suite rebuilt locally and byte-identical to the lab's files. Every row is sent unmodified
through the kit's `http` engine to our `/v1/systemone`; sharding (`scripts/decision_index_run.py`) only
parallelizes the kit's runner. Raw results and traces stay in `external/decision-index/runs/` (not
committed).

## 2026-09-26 — `oso-granite-3b-wide`, prompt `default@3`

- Model: `ibm-granite/granite-switch-4.1-3b-preview` on a Hugging Face Inference Endpoint (vLLM 0.19.1,
  `--max-logprobs 256`), 15 × L4 during the run.
- Prompt `default@3` (hash `41502c409447`), chosen on the sanity eval's `tune` split before the run and
  not changed after it. Choices up to 20 options use letters, up to 200 use numbers; per-call
  `top_logprobs` = min(256, max(20, 2 × options)).
- Coverage: 151,476 / 151,476 requests answered (`ok`), 0 unsupported, 0 errors; 151,034 scored after the
  kit's 442 standard exclusions. `complete: true`.
- Wall time 33.8 min (96 shards, 5 server processes). Kit-measured request latency (under full load):
  median 445 ms, p95 4.2 s.

**Decision Index 23.58**

| Area | Skill | Raw |
| --- | --: | --: |
| Tools & Automation | 0.374 | 0.470 |
| Language Understanding | 0.255 | 0.476 |
| Retrieval & Classification | 0.226 | 0.426 |
| Arts & Human Judgment | 0.172 | 0.393 |
| Knowledge & Reasoning | 0.153 | 0.344 |

Strongest (skill): BFCL 0.79, HellaSwag 0.68, RouterBench 0.52, FinEntity 0.51, BPoMP 0.47,
CLINC150 0.44, BANKING77 0.41 (the last two were unanswerable under the old 20-option cap).
At or below chance (skill 0): ForecastBench (Brier 0.353 vs 0.25), RAGTruth, CRUXEval, HLE, cfcolor,
POP909-CL. Near zero: ACOS, home appliance simulator, ChessBench.

Context, not a like-for-like comparison: the public leaderboard (multimodalart/jev-decision-index)
reports `balanced_skill` on its own 120,340-request suite; there Jev 1.13 is 57.89 and fine-tuned
decision models of 2–5B mostly score 20–40.
