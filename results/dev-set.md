# Decision Index dev set: baselines and prompt variants (October 2026)

Runs on [`evals/decision-index-dev/`](../evals/decision-index-dev/README.md) (34 benchmarks, 3,840
requests, from data the scored suite does not use), all through the local LiteLLM gateway and scored by
the kit's 0.2 scorers with `scripts/decision_index_run.py score --rows`. Raw results are in
`runs/di-dev/<run>/` (gitignored).

## Chance-corrected skill

Raw scores are not comparable across benchmarks, or between the dev set and the suite when their label
balance differs. Every number below is skill = clip((score × coverage − chance) / (1 − chance), 0, 1),
with chance measured on the dev rows by [`../scripts/dev_chance.py`](../scripts/dev_chance.py):

- most benchmarks: the kit's random engine (uniform choice over each question's options), 20 seeds,
  scored by the same scorer as the runs;
- RAGTruth: F1 of always answering "hallucinated" (0.667 on the 50/50 dev set; the board's 0.2.1 rule);
- ForecastBench: the board's rule against always predicting 0.5, skill = clip((0.25 − Brier) / 0.25).

Coverage counts unanswered requests as wrong. "–" means no score: over the 20-option cap (BANKING77,
CLINC150+OOS, POP909-CL for the 20-option profiles), not run (ChessBench for Micro `@1`), or every request
failed (PhishNChips for Jev: OpenRouter returned 400 on all of them). iSarcasmEval has no single score.

| Column | Profile | Model | Prompt |
| --- | --- | --- | --- |
| 3B wide @3 | `oso-granite-3b-wide-litellm` | Granite Switch 4.1 3B preview, HF endpoint, 200 options | `default@3` |
| 3B 20-opt @3 | `oso-granite-3b-v3-litellm` | same, 20 options | `default@3` |
| 3B wide @4 (doc) | `oso-granite-3b-wide-doc-litellm` | same, 200 options | `default@4` (state as a document) |
| Micro @3 | `oso-granite-micro-cf-v3-litellm` | Granite 4.0 Micro via OpenRouter (Cloudflare), 20 options | `default@3` |
| Micro @1 | `oso-granite-micro-cf-litellm` | same | `default@1` |
| Jev 1.13 | – | TypeSafe Jev via OpenRouter's System One API | – |

| # | Benchmark | Chance | 3B wide @3 | 3B 20-opt @3 | 3B wide @4 (doc) | Micro @3 | Micro @1 | Jev 1.13 |
| --: | --- | --: | --: | --: | --: | --: | --: | --: |
| 1 | BFCL | 0.162 | 0.857 | 0.833 | 0.845 | 0.761 | 0.773 | 0.952 |
| 2 | ToolRet | 0.114 | 0.268 | 0.281 | 0.280 | 0.261 | 0.268 | 0.378 |
| 4 | BANKING77 | 0.008 | 0.360 | – | 0.325 | – | – | 0.667 |
| 5 | CLINC150+OOS | 0.003 | 0.367 | – | 0.450 | – | – | 0.765 |
| 6 | RouterBench | 0.554 | 0.513 | 0.513 | 0.530 | 0.251 | 0.238 | 0.538 |
| 9 | Home appliance simulator | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.012 | 0.525 |
| 10 | SGD/SGD-X | 0.342 | 0.417 | 0.417 | 0.253 | 0.379 | 0.386 | 0.000 |
| 11 | ContractNLI | 0.304 | 0.204 | 0.204 | 0.294 | 0.180 | 0.123 | 0.601 |
| 12 | ANLI | 0.338 | 0.193 | 0.172 | 0.038 | 0.182 | 0.126 | 0.550 |
| 20 | BPoMP | 0.483 | 0.497 | 0.478 | 0.149 | 0.381 | 0.458 | 0.845 |
| 21 | Humicroedit | 0.488 | 0.257 | 0.296 | 0.257 | 0.042 | 0.120 | 0.296 |
| 22 | POP909-CL | 0.009 | 0.000 | – | 0.000 | – | – | 0.062 |
| 23 | cfcolor | 0.514 | 0.000 | 0.012 | 0.000 | 0.000 | 0.000 | 0.383 |
| 24 | MMLU | 0.250 | 0.387 | 0.373 | 0.413 | 0.480 | 0.413 | 0.840 |
| 25 | GPQA Diamond | 0.252 | 0.118 | 0.132 | 0.105 | 0.065 | 0.132 | 0.572 |
| 26 | ARC-Easy | 0.255 | 0.879 | 0.879 | 0.839 | 0.906 | 0.906 | 0.987 |
| 27 | ARC-Challenge | 0.257 | 0.664 | 0.677 | 0.704 | 0.691 | 0.650 | 0.946 |
| 28 | WinoGrande | 0.498 | 0.402 | 0.382 | 0.342 | 0.402 | 0.402 | 0.840 |
| 29 | HellaSwag | 0.247 | 0.681 | 0.681 | 0.708 | 0.641 | 0.628 | 0.920 |
| 30 | GSM8K | 0.176 | 0.199 | 0.199 | 0.224 | 0.169 | 0.126 | 0.806 |
| 31 | ChessBench | 0.072 | 0.095 | 0.009 | 0.073 | 0.019 | – | 0.116 |
| 36 | BRIGHT | 0.051 | 0.087 | 0.089 | 0.102 | 0.060 | 0.061 | 0.121 |
| 37 | Amazon ESCI | 0.190 | 0.084 | 0.082 | 0.066 | 0.128 | 0.111 | 0.398 |
| 38 | ACOS | 0.000 | 0.060 | 0.070 | 0.000 | 0.000 | 0.000 | 0.070 |
| 41 | VAST | 0.335 | 0.409 | 0.423 | 0.277 | 0.191 | 0.251 | 0.570 |
| 42 | NLI4CT | 0.490 | 0.300 | 0.317 | 0.149 | 0.284 | 0.308 | 0.682 |
| 44 | CLadder | 0.508 | 0.086 | 0.086 | 0.147 | 0.066 | 0.127 | 0.411 |
| 48 | ForecastBench | Brier 0.25 | 0.000 | 0.000 | 0.000 | 0.010 | 0.000 | 0.388 |
| 56 | PhishNChips phishing decisions | 0.484 | 0.728 | 0.728 | 0.670 | 0.864 | 0.787 | – |
| 57 | MMLU-Pro | 0.108 | 0.343 | 0.359 | 0.215 | 0.295 | 0.295 | 0.904 |
| 59 | RAGTruth response-level hallucinat | 0.667 | 0.000 | 0.000 | 0.000 | 0.102 | 0.102 | 0.564 |
| 61 | HoVer claim verification | 0.506 | 0.170 | 0.211 | 0.130 | 0.069 | 0.028 | 0.433 |
| 64 | New Yorker caption matching | 0.217 | 0.399 | 0.374 | 0.067 | 0.310 | 0.297 | 0.681 |
| | **Mean, the 28 benchmarks every run answered** | | **0.303** | **0.305** | **0.255** | **0.261** | **0.262** | **0.579** |

## Findings

- **Granite 3B is at about half of Jev's average skill** on the dev set (0.303 against 0.579 on the 28
  benchmarks every run answered). On the full suite its 0.2.1 index is 43% of Jev's (24.92 against 57.91).
- **Run-to-run noise.** The two 3B `default@3` runs differ only in their option cap, so for yes/no
  questions and choices of up to 10 options they send identical requests; their scores still differ by up
  to 0.02 per benchmark on such benchmarks (RAGTruth, ForecastBench), because
  the vLLM endpoint is not fully deterministic at temperature 0. With 100 samples per benchmark, treat
  differences under about 0.05 as noise.
- **Prompt variants.** `default@1` and `default@3` differ only in how choice options with both a key and
  a description are shown; Micro scores the same with either (0.262 against 0.261). `default@4`, which
  sends the state as a chat-template document with no system text of our own, is clearly worse (0.255
  against 0.303); on a 100-request pilot it had looked slightly better.
- **RAGTruth's dev balance.** The dev sample is 50/50 hallucinated; the suite has 35%. A model that flags
  most responses (Micro flags 87%) gets F1 0.701 on the dev set but 0.519 on the suite, for the same
  behavior. Chance-corrected, Micro is at 0.10 on the dev set and 0.00 on the suite.
- **Near chance for every Granite run:** cfcolor, ForecastBench, ACOS and the home appliance simulator.
  Strongest relative to chance: BFCL, ARC-Easy, PhishNChips (Micro 0.86), HellaSwag, ARC-Challenge.
- **Where the 3B matches or beats Jev:** RouterBench (0.513 against 0.538) and SGD (0.417 against 0.000).

## Reported Jev scores against our dev runs

On most benchmarks our dev-set Jev scores are within about 0.05 of Jev's scores on the full 0.2 suite as
reported by the leaderboard, so the dev set is a fair proxy. It runs high on MMLU-Pro (0.914 against
0.827) and RAGTruth (0.855 against 0.765, the balance effect above) and low on SGD (0.300 against 0.429),
GPQA (0.680 against 0.783), BANKING77 and CLINC150 (about 0.1).
