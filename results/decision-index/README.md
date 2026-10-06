# Decision Index runs

Full-suite runs scored with the [decision-index kit](https://github.com/apolinario/decision-index).
The edition 0.2 suite was rebuilt locally and is byte-identical to the lab's files. Every row is sent
unmodified through the kit's `http` engine to our `/v1/systemone`; sharding
(`scripts/decision_index_run.py`) only parallelizes the kit's runner. Raw results and traces stay in
`external/decision-index/runs/` (not committed).

Two editions of the scoring exist. **0.2** (kit commit `19ad28e`, our runs' original scoring) and
**0.2.1** (kit commit `87d4650`, the live board since 2026-09-27) read the same suite files; 0.2.1 changes
the area weights, adds "gold" benchmarks weighing 1.2, drops SGD and RouterBench from the index, and
rescores ACOS, ToolRet, BRIGHT, home appliances and RAGTruth's chance level. A complete 0.2 run is also a
complete 0.2.1 run: `score --edition 0.2.1` rescores it without rerunning. Kit 0.2.1 is checked out as a
separate worktree at `external/decision-index-0.2.1`; with `--edition 0.2` it reproduces our original
scores exactly.

## Summary

| Run | Date | Index 0.2 | Index 0.2.1 | Raw 0.2.1 |
| --- | --- | --: | --: | --: |
| Jev 1.13, as reported on the board | – | 51.67 | **57.91** | 68.09 |
| `oso-granite-3b-wide`, prompt `default@3` | 2026-09-26 | 23.58 | **24.92** | 42.44 |
| `oso-granite-micro-cf-v3-litellm`, prompt `default@3` | 2026-10-02 | 15.82 | **16.38** | 35.53 |
| Random baseline (kit dry run) | 2026-09-26 | 1.02 | – | – |

Jev's numbers are from the leaderboard Space's published data (`multimodalart/jev-decision-index`,
`data/index-v2.json` for 0.2 and `data/index.json` for 0.2.1; copies in `runs/decision-index-board/`).
They are scored with the same scorers on the same edition, so they compare directly with ours. Check
TypeSafe's and the board's terms before publishing these comparisons.

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

## 2026-10-02 — Granite 4.0 Micro, prompt `default@3`

- Model: `ibm-granite/granite-4.0-h-micro` (3B) through OpenRouter, pinned to Cloudflare, via the local
  LiteLLM gateway (profile `oso-granite-micro-cf-v3-litellm`). OpenRouter caps `top_logprobs` at 20, so
  the profile allows 20 options per choice.
- Coverage: 136,339 answered, **15,061 unsupported** (over 20 options), 76 errors; `complete: true`.
  Wall time 100.7 min (48 shards, 3 server processes).
- The option cap costs it four benchmarks outright: API-Bank, BANKING77, CLINC150+OOS and POP909-CL get
  0% coverage, and ChessBench 21%. If Micro had matched the 3B on API-Bank, BANKING77 and CLINC150, that
  would be worth about 3.3 points of the 0.2 index.

## Granite 3B and Micro against Jev, edition 0.2.1

Skill is chance-corrected and coverage-adjusted: 0 = random guessing, 1 = perfect, unanswered = wrong.
Native is the benchmark's own metric on the requests answered. ★ = gold benchmark (weight 1.2 in its
area).

| Area | Weight | Granite 3B | Micro | Jev |
| --- | --: | --: | --: | --: |
| Knowledge & Reasoning | 25.8% | 0.153 | 0.128 | 0.514 |
| Language Understanding | 25.8% | 0.272 | 0.199 | 0.620 |
| Retrieval & Classification | 20.0% | 0.256 | 0.073 | 0.554 |
| Tools & Automation | 18.3% | 0.390 | 0.275 | 0.751 |
| Arts & Human Taste | 10.0% | 0.167 | 0.143 | 0.377 |

| # | Benchmark | Granite 3B native | Jev native | Granite 3B skill | Micro skill | Jev skill |
| --: | --- | --: | --: | --: | --: | --: |
| | **Arts & Human Taste** | | | | | |
| 20 | BPoMP | 0.736 | 0.906 | 0.467 | 0.412 | 0.818 |
| 21 | Humicroedit | 0.609 | 0.619 | 0.218 | 0.064 | 0.237 |
| 22 | POP909-CL | 0.005 | 0.181 | 0.000 | 0.000 | 0.159 |
| 23 | cfcolor | 0.511 | 0.647 | 0.000 | 0.019 | 0.288 |
| 48 | ForecastBench ★ (Brier) | 0.353 | 0.174 | 0.000 | 0.000 | 0.306 |
| 50 | Habermas Machine | 0.434 | 0.459 | 0.178 | 0.187 | 0.215 |
| 64 | New Yorker caption matching | 0.472 | 0.701 | 0.340 | 0.347 | 0.626 |
| | **Knowledge & Reasoning** | | | | | |
| 25 | GPQA Diamond ★ | 0.311 | 0.783 | 0.082 | 0.129 | 0.714 |
| 30 | GSM8K | 0.343 | 0.799 | 0.205 | 0.153 | 0.756 |
| 31 | ChessBench | 0.101 | 0.172 | 0.021 | 0.000 | 0.098 |
| 32 | MuSR | 0.582 | 0.661 | 0.336 | 0.216 | 0.461 |
| 33 | SATA-Bench | 0.228 | 0.264 | 0.218 | 0.173 | 0.254 |
| 43 | CRUXEval | 0.346 | 0.730 | 0.000 | 0.012 | 0.571 |
| 44 | CLadder | 0.567 | 0.726 | 0.135 | 0.096 | 0.453 |
| 45 | HLE ★ | 0.124 | 0.204 | 0.000 | 0.000 | 0.047 |
| 57 | MMLU-Pro ★ | 0.338 | 0.827 | 0.255 | 0.226 | 0.805 |
| 58 | BBH fixed-option tasks ★ | 0.503 | 0.929 | 0.280 | 0.256 | 0.897 |
| | **Language Understanding** | | | | | |
| 11 | ContractNLI | 0.470 | 0.717 | 0.233 | 0.166 | 0.591 |
| 12 | ANLI ★ | 0.470 | 0.748 | 0.206 | 0.241 | 0.622 |
| 28 | WinoGrande ★ | 0.598 | 0.919 | 0.197 | 0.228 | 0.839 |
| 29 | HellaSwag ★ | 0.761 | 0.945 | 0.681 | 0.579 | 0.927 |
| 38 | ACOS | 0.181 | 0.295 | 0.154 | 0.031 | 0.273 |
| 39 | FinEntity | 0.663 | 0.870 | 0.505 | 0.206 | 0.808 |
| 40 | iSarcasmEval | – | 0.505 | 0.116 | 0.052 | 0.363 |
| 41 | VAST | 0.527 | 0.646 | 0.290 | 0.097 | 0.469 |
| 42 | NLI4CT | 0.633 | 0.841 | 0.286 | 0.304 | 0.690 |
| 59 | RAGTruth | 0.288 | 0.765 | 0.000 | 0.002 | 0.513 |
| | **Retrieval & Classification** | | | | | |
| 4 | BANKING77 ★ | 0.415 | 0.797 | 0.407 | 0.000 | 0.795 |
| 5 | CLINC150+OOS ★ | 0.438 | 0.893 | 0.435 | 0.000 | 0.892 |
| 36 | BRIGHT ★ | 0.370 | 0.475 | 0.287 | 0.238 | 0.406 |
| 37 | Amazon ESCI | 0.292 | 0.552 | 0.112 | 0.093 | 0.438 |
| 56 | PhishNChips | 0.529 | 0.625 | 0.058 | 0.010 | 0.251 |
| 61 | HoVer | 0.581 | 0.729 | 0.162 | 0.091 | 0.457 |
| | **Tools & Automation** | | | | | |
| 1 | BFCL ★ | 0.842 | 0.958 | 0.787 | 0.583 | 0.943 |
| 2 | ToolRet | 0.585 | 0.653 | 0.520 | 0.509 | 0.599 |
| 3 | API-Bank ★ | 0.283 | 0.882 | 0.270 | 0.000 | 0.880 |
| 9 | Home appliance simulator | 0.011 | 0.523 | 0.011 | 0.000 | 0.523 |
| 62 | When2Call MCQ | 0.480 | 0.810 | 0.307 | 0.276 | 0.746 |
| | **Shown but not counted in the index** | | | | | |
| 6 | RouterBench | 0.796 | 0.799 | 0.522 | 0.267 | 0.527 |
| 10 | SGD/SGD-X | 0.578 | 0.429 | 0.298 | 0.218 | – |
| 24 | MMLU | 0.611 | 0.917 | 0.482 | 0.449 | 0.893 |
| 26 | ARC-Easy | 0.926 | 0.993 | 0.901 | 0.897 | – |
| 27 | ARC-Challenge | 0.809 | 0.978 | 0.745 | 0.738 | – |
| 34 | SimpleBench | 0.300 | – | 0.160 | 0.000 | – |

Where Granite 3B is closest to Jev: RouterBench, Humicroedit, SATA-Bench, HLE, Habermas Machine and
ToolRet. The largest gaps (0.5 or more) are WinoGrande, GPQA, BBH, API-Bank, CRUXEval, GSM8K, MMLU-Pro,
RAGTruth and the home appliance simulator. Under 0.2, Granite 3B was ahead of Jev only on SGD (which 0.2.1
no longer counts).

## Adapter runs on three benchmarks

Granite Switch's built-in adapters were run on the full suite rows of RAGTruth, HoVer and NLI4CT (12,200
requests) and scored with the kit. These are experiments, not a submission: choosing an engine per
benchmark is not allowed under the house rules. Details in [`../adapters.md`](../adapters.md).
