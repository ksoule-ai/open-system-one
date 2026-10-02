# Decision Index dev set

A development set for tuning and checking engines against the Decision Index **without touching the
scored suite**. Every sample comes from data that Decision Index 0.2 does not score, keeps its original
upstream format, and is converted into Decision Index requests by the kit's own adapters, unmodified.

- 34 benchmarks, 100 samples each (fewer only where fewer exist: see *Shortfalls*).
- Binary benchmarks are balanced 50/50 on their label, where a sample has a single label.
- Every sample records exactly where it came from (`raw/<benchmark>/provenance.jsonl`).
- Built by [`scripts/decision_index_dev.py`](../../scripts/decision_index_dev.py) with the kit at commit
  `19ad28e` (edition 0.2), against the locally rebuilt, hash-verified 0.2 suite.

The scored suite itself is never used to build or tune anything here. Tuning on this set is allowed;
the Decision Index house rule (one fixed rendering for every benchmark) still applies to whatever
prompt is finally run on the real suite.

## Layout

```
evals/decision-index-dev/
├── README.md
├── manifest.json                  # per benchmark: samples, requests, source kinds, label balance
├── raw/<NN>-<benchmark>/          # the source records, unchanged, in the source's own format
│   ├── <source file subset>       # e.g. a parquet/csv/jsonl/tsv/json subset of the upstream file
│   └── provenance.jsonl           # one line per sample (below)
├── converted/<NN>-<benchmark>.jsonl.gz   # Decision Index requests produced by the kit's adapters
├── handwritten/phishnchips_legitimate.csv  # the only hand-written data (50 legitimate emails)
└── dev-rows.jsonl.gz              # all converted requests in one file (local only, gitignored)
```

Raw files are subsets of the upstream files with the same columns, keys and values. Parquet, JSON,
JSONL and TSV subsets keep the upstream structure; CSV subsets are re-serialized with Python's `csv`
module, so quoting and line endings (CRLF) can differ byte-wise from the upstream file.

`provenance.jsonl` fields: `sample_id`, `key` (the source record id), `benchmark`, `catalog_id`,
`source_kind`, `source` (dataset, revision, file, split, row index / record id, notes), `label` (for
balanced binary benchmarks), `raw_file`, and `converted_run_ids` (the request rows it became).

Converted rows are exactly the adapters' output plus:
- `_evaluation`: the kit freeze step's formula (catalog id, dataset, group id, track, payload hash…),
  with run ids prefixed **`dev:`** so they can never be confused with suite rows;
- `_dev`: `sample_id`, `key`, `source_kind`.

## Source kinds

| Kind | Meaning |
| --- | --- |
| `heldout-split` | A train / dev / validation split the suite does not use. |
| `heldout-category` | A published category the suite's adapter does not read (BFCL `multiple`, `parallel_multiple`). |
| `heldout-window` | Same files, outside the suite's date window (ForecastBench resolutions before 2026-07-01). |
| `suite-unselected` | **Rows from the same test pool that Decision Index 0.2 does not score.** Not a true held-out split: they come from the suite's own source files, and some were scored in edition 0.1 (ToolRet, BRIGHT before the 0.2 cut). Re-check against any later edition (the kit is now at 0.2.1). |
| `generator-dev` | The kit generator's own dev rows (home appliance simulator). |
| `handwritten` | Written for this dev set (50 legitimate PhishNChips emails, marked `datasource: handwritten_dev_v1`). |

## Benchmarks

| # | Benchmark | Samples | Requests | Source | 50/50 |
| --: | --- | --: | --: | --- | --- |
| 1 | BFCL | 100 | 100 | heldout-category: gorilla-llm/gorilla BFCL v3 @916260d (categories multiple + parallel_multiple) | – |
| 2 | ToolRet | 100 | 100 | suite-unselected: mangopy/ToolRet-Queries @b8c76ad + ToolRet-Tools @e06c38c (unscored in 0.2) | – |
| 4 | BANKING77 | 100 | 100 | heldout-split: PolyAI BANKING77 (train) | – |
| 5 | CLINC150+OOS | 100 | 100 | heldout-split: clinc/oos-eval data_full.json (val); heldout-split: clinc/oos-eval data_full.json (oos_val) | – |
| 6 | RouterBench | 100 | 200 | suite-unselected: withmartian/routerbench @7840214 (unscored in 0.2) | – |
| 9 | Home appliance simulator | 80 | 80 | generator-dev: kit home-appliance generator (purpose-built benchmark) | – |
| 10 | SGD/SGD-X | 100 | 100 | heldout-split: google-research-datasets/dstc8-schema-guided-dialogue @e852981 (dev) | – |
| 11 | ContractNLI | 100 | 100 | heldout-split: stanfordnlp/contract-nli @eced652 (dev); heldout-split: stanfordnlp/contract-nli @eced652 (train) | – |
| 12 | ANLI | 100 | 100 | heldout-split: facebook/anli (dev_r1); heldout-split: facebook/anli (dev_r2); heldout-split: facebook/anli (dev_r3) | – |
| 20 | BPoMP | 100 | 100 | suite-unselected: BPoMP (zenodo 7299879) (unscored in 0.2) | ✓ 50 option_0 / 50 option_1 |
| 21 | Humicroedit | 100 | 100 | heldout-split: SemEval-2020 Task 7 (Humicroedit) (dev) | ✓ 50 1 / 50 2 |
| 22 | POP909-CL | 100 | 100 | suite-unselected: POP909-CL @be90943 (unscored in 0.2) | – |
| 23 | cfcolor | 100 | 100 | suite-unselected: cfcolor (dgp.toronto.edu) (unscored in 0.2) | ✓ 50 A / 50 B |
| 24 | MMLU | 100 | 100 | heldout-split: cais/mmlu (validation) | – |
| 25 | GPQA Diamond (gitignored) | 100 | 100 | heldout-split: idavidrein/gpqa dataset.zip (main (not in Diamond)) | – |
| 26 | ARC-Easy | 100 | 100 | heldout-split: allenai/ai2_arc (validation) | – |
| 27 | ARC-Challenge | 100 | 100 | heldout-split: allenai/ai2_arc (validation) | – |
| 28 | WinoGrande | 100 | 100 | heldout-split: allenai/winogrande (train) | ✓ 50 1 / 50 2 |
| 29 | HellaSwag | 100 | 100 | heldout-split: Rowan/hellaswag (train) | – |
| 30 | GSM8K | 100 | 200 | heldout-split: openai/gsm8k (train) | – |
| 31 | ChessBench | 100 | 100 | suite-unselected: ChessBench test action-value bag (searchless_chess @90ae0e6) (unscored in 0.2) | – |
| 36 | BRIGHT | 100 | 100 | suite-unselected: xlangai/BRIGHT @3066d29 (unscored in 0.2) | – |
| 37 | Amazon ESCI | 100 | 100 | heldout-split: amazon-science/esci-data @7916cdf (train) | – |
| 38 | ACOS | 100 | 390 | heldout-split: NUSTM/ACOS @45d179a (dev) | – |
| 40 | iSarcasmEval | 100 | 100 | heldout-split: iabufarha/iSarcasmEval @dfc708b (train) | ✓ 33 0 / 33 1 |
| 41 | VAST | 100 | 100 | heldout-split: emilyallaway/zero-shot-stance (VAST) @e7c4775 (dev) | – |
| 42 | NLI4CT | 100 | 100 | heldout-split: ai-systems/Task-2-SemEval-2024 @7f32fa6 (practice (dev)) | ✓ 50 Contradiction / 50 Entailment |
| 44 | CLadder | 100 | 100 | suite-unselected: CLadder v1 (causalNLP/cladder @3d2d116) (unscored in 0.2) | ✓ 50 A / 50 B |
| 48 | ForecastBench | 100 | 100 | heldout-window: forecastingresearch/forecastbench-datasets @da48cfb (resolved before 2026-07-01) | ✓ 50 no / 50 yes |
| 56 | PhishNChips phishing decisions | 100 | 100 | handwritten: handwritten for this dev set; heldout-split: AreLit/PhishNChips (validation (not in core_emails.csv)) | ✓ 50 legitimate / 50 phishing |
| 57 | MMLU-Pro | 70 | 70 | heldout-split: TIGER-Lab/MMLU-Pro (validation) | – |
| 59 | RAGTruth response-level hallucination (gitignored) | 100 | 100 | heldout-split: ParticleMedia/RAGTruth @c103204 (train) | ✓ 50 False / 50 True |
| 61 | HoVer claim verification | 100 | 100 | heldout-split: hover-nlp/hover @39b8469 (train) | ✓ 50 NOT_SUPPORTED / 50 SUPPORTED |
| 64 | New Yorker caption matching | 100 | 100 | heldout-split: jmhessel/newyorker_caption_contest (validation) | – |
| | **Total** | **3350** | **3840** | | |

## How conversion works (and every place it needed help)

Held-out records are written, in their original format, into the input slot the adapter reads, inside a
separate workspace (`work-dev/` next to the kit); everything else the adapter needs (category lists,
schemas, trial documents, the HoVer Wikipedia DB, corpora) is linked unchanged from the real build. The
adapter then runs as is. Suite-unselected rows are taken straight from the adapters' output for the real
build. Where a slot needed more than a straight copy, it is listed here:

| Benchmark | What the adapter's input copy needed |
| --- | --- |
| RAGTruth, Amazon ESCI | The adapter keeps only `split == "test"`; the input copy carries `test`. Raw files keep the true split (`train`). |
| iSarcasmEval | Train files use `tweet` where test files use `text`; columns mapped in the input copy. Task C (pairs) is not covered: there is no train-side pairs file. |
| ACOS | The category inventory is built from all of ACOS's files, so dev reviews are *added* as `*_devsample_test.tsv` instead of replacing the test file; questions are identical to the suite's. |
| BFCL | Held-out categories are placed in the slot of a category the adapter reads; the true category is in provenance. |
| GPQA | The adapter asserts exactly 198 rows; the slot holds 198 held-out (main, not Diamond) questions and only the 100 samples are kept. |
| ForecastBench | The adapter's date window is widened **in memory** to the period before the suite window (constants only; kit code unchanged). |
| PhishNChips | The adapter checks its input file's hash; the check is satisfied **in memory** for the dev copy (kit code unchanged). |
| ContractNLI | The dev split has 61 documents; 39 more come from train. |
| GSM8K | Problems whose answer gives the adapter's distractor pool fewer than 9 wrong answers are skipped (the adapter would fail on them); none were in the test split. |
| MMLU, ARC, WinoGrande, HellaSwag, GSM8K | Held-out items whose text also appears in the suite's source file are excluded. |
| All | Any candidate whose converted request is identical (payload hash) to a scored suite request is excluded. Verified: **0** of the converted requests match the suite. |

## Shortfalls and balance

- **Home appliance simulator: 80** (all of the generator's dev rows). **MMLU-Pro: 70** (its whole validation
  split). No hand-written top-ups.
- **Balanced 50/50:** BPoMP, CLadder, cfcolor, Humicroedit, WinoGrande, NLI4CT, ForecastBench, PhishNChips,
  RAGTruth, HoVer, iSarcasmEval task A (per language).
- **Not balanceable** (many yes/no questions per sample, mostly "no"): ACOS, BFCL, ToolRet, BRIGHT. Their
  natural label rates are kept.
- Some benchmarks expand one sample into several requests, exactly as in the suite: ACOS (category chunks
  of 64), GSM8K (4- and 10-choice), RouterBench (0- and 5-shot).

## Not covered

- **SATA-Bench**: no held-out data exists (the HF release's "extra" items are the suite's questions with
  different markup).
- **Multiple-choice benchmarks without held-out data**, which would need hand-written samples: API-Bank,
  MuSR, SimpleBench, FinEntity, HLE, CRUXEval, BBH, Habermas Machine, When2Call.

## Licensing

Raw and converted files for **RAGTruth** (Yelp text: no redistribution) and **GPQA** (authors ask that
questions not be posted) are written locally but **gitignored**; rebuild them with the script. Several
committed benchmarks have terms that are unclear rather than restrictive — ACOS and VAST (none stated),
Humicroedit and NLI4CT (SemEval task terms), POP909 (research use), cfcolor, ToolRet, BPoMP, ContractNLI,
RouterBench, CLadder, ForecastBench (see their repositories) — and PhishNChips is cleared for academic use
with attribution. Check those before pushing this repository anywhere public.

## Rebuild

```bash
K=~/Documents/dev/open-system-one/decision-index     # the kit, with work/ and suite-0.2/ built
$K/.venv/bin/python scripts/decision_index_dev.py                 # everything
$K/.venv/bin/python scripts/decision_index_dev.py --only 59 61    # some benchmarks
```

Sampling is deterministic (seed `20261002`); the build re-downloads only the held-out files it needs from
their pinned sources.
