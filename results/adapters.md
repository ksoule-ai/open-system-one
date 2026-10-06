# Granite Switch adapters as decision engines (October 2026)

Can Granite Switch's built-in adapters answer Decision Index questions better than the plain System One
prompt (`default@3`) on the same model? We tried the hallucination-detection, factuality-detection,
guardian-core (groundedness) and answerability adapters on the benchmarks where they plausibly apply:
RAGTruth, HoVer and NLI4CT ("is this text supported by that source?"), then ANLI and ContractNLI.

**Short answer:** no adapter is better across the board. Each wins on one benchmark and loses or ties
on others, and all stay well below Jev. The best per benchmark differ (RAGTruth: factuality on the suite;
HoVer: guardian on the dev set; NLI4CT: the plain prompt), and choosing an engine per benchmark is not
allowed under the Decision Index house rules, so using them would need a router that decides from the
request itself ([`../proposals/adapter-router.md`](../proposals/adapter-router.md)).

## Setup

- Model: `ibm-granite/granite-switch-4.1-3b-preview` on our Hugging Face endpoint (vLLM), through the
  LiteLLM gateway. The adapters are built into the model and selected per request.
- All calls go through Mellea 0.8.0 (`OpenAIBackend(load_embedded_adapters=True)`), which downloads the
  adapters' configs from the model's Hugging Face repo, rewrites the input (sentence markers, the
  adapter's instruction message, a JSON output schema, `chat_template_kwargs.adapter_name`) and parses the
  output. Mellea's adapter functions return only the parsed answer; to get probabilities we call the same
  path one level down (`mfuncs.act` on an `Intrinsic`) with `logprobs` and read the raw response. See
  [`../docs/logprob-decisions.md`](../docs/logprob-decisions.md#reading-a-yesno-probability-from-a-granite-switch-adapter).
- Adapter inputs, per benchmark: the response to check is the last assistant turn and the source is a
  document; the user turn is either the **default** (RAGTruth: the question or instruction the response
  answered; HoVer / NLI4CT: the question's first instruction line) or the **generic** "Tell me about the
  documents.". RAGTruth's inputs are parsed from the request's own prompt (identical to the source
  records on all 100 dev rows).
- Mapping to answers: a "hallucinated / ungrounded / factually incorrect" verdict means RAGTruth
  hallucinated, HoVer `NOT_SUPPORTED`, NLI4CT `Contradiction`. The hallucination adapter rates each
  sentence `faithful`, `partial`, `unfaithful` or `NA`; a response counts as hallucinated when any
  sentence is `unfaithful` or `partial` (the rule in the adapter's own README). Factuality and guardian
  answer yes/no; we use P("yes") from the answer token.
- Scores: the Decision Index's metrics (RAGTruth F1 on the hallucinated class, HoVer accuracy, NLI4CT
  macro-F1), then chance-corrected. Dev chance is measured on the dev rows
  ([`../scripts/dev_chance.py`](../scripts/dev_chance.py)): RAGTruth 0.667 (always answering
  "hallucinated" on the 50/50 dev set), HoVer 0.506, NLI4CT 0.490. Suite chance uses edition 0.2.1's
  levels: 0.518, 0.5, 0.486.

## Dev set (100 samples per benchmark)

Chance-corrected skill (native score in brackets).

| Engine | RAGTruth | HoVer | NLI4CT |
| --- | --: | --: | --: |
| Hallucination adapter, default user turn | 0.000 (0.608) | 0.089 (0.550) | 0.192 (0.588) |
| Hallucination adapter, generic user turn | 0.000 (0.627) | 0.190 (0.600) | 0.210 (0.597) |
| Factuality adapter, generic user turn | 0.000 (0.621) | 0.170 (0.590) | 0.000 (0.435) |
| Factuality adapter, default user turn | 0.085 (0.695) | 0.190 (0.600) | 0.000 (0.429) |
| **Guardian groundedness**, default user turn | 0.000 (0.514) | **0.393 (0.700)** | 0.024 (0.502) |
| Granite 3B wide, `default@3` | 0.000 (0.301) | 0.170 (0.590) | **0.300 (0.643)** |
| Granite 4.0 Micro, `default@3` | 0.102 (0.701) | 0.069 (0.540) | 0.284 (0.635) |
| Jev 1.13 | 0.564 (0.855) | 0.433 (0.720) | 0.682 (0.838) |

Observations:

- **Flag rates explain most of it.** On 50/50 sets the factuality adapter (generic user turn) called 77
  of 100 HoVer claims and 91 of 100 NLI4CT statements "factually incorrect"; guardian flagged only 24 of 100 RAGTruth
  responses but 84 of 100 NLI4CT statements. RAGTruth on the dev set rewards flagging (half the responses
  are hallucinated), so raw F1 there overstates heavy flaggers; chance correction removes that.
- **The hallucination adapter's flag rule matters.** On RAGTruth (default user turn) the response-level
  F1 was 0.579 with `unfaithful` only, 0.608 with `unfaithful` + `partial`, 0.674 counting `NA` too
  (`NA` sentences such as "Unable to answer based on given passages." then count as hallucinated). The
  best rule differs per benchmark; we fixed `unfaithful` + `partial`.
- **The user turn matters little.** Switching the hallucination adapter to the generic user turn changed
  1 of 100 HoVer and 3 of 100 NLI4CT verdicts; for factuality, the real question helped on RAGTruth only.
- **Explanations are unreliable.** The hallucination adapter's per-sentence explanations often describe
  a different sentence or misread the document, even when the verdict is right.
- **Some NLI4CT statements carry an irrelevant extra sentence** after the claim being tested. Factuality's
  definition ("even if only a small portion… is unsupported") then calls the whole statement incorrect,
  which may explain much of its contradiction bias. How many statements do this has not been counted.
- **Guardian's invocation text.** The chat template renders guardian's activation as
  `<|guardian-core|>guardian>`: the adapter token replaces the `<` token of `<guardian>` (tokens `<`,
  `guard`, `ian`, `>`). Sending the prompt as raw text without `guardian>` changed 8 of 300 verdicts, within
  noise ([`../spikes/guardian_raw_dev.py`](../spikes/guardian_raw_dev.py)).

## Full suite (RAGTruth 2,700, HoVer 4,000, NLI4CT 5,500 requests)

Scored by the kit; skill with 0.2.1 chance levels (native in brackets). Default user turn.

| Engine | RAGTruth | HoVer | NLI4CT | Median latency under load |
| --- | --: | --: | --: | --- |
| Factuality adapter | **0.256 (0.641)** | 0.196 (0.598) | 0.000 (0.450) | 1.9 s |
| Hallucination adapter | 0.000 (0.506) | **0.257 (0.630)** | 0.265 (0.622) | 59 s / 13 s / 10 s |
| Granite 3B wide, `default@3` | 0.000 (0.288) | 0.162 (0.581) | 0.286 (0.633) | – |
| Granite 4.0 Micro, `default@3` | 0.002 (0.519) | 0.091 (0.545) | **0.304 (0.642)** | – |
| Jev 1.13 (board) | 0.513 (0.765) | 0.457 (0.729) | 0.690 (0.841) | – |

- Factuality: all 12,200 answered, 4 minutes on 4 × 24 concurrent calls. It called 5,335 of 5,500 NLI4CT
  statements incorrect.
- Hallucination: 12,158 answered; 42 failed with malformed JSON that repeats at temperature 0 and count as
  wrong. Long RAGTruth responses needed a 10-minute timeout under load. It is a LoRA adapter (the whole
  prompt is reprocessed) and writes an explanation per sentence; factuality and guardian are aLoRA adapters
  that reuse the cached prompt and answer with one short label.
- The hallucination adapter flags too little on the suite's RAGTruth (612 of 2,671), so its F1 falls just
  under always answering "hallucinated"; on the dev set it had looked better.
- Using factuality for RAGTruth and the hallucination adapter for HoVer, with `default@3` elsewhere, would
  add roughly 0.8 points to the 3B's 0.2.1 index (24.92), by our estimate; including NLI4CT would cancel it.
- Guardian groundedness has not been run on the suite.

## ANLI and ContractNLI: mapping yes/no probabilities onto three classes

Both are 3-way (entailment / neutral or not mentioned / contradiction). We ran factuality ("Tell me about
the documents.", hypothesis as the assistant turn, premise or contract as the document) and answerability
(user question "Is the following statement true? <hypothesis>", premise or contract as the document) on
all dev items (ANLI 100, ContractNLI 1,700 = 17 hypotheses × 100 contracts) and recorded P(yes).
3-class macro-F1:

| Rule | ANLI | ContractNLI |
| --- | --: | --: |
| Factuality, fixed thresholds (entailment > 0.75, contradiction < 0.25) | 0.245 | 0.234 |
| Factuality, best thresholds (2-fold cross-validated by contract) | 0.358 | 0.494 |
| Answerability, fixed thresholds | 0.372 | 0.353 |
| Answerability, best thresholds (cross-validated) | 0.354 | 0.518 |
| **Both: answerability decides neutral, then factuality splits the rest** (cross-validated) | 0.391 | **0.629** |
| Granite 3B wide, `default@3` | **0.466** | 0.446 |
| Granite 4.0 Micro, `default@3` | 0.459 | 0.434 |
| Jev 1.13 | 0.702 | 0.722 |

On ContractNLI, answerability separates "entailed" (median P(answerable) 0.98) from "not mentioned"
(0.004), but contradictions spread across the range, because a contradiction is also answerable. The
combined rule (answerability below about 0.04 → neutral; else P(not incorrect) ≥ about 0.14 → entailment,
otherwise contradiction) gets ContractNLI halfway from `default@3` to Jev. The thresholds are fitted on the
dev set, and ContractNLI reuses the same 17 hypothesis texts for every contract, so the 0.629 may be
optimistic. On ANLI nothing beats `default@3`.

## Where things are

| What | Script | Results (gitignored) |
| --- | --- | --- |
| Hallucination adapter, RAGTruth dev | `spikes/ragtruth_hallucination.py` | `runs/spikes/ragtruth-hallucination*/` |
| Hallucination adapter, HoVer / NLI4CT dev | `spikes/hallucination_adapter_dev.py` | `runs/spikes/hallucination-adapter-dev*/` |
| Factuality adapter, three benchmarks dev | `spikes/factuality_adapter_dev.py` | `runs/spikes/factuality-adapter-dev*/` |
| Any adapter on suite or dev rows, kit-format results | `spikes/factuality_adapter_suite.py --adapter …` | `external/decision-index/runs/{factuality,hallucination}-adapter-suite/`, `runs/di-dev/guardian-groundedness-dev/` |
| Guardian as raw text (invocation variants) | `spikes/guardian_raw_dev.py` | `runs/di-dev/guardian-raw-*/` |
| ANLI / ContractNLI with answerability and factuality | `spikes/nli_adapters_dev.py`, `spikes/nli_adapters_analyze.py` | `runs/spikes/nli-adapters-dev/` |
| Re-send one sample to several engines, with gateway log ids | `spikes/rerun_sample.py` | `runs/spikes/rerun-<key>/` |

All adapter calls are tagged in the LiteLLM gateway's logs (`<adapter>-suite`, `nli-<adapter>`,
`factuality-dev`, …). The gateway logs show the messages and a separate `documents` field; the chat
template that combines them is in the model's repo (`chat_template.jinja`; a copy is in
`runs/spikes/granite-switch-4.1-3b-preview.chat_template.jinja`).
