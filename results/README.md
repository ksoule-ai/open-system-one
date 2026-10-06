# Results

| Page | What it covers | Headline |
| --- | --- | --- |
| [`decision-index/`](decision-index/README.md) | Full Decision Index suite runs, rescored for edition 0.2.1, against Jev's reported scores | Granite 3B (`default@3`) 24.92, Granite 4.0 Micro 16.38, Jev 1.13 57.91 |
| [`dev-set.md`](dev-set.md) | Every dev-set run, chance-corrected; prompt variants; how the dev set compares with the suite | Granite 3B at about half of Jev's mean skill; `default@4` worse than `default@3` |
| [`adapters.md`](adapters.md) | Granite Switch adapters (hallucination, factuality, guardian, answerability) on RAGTruth, HoVer, NLI4CT, ANLI, ContractNLI | Each adapter wins on one benchmark and loses on others; none close to Jev |

How the numbers are produced, and how to use the method yourself:
[`../docs/logprob-decisions.md`](../docs/logprob-decisions.md).

Raw results, traces and the leaderboard's data files are gitignored (`runs/`, `external/`). Jev numbers
come from our own calls through OpenRouter (dev set) or the leaderboard's published data (full suite);
check TypeSafe's and OpenRouter's terms before publishing comparisons. Several benchmarks' data may not be
redistributed (see the dev set's licensing notes), so these pages report scores only.
