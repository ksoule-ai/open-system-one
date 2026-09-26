# Prompt rules

- **Prompts live only in `configs/prompts/*.yaml`.** No prompt text, label schemes, or answer primers in
  Python code; code renders whatever the config says.
- Every prompt change is a **new version** of the config. Traces, eval runs, and responses record the
  prompt name, version, and content hash.
- **Tune on `tune` cases only.** `holdout` cases are run once per final version and never used to choose
  between versions.
- Keep the shared prefix shared: anything that differs per question goes after the state, so the batching
  strategies can reuse the cache.
- Never render question keys into the prompt.
- One prompt config per profile applies to every request; no per-case or per-category prompt switching.
