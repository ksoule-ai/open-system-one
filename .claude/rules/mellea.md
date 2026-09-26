# Mellea conventions

Mellea is a fast-moving 0.x library; general/training-data knowledge of its API is often stale or wrong.
Treat the following as the source of truth, in order:

1. **The installed source** at `.venv/lib/python3.*/site-packages/mellea/`. When unsure about an import
   path, signature, or model option, read it rather than guessing. It is exactly the pinned version.
2. **The official docs:** https://docs.mellea.ai (repo: https://github.com/generative-computing/mellea).

Rules:
- Do not invent Mellea APIs. If a symbol isn't in the installed source, it doesn't exist in this version.
- The pinned version is `mellea==0.8.0`. Don't upgrade it without updating this project's docs.
- **All model calls go through Mellea.** If Mellea lacks something (e.g. a way to read logprobs), add a
  thin, documented Mellea-level extension or propose an upstream change — never a side call around it.
- Known 0.8.0 details (verify in source before relying on them):
  - `OpenAIBackend` stores the full non-streaming response in `mot.raw.response`; the streaming merge sets
    `logprobs` to `None`. Use non-streaming calls when logprobs are needed.
  - `OllamaModelBackend` passes `logprobs` / `top_logprobs` model options through to Ollama.
  - `LiteLLMBackend` stores returned logprobs in the output thunk's metadata.
