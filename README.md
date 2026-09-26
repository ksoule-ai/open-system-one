# open-system-one

Build your own **System One** decision model with [Mellea](https://github.com/generative-computing/mellea):
a drop-in, Jev-compatible API (`POST /v1/systemone`) over any chat LLM — an OpenRouter model, your own
Hugging Face Inference Endpoint, or a local Ollama model.

Each question's options are shown to the model under single-token labels. The model produces one answer
token, and each option's probability is its label token's probability divided by the sum over all label
tokens. The prompt structure is fully configurable, and a lightweight eval compares any configuration
against Jev on the same requests.

Architecture and status: [`CLAUDE.md`](./CLAUDE.md). API: [`api-schema.md`](./api-schema.md). Eval:
[`eval-design.md`](./eval-design.md).

Not affiliated with TypeSafe AI.

## Quickstart

```bash
uv sync
cp .env.example .env                                   # keys, endpoint URL, Ollama host

uv run --env-file .env open-system-one serve --port 8080

# Any TypeSafe SDK client works against it:
TYPESAFE_BASE_URL=http://localhost:8080 TYPESAFE_API_KEY=<your OSO_API_KEY> python your_app.py

# Compare a profile against Jev on the sanity set
uv run --env-file .env open-system-one eval --cases evals/sanity-v0.jsonl \
    --target oso-latest --baseline jev-1.13 --split tune
```

## Status

Planning. See `CLAUDE.md` > Setup status / next steps.
