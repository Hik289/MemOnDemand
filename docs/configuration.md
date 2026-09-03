# Configuration

MemOnDemand reads configuration from environment variables. By default it loads a
local `.env` file from `MEMONDEMAND_REPO_ROOT` or the current working directory.

## Environment Files

```bash
cp .env.example .env
```

Set `MEMONDEMAND_ENV_FILE` when you want to keep configuration outside the project
directory:

```bash
export MEMONDEMAND_ENV_FILE=/secure/path/memondemand.env
```

## API Settings

| Variable | Required for | Notes |
| --- | --- | --- |
| `MEMONDEMAND_API_BASE_URL` | Model-backed answering and judging | General chat-completions base URL. |
| `MEMONDEMAND_API_KEY` | Model-backed answering and judging | API key for your model gateway. |
| `MEMONDEMAND_API_MODEL` | Model-backed answering and judging | Chat model name or deployment alias. |
| `MEMONDEMAND_API_TOKEN_PARAMETER` | Model-backed answering and judging | `max_tokens` or `max_completion_tokens`. |
| `MEMONDEMAND_API_SEND_TEMPERATURE` | Model-backed answering and judging | Include or omit `temperature` in requests. |
| `MEMONDEMAND_API_INPUT_COST_PER_MILLION` | Token accounting | Input price used for cost estimates; defaults to `0`. |
| `MEMONDEMAND_API_OUTPUT_COST_PER_MILLION` | Token accounting | Output price used for cost estimates; defaults to `0`. |
| `MEMONDEMAND_EMBED_API_BASE_URL` | API-backed embeddings | Optional general embedding base URL. |
| `MEMONDEMAND_EMBED_API_KEY` | API-backed embeddings | Optional embedding API key. |
| `MEMONDEMAND_EMBED_API_MODEL` | API-backed embeddings | Optional embedding model or deployment alias. |
| `MEMONDEMAND_EMBED_DIM` | API-backed embeddings | Optional embedding dimension override. |

MemOnDemand exposes one public alias, `general`. Chat requests use
`POST /chat/completions`; embedding requests use `POST /embeddings`. Both send a
bearer credential and JSON payload. Custom tenancy, routing, and audit behavior stays
behind the configured gateway instead of entering application code.

## Retrieval Settings

| Variable | Default | Notes |
| --- | --- | --- |
| `MEMONDEMAND_EMBED_BACKEND` | `minilm` | Local embedding backend for development. |
| `MEMONDEMAND_L0_RETRIEVAL` | `bm25` | L0 candidate retrieval mode. |
| `MEMONDEMAND_SKIP_L0_EMBED` | `0` | Set to `1` to avoid embedding all L0 nodes. |
| `MEMONDEMAND_INDEX_CACHE_DIR` | unset | Optional cache directory for retrieval indexes. |
| `MEMONDEMAND_ANSWER_MODE` | runner default | Use `detailed_truncated` for compact evidence contexts. |

## Secret Policy

Do not commit `.env`, generated JSONL answers, parquet manifests, logs, caches,
or gateway keys. `.gitignore` already excludes common local artifacts.
