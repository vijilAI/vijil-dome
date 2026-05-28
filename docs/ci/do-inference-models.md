# DigitalOcean Inference — Model Catalog

CI for Dependabot PRs routes LLM-detector calls through DigitalOcean
Inference (`https://inference.do-ai.run/v1`) because the real OpenAI
secret is in the Actions secret store and GitHub silos that store from
Dependabot-triggered runs. This file is the model catalog we resolve
against without re-querying DO each CI run.

## How CI uses this

Three GitHub Actions secrets in the **Dependabot store** (not Actions):

| Secret | Value |
|---|---|
| `OPENAI_API_KEY` | DO API key (`sk-do-…`) — OpenAI-compatible auth |
| `OPENAI_BASE_URL` | `https://inference.do-ai.run/v1` |
| `VIJIL_LLM_MODEL` | A DO model id from the catalog below, prefixed `openai/` so litellm routes through the OpenAI hub (e.g. `openai/openai-gpt-4o-mini`) |

`vijil_dome/defaults.py` reads `VIJIL_LLM_MODEL`; `LlmBaseDetector`
(`vijil_dome/detectors/utils/llm_api_base.py`) reads `OPENAI_BASE_URL`
for the `openai` hub. Both fall back to existing defaults when unset,
so main-branch CI and library consumers behave identically to before.

## Refreshing this list

```bash
curl -s "https://inference.do-ai.run/v1/models" \
  -H "Authorization: Bearer $DO_API_KEY" \
  | jq -r '.data | sort_by(.id) | .[] | "\(.id)\t\(.owned_by)"'
```

`/v1/models` does not include embedding-only models reliably; see
the team's separate notes on DO Inference embeddings.

## Catalog (snapshot: 2026-05-28)

### OpenAI-owned (proxied)

OpenAI-compatible request shape; use these when the Dome test exercises
an OpenAI-style chat completion path.

| Model id | Notes |
|---|---|
| `openai-gpt-4o` | |
| `openai-gpt-4o-mini` | **CI default — cheap, fast, sufficient for binary-classifier prompts** |
| `openai-gpt-4.1` | |
| `openai-gpt-5` | |
| `openai-gpt-5-mini` | |
| `openai-gpt-5-nano` | |
| `openai-gpt-5.1-codex-max` | |
| `openai-gpt-5.2` / `openai-gpt-5.2-pro` | |
| `openai-gpt-5.3-codex` | Self-truncates on long contexts per team notes |
| `openai-gpt-5.4` / `openai-gpt-5.4-mini` / `openai-gpt-5.4-nano` / `openai-gpt-5.4-pro` | |
| `openai-gpt-5.5` | |
| `openai-o1` / `openai-o3` / `openai-o3-mini` | Reasoning models |
| `openai-gpt-image-1` / `openai-gpt-image-1.5` / `openai-gpt-image-2` | Image generation, not chat |

### Anthropic-owned (proxied)

| Model id |
|---|
| `anthropic-claude-4.1-opus` |
| `anthropic-claude-4.5-sonnet` |
| `anthropic-claude-4.6-sonnet` |
| `anthropic-claude-haiku-4.5` |
| `anthropic-claude-opus-4` |
| `anthropic-claude-opus-4.5` |
| `anthropic-claude-opus-4.6` |
| `anthropic-claude-opus-4.7` |
| `anthropic-claude-sonnet-4` |

### DigitalOcean-hosted (open-weight + DO-original)

| Model id | Notes |
|---|---|
| `alibaba-qwen3-32b` | |
| `arcee-trinity-large-thinking` | |
| `deepseek-3.2` / `deepseek-4-flash` / `deepseek-v4-pro` | |
| `deepseek-r1-distill-llama-70b` | |
| `gemma-4-31B-it` | |
| `glm-5` | |
| `kimi-k2.5` / `kimi-k2.6` | |
| `llama-4-maverick` | |
| `llama3.3-70b-instruct` | |
| `minimax-m2.5` | |
| `mistral-3-14B` | |
| `nemotron-3-nano-omni` | |
| `nemotron-nano-12b-v2-vl` | |
| `nvidia-nemotron-3-super-120b` | |
| `openai-gpt-oss-120b` / `openai-gpt-oss-20b` | DO's open-weight OSS Safeguard fork lives here |
| `qwen3-coder-flash` | |
| `qwen3.5-397b-a17b` | |

### Embedding / reranker / image / video

Not chat models. Listed for reference; tests that need these should
configure them separately.

| Model id | Type |
|---|---|
| `all-mini-lm-l6-v2` | Embedding |
| `bge-m3` | Embedding |
| `bge-reranker-v2-m3` | Reranker |
| `e5-large-v2` | Embedding |
| `gte-large-en-v1.5` | Embedding (referenced in team's DO embeddings notes) |
| `multi-qa-mpnet-base-dot-v1` | Embedding |
| `qwen3-embedding-0.6b` | Embedding |
| `qwen3-tts-voicedesign` | TTS |
| `stable-diffusion-3.5-large` | Image |
| `wan2-2-t2v-a14b` | Video |

### Routers

| Router id |
|---|
| `router:general` |
| `router:knowledge-base-document` |
| `router:software-engineering` |
| `router:writing` |
