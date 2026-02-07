# Prompt Injection Experiments

This folder contains reproducible experiments for prompt-injection benchmarking.

## Structure

- `scripts/build_dataset.py`
  - Builds benchmark datasets from trusted Hugging Face sources.
  - Builds your provided local dataset as a separate corpus.
  - Applies exact and near-duplicate removal.
  - Produces stratified `dev` and `holdout` splits per corpus.
- `data/datasets/`
  - Generated dataset artifacts.
  - `benchmark_pool/` and `local_paytm/` each contain:
    - `all.{csv,parquet}`
    - `dev.{csv,parquet}`
    - `holdout.{csv,parquet}`
    - `source_label_counts.csv`
    - `manifest.json`
  - Root `manifest.json` summarizes both corpora.
- `data/legacy/`
  - Older outputs moved from `examples/data`.

## Run

```bash
poetry -C /Users/dzen/Spaces/vjl/vijil-dome run python experiments/prompt_injection/scripts/build_dataset.py
```

## Evaluate (HF Classifiers)

```bash
poetry -C /Users/dzen/Spaces/vjl/vijil-dome run python experiments/prompt_injection/scripts/run_eval.py \
  --results-root experiments/prompt_injection/data/results/$(date +%Y%m%d_%H%M%S)_hf \
  --corpora benchmark_pool,local_paytm
```

## Evaluate (OpenGuardrails LLM via OpenAI-Compatible API)

The OpenGuardrails model card recommends `vllm serve ...` to expose an OpenAI-compatible API.
On Mac, vLLM often isn't available, so we provide a tiny local OpenAI-compatible server using
Transformers:

Terminal 1 (start server):

```bash
poetry -C /Users/dzen/Spaces/vjl/vijil-dome run python experiments/prompt_injection/scripts/serve_openai_compat_transformers.py \
  --model-id openguardrails/OpenGuardrails-Text-4B-0124 \
  --served-model-name OpenGuardrails-Text-4B-0124 \
  --port 8000
```

Terminal 2 (run benchmark through the OpenAI-compatible runner):

```bash
poetry -C /Users/dzen/Spaces/vjl/vijil-dome run python experiments/prompt_injection/scripts/run_eval.py \
  --results-root experiments/prompt_injection/data/results/$(date +%Y%m%d_%H%M%S)_ogr_oai \
  --corpora benchmark_pool \
  --max-rows-dev 200 \
  --max-rows-holdout 200 \
  --no-dome \
  --open-guardrails-model "" \
  --og-openai-base-url http://127.0.0.1:8000/v1 \
  --og-openai-model OpenGuardrails-Text-4B-0124 \
  --og-openai-positive-category S9 \
  --og-openai-concurrency 1
```

## Secrets (.env.local)

Both `build_dataset.py` and `run_eval.py` will load secrets from:

- `/Users/dzen/Spaces/vjl/vijil-dome/.env.local` (repo root)
- `/Users/dzen/Spaces/vjl/.env.local` (workspace root)

Useful vars:

- `HF_TOKEN` or `HUGGINGFACE_TOKEN`: for private HF models/datasets
