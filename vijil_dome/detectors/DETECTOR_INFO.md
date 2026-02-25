# Dome Detectors Reference

This document lists every built-in detector, grouped by category. Each entry
shows the **method name** used in TOML/dict configuration, the underlying model
or service, and all configurable parameters.

> Parameters are passed as key-value pairs under the method name in your
> configuration. See [Configuration](#configuration) at the bottom for examples.

---

## Security

Detectors that identify adversarial inputs such as prompt injections,
jailbreak attempts, and encoded/obfuscated payloads.

### `prompt-injection-deberta-v3-base`

DeBERTa v3 model for prompt injection detection.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `truncation` | `bool` | `True` | Truncate inputs exceeding `max_length` |
| `max_length` | `int` | `512` | Maximum tokens per window (DeBERTa limit) |
| `window_stride` | `int` | `256` | Token step size between sliding windows |

- **Class**: `DebertaPromptInjectionModel`
- **Model**: [protectai/deberta-v3-base-prompt-injection-v2](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2)

### `prompt-injection-deberta-finetuned-11122024`

Vijil-finetuned DeBERTa model for prompt injection detection.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `truncation` | `bool` | `True` | Truncate inputs exceeding `max_length` |
| `max_length` | `int` | `512` | Maximum tokens per window (DeBERTa limit) |
| `window_stride` | `int` | `256` | Token step size between sliding windows |

- **Class**: `DebertaTuned60PromptInjectionModel`
- **Model**: [vijil/pi_deberta_finetuned_11122024](https://huggingface.co/vijil/pi_deberta_finetuned_11122024)

### `prompt-injection-mbert`

Vijil ModernBERT model for prompt injection detection. Supports up to 8,192
tokens natively, so sliding windows only activate for very long inputs.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `score_threshold` | `float` | `0.5` | Injection probability above which input is flagged |
| `truncation` | `bool` | `True` | Truncate inputs exceeding `max_length` |
| `max_length` | `int` | `8192` | Maximum tokens per window |
| `window_stride` | `int` | `4096` | Token step size between sliding windows |

- **Class**: `MBertPromptInjectionModel`
- **Model**: [vijil/vijil_dome_prompt_injection_detection](https://huggingface.co/vijil/vijil_dome_prompt_injection_detection)

### `security-promptguard`

Meta Prompt Guard model for jailbreak and prompt injection detection.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `score_threshold` | `float` | `0.5` | Jailbreak probability threshold |
| `truncation` | `bool` | `True` | Truncate inputs exceeding `max_length` |
| `max_length` | `int` | `512` | Maximum tokens per window |
| `window_stride` | `int` | `256` | Token step size between sliding windows |

- **Class**: `PromptGuardSecurityModel`
- **Model**: [meta-llama/Prompt-Guard-86M](https://huggingface.co/meta-llama/Prompt-Guard-86M)

### `security-llm`

LLM-based security classification via LiteLLM.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hub_name` | `str` | `"openai"` | LLM API provider |
| `model_name` | `str` | `"gpt-4-turbo"` | Model name |
| `api_key` | `str` | `None` | API key (falls back to env var) |
| `max_input_chars` | `int` | `None` | Truncate input to this many characters |

- **Class**: `LlmSecurity`

### `security-embeddings`

Jailbreak detection via embedding similarity against a known-jailbreak corpus.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `engine` | `str` | `"SentenceTransformers"` | Embedding engine |
| `model` | `str` | `"all-MiniLM-L6-v2"` | Embedding model name |
| `threshold` | `float` | `0.7` | Similarity threshold |
| `in_mem` | `bool` | `True` | Load embeddings in memory |

- **Class**: `JailbreakEmbeddingsDetector`
- **Model**: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

### `jb-length-per-perplexity`

Perplexity-based heuristic that flags jailbreaks by their length-to-perplexity
ratio.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_id` | `str` | `"gpt2-large"` | HuggingFace model for perplexity |
| `batch_size` | `int` | `16` | Batch size |
| `stride_length` | `int` | `512` | Stride for perplexity calculation |
| `threshold` | `float` | `89.79` | Length-per-perplexity threshold |

- **Class**: `LengthPerPerplexityModel`

### `jb-prefix-suffix-perplexity`

Perplexity-based heuristic that analyses the prefix and suffix of inputs
separately.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_id` | `str` | `"gpt2-large"` | HuggingFace model for perplexity |
| `batch_size` | `int` | `16` | Batch size |
| `stride_length` | `int` | `512` | Stride for perplexity calculation |
| `prefix_threshold` | `float` | `1845.65` | Prefix perplexity threshold |
| `suffix_threshold` | `float` | `1845.65` | Suffix perplexity threshold |
| `prefix_length` | `int` | `20` | Number of prefix words to analyse |
| `suffix_length` | `int` | `20` | Number of suffix words to analyse |

- **Class**: `PrefixSuffixPerplexityModel`

### `encoding-heuristics`

Rule-based detector for encoded or obfuscated payloads (base64, ROT13, hex,
URL encoding, Unicode tricks, etc.).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold_map` | `dict` | *(see below)* | Per-encoding-type thresholds |

Default `threshold_map`:

| Encoding Type | Threshold |
|---------------|-----------|
| `base64` | `0.7` |
| `rot13` | `0.7` |
| `ascii_escape` | `0.05` |
| `hex_encoding` | `0.15` |
| `url_encoding` | `0.15` |
| `cyrillic_homoglyphs` | `0.05` |
| `mixed_scripts` | `0.05` |
| `zero_width` | `0.01` |
| `excessive_whitespace` | `0.4` |

- **Class**: `EncodingHeuristicsDetector`

---

## Moderation

Detectors for toxic, harmful, or otherwise inappropriate content.

### `moderation-deberta`

DeBERTa model for toxicity scoring. The 208-token context window means the
sliding window activates for most non-trivial inputs.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `truncation` | `bool` | `True` | Truncate inputs exceeding `max_length` |
| `max_length` | `int` | `208` | Maximum tokens per window |
| `window_stride` | `int` | `104` | Token step size between sliding windows |
| `device` | `str` | `None` | Torch device (auto-selects CUDA if available) |

- **Class**: `ToxicityDeberta`
- **Model**: [cooperleong00/deberta-v3-large_toxicity-scorer](https://huggingface.co/cooperleong00/deberta-v3-large_toxicity-scorer)

### `moderation-mbert`

Vijil ModernBERT model for toxic content detection. Supports up to 8,192
tokens natively.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `score_threshold` | `float` | `0.5` | Toxicity probability threshold |
| `truncation` | `bool` | `True` | Truncate inputs exceeding `max_length` |
| `max_length` | `int` | `8192` | Maximum tokens per window |
| `window_stride` | `int` | `4096` | Token step size between sliding windows |

- **Class**: `MBertToxicContentModel`
- **Model**: [vijil/vijil_dome_toxic_content_detection](https://huggingface.co/vijil/vijil_dome_toxic_content_detection)

### `moderations-oai-api`

OpenAI Moderation API with per-category score thresholds.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `score_threshold_dict` | `dict` | `None` | Custom thresholds per category |

Supported categories: `hate`, `hate/threatening`, `self-harm`, `sexual`,
`sexual/minors`, `violence`, `violence/graphic`, `harassment`,
`harassment/threatening`, `illegal`, `illicit`, `self-harm/intent`,
`self-harm/instructions`, `sexual/instructions`.

- **Class**: `OpenAIModerations`
- **Requires**: `OPENAI_API_KEY` environment variable

### `moderation-perspective-api`

Google Perspective API for toxicity and other attributes.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str` | `None` | Google API key (falls back to `PERSPECTIVE_API_KEY`) |
| `attributes` | `dict` | `{"TOXICITY": {}}` | Attributes to analyse |
| `score_threshold` | `dict` | `{"TOXICITY": 0.5}` | Per-attribute thresholds |

Available attributes: `TOXICITY`, `SEVERE_TOXICITY`, `IDENTITY_ATTACK`,
`INSULT`, `PROFANITY`, `THREAT`.

- **Class**: `PerspectiveAPI`
- **Requires**: `PERSPECTIVE_API_KEY` environment variable

### `moderation-prompt-engineering`

LLM-based moderation classification via LiteLLM.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hub_name` | `str` | `"openai"` | LLM API provider |
| `model_name` | `str` | `"gpt-4-turbo"` | Model name |
| `api_key` | `str` | `None` | API key (falls back to env var) |
| `max_input_chars` | `int` | `None` | Truncate input to this many characters |

- **Class**: `LlmModerations`

### `moderation-flashtext`

Keyword ban-list detector using FlashText for fast matching.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `banlist_filepaths` | `list[str]` | `None` | Paths to ban-list files (uses built-in default list if omitted) |

- **Class**: `KWBanList`

---

## Privacy

Detectors for personally identifiable information (PII) and secrets.

### `privacy-presidio`

Microsoft Presidio-based PII detection and redaction.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `score_threshold` | `float` | `0.5` | Confidence threshold for PII detection |
| `anonymize` | `bool` | `True` | Redact detected PII in the response |
| `allow_list_files` | `list[str]` | `None` | Files with values to exclude from detection |
| `redaction_style` | `str` | `"labeled"` | Redaction style: `"labeled"` or `"masked"` |

- **Class**: `PresidioDetector`

### `detect-secrets`

Pattern-based secret and credential detection (API keys, tokens, etc.).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `censor` | `bool` | `True` | Censor detected secrets in the response |

Includes 25 detector plugins: ArtifactoryDetector, AWSKeyDetector,
AzureStorageKeyDetector, BasicAuthDetector, CloudantDetector,
DiscordBotTokenDetector, GitHubTokenDetector, GitLabTokenDetector,
IbmCloudIamDetector, IbmCosHmacDetector, IPPublicDetector, JwtTokenDetector,
KeywordDetector, MailchimpDetector, NpmDetector, OpenAIDetector,
PrivateKeyDetector, PypiTokenDetector, SendGridDetector, SlackDetector,
SoftlayerDetector, SquareOAuthDetector, StripeDetector,
TelegramBotTokenDetector, TwilioKeyDetector.

- **Class**: `SecretDetector`

---

## Integrity

Detectors for hallucinations and factual accuracy. These typically require a
reference `context` to compare against.

### `hhem-hallucination`

Vectara HHEM model for hallucination detection by comparing output against a
reference context.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `context` | `str` | `""` | Reference context to compare against |
| `factual_consistency_score_threshold` | `float` | `0.5` | Score below which output is flagged |
| `trust_remote_code` | `bool` | `True` | Trust remote code from model hub |

- **Class**: `HhemHallucinationModel`
- **Model**: [vectara/hallucination_evaluation_model](https://huggingface.co/vectara/hallucination_evaluation_model)

### `fact-check-roberta`

RoBERTa model for detecting factual contradictions between output and context.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `context` | `str` | `""` | Reference context to check against |

- **Class**: `RobertaFactCheckModel`
- **Model**: [Dzeniks/roberta-fact-check](https://huggingface.co/Dzeniks/roberta-fact-check)

### `hallucination-llm`

LLM-based hallucination detection with reference context.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hub_name` | `str` | `"openai"` | LLM API provider |
| `model_name` | `str` | `"gpt-4-turbo"` | Model name |
| `api_key` | `str` | `None` | API key (falls back to env var) |
| `max_input_chars` | `int` | `None` | Truncate input to this many characters |
| `context` | `str` | `None` | Reference context for comparison |

- **Class**: `LlmHallucination`

### `fact-check-llm`

LLM-based fact-checking with reference context.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hub_name` | `str` | `"openai"` | LLM API provider |
| `model_name` | `str` | `"gpt-4-turbo"` | Model name |
| `api_key` | `str` | `None` | API key (falls back to env var) |
| `max_input_chars` | `int` | `None` | Truncate input to this many characters |
| `context` | `str` | `None` | Reference context for comparison |

- **Class**: `LlmFactcheck`

---

## Generic

Flexible detectors that can be customised for arbitrary use cases.

### `generic-llm`

Custom LLM-based detection with user-provided system prompts and trigger words.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sys_prompt_template` | `str` | *(required)* | System prompt with `$query_string` placeholder |
| `trigger_word_list` | `list[str]` | *(required)* | Words in LLM response that indicate a hit |
| `hub_name` | `str` | `"openai"` | LLM API provider |
| `model_name` | `str` | `"gpt-4-turbo"` | Model name |
| `api_key` | `str` | `None` | API key (falls back to env var) |
| `max_input_chars` | `int` | `None` | Truncate input to this many characters |

- **Class**: `GenericLLMDetector`

### `policy-gpt-oss-safeguard`

Policy-based content classification using GPT-OSS-Safeguard.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `policy_file` | `str` | *(required)* | Path to policy file with classification rules |
| `hub_name` | `str` | `"groq"` | LLM API provider |
| `model_name` | `str` | `"openai/gpt-oss-safeguard-20b"` | Model name |
| `output_format` | `str` | `"policy_ref"` | `"binary"`, `"policy_ref"`, or `"with_rationale"` |
| `reasoning_effort` | `str` | `"medium"` | `"low"`, `"medium"`, or `"high"` |
| `api_key` | `str` | `None` | API key (falls back to env var) |
| `timeout` | `int` | `60` | Request timeout in seconds |
| `max_retries` | `int` | `3` | Maximum retry attempts |
| `max_input_chars` | `int` | `None` | Truncate input to this many characters |

- **Class**: `PolicyGptOssSafeguard`

---

## Sliding Window Behaviour

HuggingFace-based detectors (DeBERTa, ModernBERT, PromptGuard) use a sliding
window to handle inputs longer than their `max_length`. Key points:

- **Fast path**: inputs that fit in a single window are processed unchanged.
- **Overlap**: `window_stride` < usable window size creates overlapping windows,
  ensuring content at boundaries is not missed.
- **Aggregation**: any window flagged as unsafe causes the entire input to be
  flagged (*any-positive* strategy). For score-based detectors, the maximum
  score across windows is reported.
- **Batch processing**: `detect_batch()` flattens all chunks from all inputs
  into a single pipeline call, then re-aggregates results per input.

The `window_stride` parameter is configurable per detector via TOML or dict
config.

## Configuration

Parameters are passed under the method name in your guard configuration:

### TOML

```toml
[prompt-injection]
type = "security"
methods = ["prompt-injection-deberta-v3-base"]

[prompt-injection.prompt-injection-deberta-v3-base]
window_stride = 128  # More overlap for thorough detection

[input-toxicity]
type = "moderation"
methods = ["moderation-mbert"]

[input-toxicity.moderation-mbert]
score_threshold = 0.7

[output-safety]
type = "security"
methods = ["security-llm"]

[output-safety.security-llm]
max_input_chars = 50000
model_name = "gpt-4o"
```

### Python dict

```python
config = {
    "input-guards": ["prompt-injection"],
    "prompt-injection": {
        "type": "security",
        "methods": ["prompt-injection-deberta-v3-base"],
        "prompt-injection-deberta-v3-base": {
            "window_stride": 128,
        },
    },
}
```
