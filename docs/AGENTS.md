# AGENTS.md

This file provides detailed guidance for AI agents (Claude Code, etc.) when working with the vijil-dome repository.

## Project Overview

**vijil-dome** is Vijil's runtime guardrail library for AI agents. It's a **pip-installable Python library** (not a service) that provides:

- **Input Guards** - Detect prompt injections, jailbreaks, toxicity before agent processing
- **Output Guards** - Detect unsafe outputs, mask PII, filter harmful content
- **Detection Methods** - ~20 prebuilt detectors using HuggingFace, LLMs, heuristics
- **Framework Integrations** - Google ADK, LangChain, MCP, Strands, OpenTelemetry

## Architecture

```
vijil_dome/
├── Dome.py                    # Main entry point - Dome class
├── defaults.py                # Default configuration
├── types.py                   # Type definitions
│
├── detectors/                 # Detection methods
│   ├── methods/               # Detector implementations
│   │   ├── pi_hf_deberta.py   # Prompt injection (DeBERTa)
│   │   ├── pi_hf_mbert.py     # Prompt injection (mBERT)
│   │   ├── toxicity_deberta.py # Toxicity detection
│   │   ├── pii_presidio.py    # PII detection (Presidio)
│   │   ├── secret_detector.py # Secret/credential detection
│   │   ├── llm_models.py      # LLM-based detection
│   │   ├── openai_models.py   # OpenAI moderation API
│   │   ├── perspective.py     # Google Perspective API
│   │   ├── hhem_hallucination.py # Hallucination detection
│   │   └── ...
│   └── utils/                 # Detector utilities
│       ├── hf_model.py        # HuggingFace model loading
│       ├── llm_api_base.py    # LLM API base class
│       └── embeddings_base.py # Embedding utilities
│
├── guardrails/                # Guardrail configuration
│   ├── config_parser.py       # TOML/dict config parsing
│   └── defaults.py            # Default guardrail configs
│
├── embeds/                    # Embedding models
│   ├── embedder.py            # Main embedder class
│   └── models/                # Model implementations
│
├── integrations/              # Framework integrations
│   ├── adk/                   # Google ADK callbacks
│   ├── langchain/             # LangChain runnable
│   ├── mcp/                   # MCP wrapper
│   ├── strands/               # Strands agent hooks
│   └── vijil/                 # Vijil platform integration
│
└── instrumentation/           # Observability
    ├── tracing.py             # OTEL tracing
    └── logging/               # Structured logging
```

## Detector Categories

| Category | Detectors | Purpose |
|----------|-----------|---------|
| **Security** | `prompt-injection-deberta`, `security-llm`, `jailbreak-heuristics` | Detect adversarial inputs |
| **Moderation** | `toxicity-deberta`, `moderations-oai-api`, `perspective-api` | Content moderation |
| **Privacy** | `pii-presidio`, `secret-detector` | PII/credential detection |
| **Integrity** | `hhem-hallucination`, `factcheck-roberta` | Factual accuracy |

## Common Usage

### Basic Usage

```python
from vijil_dome import Dome

dome = Dome()

# Guard input
input_scan = dome.guard_input("How can I hack a system?")
if not input_scan.is_safe():
    return input_scan.guarded_response()

# Guard output
output_scan = dome.guard_output(agent_response)
if not output_scan.is_safe():
    return output_scan.guarded_response()
```

### Configuration

```python
# Via dictionary
config = {
    "input-guards": ["prompt-injection", "input-toxicity"],
    "output-guards": ["output-toxicity", "pii-masking"],
    "prompt-injection": {
        "type": "security",
        "methods": ["prompt-injection-deberta-v3-base"]
    }
}
dome = Dome(config)

# Via TOML file
dome = Dome.from_toml("guardrails.toml")
```

### Google ADK Integration

```python
from vijil_dome.integrations.adk import DomeCallback
from google.adk import Agent

agent = Agent(
    model="gemini-2.0-flash",
    callbacks=[DomeCallback()]
)
```

## Common Commands

```bash
# Install from PyPI
pip install vijil-dome

# Install with extras
pip install "vijil-dome[opentelemetry,langchain,embeddings]"

# CPU-only PyTorch (reduces image size by ~2GB)
pip install vijil-dome
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cpu

# Run tests
poetry run pytest

# Run specific detector tests
poetry run pytest vijil_dome/tests/detectors/test_security_detectors.py -v
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | For OpenAI moderation API |
| `GOOGLE_API_KEY` | - | For Perspective API |
| `ANTHROPIC_API_KEY` | - | For Claude-based detection |
| `HF_TOKEN` | - | For gated HuggingFace models |

## Testing Conventions

- **Location**: `vijil_dome/tests/detectors/` for detector tests
- **Categories**: `test_security_detectors.py`, `test_privacy_detectors.py`, etc.
- **Fixtures**: Use `@pytest.fixture` for Dome instances with specific configs
- **Assertions**: Check `is_safe()`, `guarded_response()`, detection scores

## Development Notes

### Adding a New Detector

1. Create detector class in `detectors/methods/`
2. Inherit from appropriate base (`HFModel`, `LLMAPIBase`, etc.)
3. Register in detector registry
4. Add to default config if appropriate
5. Add tests in `tests/detectors/`

### Integration Patterns

- **ADK**: Use `DomeCallback` for before/after model hooks
- **LangChain**: Use `DomeRunnable` in chains
- **MCP**: Use `DomeMCPWrapper` for tool wrapping
- **Strands**: Use `DomeHookProvider` for before/after model hooks

### Strands Integration

```python
from vijil_dome import Dome
from vijil_dome.integrations.strands import DomeHookProvider
from strands import Agent

dome = Dome(config)
agent = Agent(hooks=[DomeHookProvider(dome, agent_id="my-agent")])
```

## Relationship to Other Repos

| Repo | Relationship |
|------|-------------|
| **vijil-console** | Dome is used for evaluation via Diamond |
| **vijil-diamond** | Diamond uses Dome detectors for agent evaluation |
| **vijil-travel-agent** | Demo agent using Dome for protection |
| **vijil-domed-travel-agent** | Travel agent with Dome guardrails |

## External Documentation

Full documentation at [docs.vijil.ai/dome](https://docs.vijil.ai/dome/intro.html)
