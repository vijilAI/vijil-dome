# Vijil Dome

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Python Version](https://img.shields.io/pypi/pyversions/vijil-dome)
[![Downloads](https://static.pepy.tech/badge/vijil-dome)](https://pepy.tech/project/vijil-dome)
[![Docs](https://img.shields.io/badge/Docs-blue?link=https%3A%2F%2Fdocs.vijil.ai%2Fdome%2Fintro.html)](https://docs.vijil.ai/dome/intro.html)

**Vijil Dome** is a fast, lightweight, and highly configurable library for adding runtime guardrails to your AI agents. It combines top open-source LLM safety tools with proprietary Vijil models to detect and respond to unsafe content — with built-in support for observability, tracing, and popular agent frameworks.


## 🚀 Installation

Install the core library:

```bash
pip install vijil-dome
```

Optional extras for common integrations:

* `opentelemetry` – OTel-compatible tracing/logging
* `google` – GCP-native metrics and logging
* `langchain` – Seamless integration with LangChain/LangGraph
* `embeddings` – Fast similarity search using `annoy`

> ⚠️ Note: `annoy` is not currently compatible with agents built using Google ADK + Cloud Run. Use in-memory embeddings in those cases.


## 🔒 Guarding Agents in One Line

```python
from vijil_dome import Dome

dome = Dome()

query = "How can I rob a bank?"
input_scan = dome.guard_input(query)
print(input_scan.is_safe(), input_scan.guarded_response())

# Get a response from your agent 

response = "Here's how to rob a bank!"
output_scan = dome.guard_output(response)
print(output_scan.is_safe(), output_scan.guarded_response())
```

By default, Dome:

* Scans inputs for prompt injections, jailbreaks, and toxicity
* Scans outputs for toxicity and masks PII


## ⚙️ Configuration Options

You can configure Dome using a TOML file or a Python dictionary.

### Example TOML

```toml
[guardrail]
input-guards = ["prompt-injection", "input-toxicity"]
output-guards = ["output-toxicity"]
input-early-exit = false

[prompt-injection]
type = "security"
early-exit = false
methods = ["prompt-injection-deberta-v3-base", "security-llm"]

[prompt-injection.security-llm]
model_name = "gpt-4o"

[input-toxicity]
type = "moderation"
methods = ["moderations-oai-api"]

[output-toxicity]
type = "moderation"
methods = ["moderation-prompt-engineering"]
```

### Same Configuration in Python

```python
config = {
    "input-guards": ["prompt-injection", "input-toxicity"],
    "output-guards": ["output-toxicity"],
    "input-early-exit": False,
    "prompt-injection": {
        "type": "security",
        "early-exit": False,
        "methods": ["prompt-injection-deberta-v3-base", "security-llm"],
        "security-llm": {
            "model_name": "gpt-4o"
        }
    },
    "input-toxicity": {
        "type": "moderation",
        "methods": ["moderations-oai-api"]
    },
    "output-toxicity": {
        "type": "moderation",
        "methods": ["moderation-prompt-engineering"]
    },
}
```

Dome includes nearly 20 prebuilt guardrails and supports building your own!

👉 For the full list of guardrail methods, advanced config options, and extensibility, check out the [Docs](https://docs.vijil.ai/dome/intro.html).

## 🔌 Compatibility

Dome works with **any agent framework or LLM** — it operates directly on strings, so there's no dependency on your stack!

For popular frameworks, we provide dedicated wrappers and tutorials to make integration seamless:

* [**Google ADK**](https://docs.vijil.ai/dome/tutorials/adk.html)
* [**LangChain & LangGraph**](https://docs.vijil.ai/dome/tutorials/)

### Observability Integrations:

Dome is compatible with the following observability framworks out of the box

* **OpenTelemetry**
* **Weave** (Weights & Biases)
* **AgentOps**
* **Google Cloud Trace**

See the [documentation](https://docs.vijil.ai/dome/tutorials/observability.html) for more details


📚 Learn More
---
Get detailed guides, examples, and custom guardrail walkthroughs in the [official documentation →](https://docs.vijil.ai/dome/intro.html)

Have more questions, or want us to include another guardrailing technique? Reach out to us at contact@vijil.ai!
