# GPT-OSS-Safeguard Guardrail Guide

This guide explains how to use `policy-gpt-oss-safeguard` as:

1. A direct detector (`PolicyGptOssSafeguard`)
2. A Dome guardrail method in standard config format (`type = "generic"`)

Source implementation:
- `vijil_dome/detectors/methods/gpt_oss_safeguard_policy.py`

Example script:
- `vijil_dome/integrations/examples/gpt_oss_safeguard_example.py`

Config example:
- `examples/gpt_oss_safeguard_guardrail.toml`

## Prerequisites

1. Set `GROQ_API_KEY`
2. Provide a policy file (`.md` or `.txt`)

```bash
export GROQ_API_KEY="your-key"
```

## Accepted Input Shapes

The detector accepts:

1. `User request: ...` + `Agent response: ...`
2. Only `Agent response: ...` / `Assistant: ...`
3. Only `User request: ...` / `User: ...`
4. Freeform text

If both sides are present, classification prioritizes the latest assistant/agent response.

## Direct Detector Usage

```python
from vijil_dome.detectors.methods.gpt_oss_safeguard_policy import PolicyGptOssSafeguard

detector = PolicyGptOssSafeguard(
    policy_file="/absolute/path/to/policy.md",
    hub_name="groq",
    model_name="openai/gpt-oss-safeguard-20b",
    output_format="policy_ref",   # binary | policy_ref | with_rationale
    reasoning_effort="medium",    # low | medium | high
    timeout=60,
    max_retries=3,
)

flagged, metadata = await detector.detect(
    "User request: ...\nAgent response: ..."
)
```

## Dome Config Pattern (Dict)

```python
from vijil_dome import Dome, create_dome_config

config = {
    "input-guards": ["policy-input"],
    "output-guards": [],
    "policy-input": {
        "type": "generic",
        "methods": ["policy-gpt-oss-safeguard"],
        "policy-gpt-oss-safeguard": {
            "policy_file": "/absolute/path/to/policy.md",
            "hub_name": "groq",
            "model_name": "openai/gpt-oss-safeguard-20b",
            "output_format": "policy_ref",
            "reasoning_effort": "medium",
            "timeout": 60,
            "max_retries": 3,
        },
    },
}

dome = Dome(dome_config=create_dome_config(config))
result = await dome.async_guard_input("User request: ...\nAgent response: ...")
```

## Dome Config Pattern (TOML)

Use:
- `examples/gpt_oss_safeguard_guardrail.toml`

```python
from vijil_dome import Dome

dome = Dome(dome_config="examples/gpt_oss_safeguard_guardrail.toml")
```

## Detector Parameters

Required:
- `policy_file: str`

Optional:
- `hub_name: str = "groq"`
- `model_name: str = "openai/gpt-oss-safeguard-20b"`
- `output_format: "binary" | "policy_ref" | "with_rationale" = "policy_ref"`
- `reasoning_effort: "low" | "medium" | "high" = "medium"`
- `api_key: Optional[str] = None` (defaults to `GROQ_API_KEY` for Groq)
- `timeout: int = 60`
- `max_retries: int = 3`

## Output Modes

1. `binary`
Example output: `0` or `1`

2. `policy_ref`
Example output:
`{"violation": 1, "policy_category": "H2.f"}`

3. `with_rationale`
Example output:
`{"violation":1,"policy_category":"H2.f","rule_ids":["H2.d","H2.f"],"confidence":"high","rationale":"..."}`

## Test Coverage

Policy safeguard tests:
- `vijil_dome/tests/detectors/test_gpt_oss_sg_policy.py`

Includes:
1. 3x3 integration matrix: 3 reasoning levels x 3 output formats
2. Output parsing checks for all three modes
3. Input-shape normalization checks

Run:

```bash
python -m pytest vijil_dome/tests/detectors/test_gpt_oss_sg_policy.py -q
```
