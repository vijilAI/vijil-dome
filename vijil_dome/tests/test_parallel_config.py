import pytest
from vijil_dome import Dome, create_dome_config

PARALLEL_CONFIG = {
    "input-guards": [
        "security-input-guard",
        "moderation-input-guard",
        "privacy-input-guard",
    ],
    "output-guards": ["moderation-output-guard"],
    "input-early-exit": False,
    "output-early-exit": False,
    "input-run-parallel": True,  # ✅ Run input guards in parallel (correct attribute name)
    "output-run-parallel": True,  # ✅ Run output guards in parallel
    "security-input-guard": {
        "type": "security",
        "early-exit": False,
        "run-parallel": True,  # ✅ Run detectors in this guard in parallel (uses thread pool)
        "methods": ["prompt-injection-mbert", "encoding-heuristics"],
    },
    "moderation-input-guard": {
        "type": "moderation",
        "early-exit": False,
        "run-parallel": True,  # ✅ Run detectors in parallel
        "methods": ["moderation-flashtext"],
    },
    "moderation-output-guard": {
        "type": "moderation",
        "early-exit": False,
        "run-parallel": True,  # ✅ Run detectors in parallel (DeBERTa needs thread pool)
        "methods": ["moderation-deberta", "moderation-flashtext"],
    },
    "privacy-input-guard": {
        "type": "privacy",
        "run-parallel": True,  # ✅ Run detectors in parallel (Presidio needs thread pool)
        "methods": ["privacy-presidio"],
    },
}


@pytest.mark.asyncio
async def test_dome_parallel_config():
    dome = Dome(dome_config=create_dome_config(PARALLEL_CONFIG))
    input_str = (
        "this is an input prompt that gets run through a bunch of detectors in parallel"
    )
    await dome.async_guard_input(input_str)
    output_str = "this is an output prompt that gets run through a bunch of detectors in parallel"
    await dome.async_guard_output(output_str)
