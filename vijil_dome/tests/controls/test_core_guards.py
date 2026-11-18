from pathlib import Path

import pytest

from vijil_dome.controls.engine import run_sync
from vijil_dome.core import VijilDome
from vijil_dome.Dome import create_dome_config

HEURISTIC_CFG = {
    "input-guards": ["in-guard"],
    "output-guards": ["out-guard"],
    "in-guard": {"type": "security", "methods": ["encoding-heuristics"]},
    "out-guard": {"type": "security", "methods": ["encoding-heuristics"]},
}


def _guard_names(guardrail) -> list[str]:
    return [g.guard_name for g in guardrail.guard_list]


def test_run_sync_without_loop():
    async def f():
        return 7

    assert run_sync(f()) == 7


@pytest.mark.asyncio
async def test_run_sync_inside_running_loop():
    async def f():
        return 7

    assert run_sync(f()) == 7


def test_guards_none_means_no_guardrails():
    v = VijilDome()
    assert v.input_guardrail is None and v.output_guardrail is None


def test_guards_dict_builds_both_guardrails():
    v = VijilDome(guards=HEURISTIC_CFG)
    assert _guard_names(v.input_guardrail) == ["in-guard"]
    assert _guard_names(v.output_guardrail) == ["out-guard"]


def test_guards_enforce_defaults_to_fail_closed():
    assert VijilDome(guards=HEURISTIC_CFG, enforce=True).input_guardrail.on_error == "fail_closed"
    assert VijilDome(guards=HEURISTIC_CFG, enforce=False).input_guardrail.on_error == "fail_open"


def test_guards_accepts_toml_path(tmp_path: Path):
    toml_path = tmp_path / "guards.toml"
    toml_path.write_text(
        '[guardrail]\ninput-guards = ["in-guard"]\noutput-guards = ["out-guard"]\n'
        '[in-guard]\ntype = "security"\nmethods = ["encoding-heuristics"]\n'
        '[out-guard]\ntype = "security"\nmethods = ["encoding-heuristics"]\n'
    )
    v = VijilDome(guards=str(toml_path))
    assert _guard_names(v.input_guardrail) == ["in-guard"]
    assert _guard_names(v.output_guardrail) == ["out-guard"]


def test_guards_accepts_domeconfig():
    v = VijilDome(guards=create_dome_config(HEURISTIC_CFG))
    assert _guard_names(v.input_guardrail) == ["in-guard"]
    assert _guard_names(v.output_guardrail) == ["out-guard"]


def test_guards_rejects_other_types():
    with pytest.raises(ValueError):
        VijilDome(guards=42)
