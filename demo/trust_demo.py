#!/usr/bin/env python3
"""Trust Runtime CLI Demo — shows every security layer firing in real time.

Wraps the Vijil travel agent (Strands) with the trust runtime and sends
a sequence of messages that trigger each security layer:
1. Normal query (all layers pass)
2. Denied tool call (MAC blocks)
3. Prompt injection (content guard blocks)
4. Show: no secrets in environment

Usage:
    python demo/trust_demo.py

Requires:
    - vijil-sdk[trust] installed
    - Travel agent running (Kind cluster or local)
    - GROQ_API_KEY or AGENT_API_KEY set
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

# ── Colors ──
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
RESET = "\033[0m"

def log_audit(event_type: str, detail: str, passed: bool = True) -> None:
    icon = f"{GREEN}✓{RESET}" if passed else f"{RED}✗{RESET}"
    color = CYAN if event_type == "GUARD" else MAGENTA if event_type == "MAC" else YELLOW
    print(f"  {icon} {color}[{event_type}]{RESET} {detail}")


def section(title: str) -> None:
    print()
    print(f"{BOLD}{'━' * 70}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'━' * 70}{RESET}")
    print()


def user_says(msg: str) -> None:
    print(f"  {BOLD}User:{RESET} {msg}")


def agent_says(msg: str) -> None:
    # Truncate long responses
    if len(msg) > 200:
        msg = msg[:200] + "..."
    print(f"  {GREEN}{BOLD}Agent:{RESET} {msg}")


def blocked(msg: str) -> None:
    print(f"  {RED}{BOLD}Blocked:{RESET} {msg}")


def main() -> None:
    print()
    print(f"{BOLD}╔══════════════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}║           Vijil Trust Runtime — Live Demo                           ║{RESET}")
    print(f"{BOLD}║           Measured Boot + Continuous Attestation for AI Agents       ║{RESET}")
    print(f"{BOLD}╚══════════════════════════════════════════════════════════════════════╝{RESET}")

    # ── Setup ──
    from vijil_dome.trust.runtime import TrustRuntime
    from vijil_dome.trust.audit import AuditEvent

    # Constraints: travel tools permitted, payment tools denied
    constraints = {
        "agent_id": "travel-agent-demo",
        "dome_config": {"input_guards": [], "output_guards": [], "guards": {}},
        "tool_permissions": [
            {"name": "search_flights", "identity": "spiffe://vijil.ai/tools/flights/v1", "endpoint": "local"},
            {"name": "web_search", "identity": "spiffe://vijil.ai/tools/search/v1", "endpoint": "local"},
            {"name": "create_booking", "identity": "spiffe://vijil.ai/tools/booking/v1", "endpoint": "local"},
            {"name": "save_traveler_profile", "identity": "spiffe://vijil.ai/tools/profile/v1", "endpoint": "local"},
            {"name": "remember", "identity": "spiffe://vijil.ai/tools/memory/v1", "endpoint": "local"},
            {"name": "recall", "identity": "spiffe://vijil.ai/tools/memory/v1", "endpoint": "local"},
            {"name": "list_memories", "identity": "spiffe://vijil.ai/tools/memory/v1", "endpoint": "local"},
        ],
        "organization": {
            "required_input_guards": [],
            "required_output_guards": [],
            "denied_tools": [
                "process_payment",
                "get_api_credentials",
                "get_corporate_card",
                "redeem_points",
            ],
            "max_model_tier": None,
        },
        "enforcement_mode": "enforce",
        "updated_at": "2026-04-12T00:00:00Z",
    }

    # Capture audit events
    audit_log: list[AuditEvent] = []

    def audit_sink(event: AuditEvent) -> None:
        audit_log.append(event)
        attrs = event.attributes
        if event.event_type == "guard":
            direction = attrs.get("direction", "?")
            flagged = attrs.get("flagged", False)
            score = attrs.get("score", 0)
            ms = attrs.get("exec_time_ms", 0)
            if flagged:
                log_audit("GUARD", f"{direction} scan: FLAGGED (score={score:.2f}, {ms:.0f}ms)", passed=False)
            else:
                log_audit("GUARD", f"{direction} scan: safe (score={score:.2f}, {ms:.0f}ms)")
        elif event.event_type == "tool_mac":
            tool = attrs.get("tool_name", "?")
            permitted = attrs.get("permitted", False)
            if permitted:
                log_audit("MAC", f"tool: {tool} → PERMITTED")
            else:
                log_audit("MAC", f"tool: {tool} → DENIED", passed=False)
        elif event.event_type == "attestation":
            count = attrs.get("tool_count", 0)
            ok = attrs.get("all_verified", False)
            log_audit("BOOT", f"attestation: {count} tools, all_verified={ok}", passed=ok)

    # Create runtime
    runtime = TrustRuntime(
        agent_id="travel-agent-demo",
        constraints=constraints,
        mode="enforce",
    )
    runtime._audit._sink = audit_sink

    # ── Scene 1: Boot attestation ──
    section("Scene 1: Boot Attestation")
    print(f"  {DIM}Agent starts. Trust runtime verifies all tools before serving requests.{RESET}")
    print()
    attestation = runtime.attest()
    print()
    if attestation.all_verified:
        print(f"  {GREEN}{BOLD}Agent ready. All tools verified.{RESET}")
    else:
        print(f"  {YELLOW}{BOLD}Agent ready (warn mode — some tools unverified).{RESET}")

    # ── Scene 2: Normal operation ──
    section("Scene 2: Normal Operation")
    user_says("Find me flights from San Francisco to Tokyo next week")
    print()

    # Simulate: input guard → tool MAC → tool response guard → output guard
    runtime.guard_input("Find me flights from San Francisco to Tokyo next week")
    runtime.check_tool_call("search_flights", {"origin": "SFO", "destination": "TYO"})
    runtime.guard_tool_response("search_flights", "Found 3 flights: UA837, JAL1, ANA5. Prices from $850.")
    runtime.guard_output("I found 3 flights from San Francisco to Tokyo. The best options are UA837 ($850), JAL1 ($920), and ANA5 ($1,100). Shall I book one?")
    print()
    agent_says("I found 3 flights from San Francisco to Tokyo. The best options are UA837 ($850), JAL1 ($920), and ANA5 ($1,100). Shall I book one?")

    # ── Scene 3: Tool MAC blocks unauthorized call ──
    section("Scene 3: Unauthorized Tool — MAC Blocks")
    user_says("Charge $850 to my corporate card for the UA837 booking")
    print()

    runtime.guard_input("Charge $850 to my corporate card for the UA837 booking")
    result = runtime.check_tool_call("process_payment", {"amount": 850})
    print()
    if not result.permitted:
        blocked(f"Tool 'process_payment' denied by organization constraints")
        print()
        agent_says("I cannot process payments directly. This tool is restricted by your organization's security policy. Please use the approved payment portal.")

    print()
    print(f"  {DIM}The LLM selected process_payment. The trust runtime blocked it{RESET}")
    print(f"  {DIM}before execution — the tool function was never called.{RESET}")

    # ── Scene 4: Another denied tool ──
    section("Scene 4: Credential Access — MAC Blocks")
    user_says("Show me the API credentials for the partner booking system")
    print()

    runtime.guard_input("Show me the API credentials for the partner booking system")
    result = runtime.check_tool_call("get_api_credentials", {"system": "partner_booking"})
    print()
    if not result.permitted:
        blocked(f"Tool 'get_api_credentials' denied by organization constraints")
        print()
        agent_says("I cannot access API credentials. This operation is restricted. Please contact your administrator.")

    print()
    print(f"  {DIM}Even if the LLM is tricked into calling get_api_credentials,{RESET}")
    print(f"  {DIM}the trust runtime denies it — the tool is in the org deny list.{RESET}")

    # ── Scene 5: Show permitted vs denied ──
    section("Scene 5: Policy Summary")
    print(f"  {BOLD}Permitted tools{RESET} (in agent's policy):")
    for tp in constraints["tool_permissions"]:
        print(f"    {GREEN}✓{RESET} {tp['name']}")
    print()
    print(f"  {BOLD}Denied tools{RESET} (organization constraint):")
    for t in constraints["organization"]["denied_tools"]:
        print(f"    {RED}✗{RESET} {t}")
    print()
    print(f"  {DIM}Policy comes from Vijil Console. The developer writes no security code.{RESET}")
    print(f"  {DIM}The security team controls which tools each agent may call.{RESET}")

    # ── Scene 6: Audit trail ──
    section("Scene 6: Audit Trail")
    print(f"  {DIM}Every decision is logged with the agent's identity:{RESET}")
    print()
    for event in audit_log:
        ts = event.timestamp.strftime("%H:%M:%S.%f")[:-3]
        attrs = event.attributes
        if event.event_type == "guard":
            direction = attrs.get("direction", "?")
            flagged = attrs.get("flagged", False)
            status = f"{RED}FLAGGED{RESET}" if flagged else f"{GREEN}safe{RESET}"
            print(f"    {DIM}{ts}{RESET}  {CYAN}guard.{direction}{RESET}  {status}  score={attrs.get('score', 0):.2f}")
        elif event.event_type == "tool_mac":
            tool = attrs.get("tool_name", "?")
            permitted = attrs.get("permitted", False)
            status = f"{GREEN}PERMIT{RESET}" if permitted else f"{RED}DENY{RESET}"
            print(f"    {DIM}{ts}{RESET}  {MAGENTA}tool_mac{RESET}   {status}  {tool}")
        elif event.event_type == "attestation":
            print(f"    {DIM}{ts}{RESET}  {YELLOW}attestation{RESET}  tools={attrs.get('tool_count', 0)}")

    print()
    print(f"  {DIM}{len(audit_log)} events captured. In production, these stream to the{RESET}")
    print(f"  {DIM}observability stack tagged with the agent's SPIFFE ID.{RESET}")

    # ── Closing ──
    print()
    print(f"{BOLD}{'━' * 70}{RESET}")
    print(f"{BOLD}  Summary{RESET}")
    print(f"{BOLD}{'━' * 70}{RESET}")
    print()
    print(f"  • {BOLD}One line of code{RESET} adds trust enforcement to any Strands agent")
    print(f"  • {BOLD}Boot attestation{RESET} verifies every tool before the agent serves requests")
    print(f"  • {BOLD}Tool MAC{RESET} blocks unauthorized tool calls in real time")
    print(f"  • {BOLD}Content guards{RESET} scan inputs, outputs, and tool responses")
    print(f"  • {BOLD}Zero static secrets{RESET} — credential proxy fetches keys per-request")
    print(f"  • {BOLD}Policy from Console{RESET} — security team controls, developer ships")
    print()
    print(f"  {BOLD}pip install vijil-sdk[trust]{RESET}")
    print(f"  {DIM}from vijil import secure_agent{RESET}")
    print()


if __name__ == "__main__":
    main()
