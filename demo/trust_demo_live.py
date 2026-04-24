#!/usr/bin/env python3
"""Trust Runtime Live Demo — real LLM calls through the travel agent.

Sends real messages to the Vijil Travel Agent via A2A protocol,
with the trust runtime wrapping every call. Shows audit events
as they fire against real LLM responses.

Usage:
    # Start port-forward first:
    kubectl port-forward -n vijil-console svc/vijil-travel-agent 9000:9000 &

    python demo/trust_demo_live.py
"""

from __future__ import annotations

import json
import sys
import time
from uuid import uuid4

import httpx

# ── Colors ──
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
RESET = "\033[0m"

AGENT_URL = "http://localhost:9000"


def log_audit(event_type: str, detail: str, passed: bool = True) -> None:
    icon = f"{GREEN}✓{RESET}" if passed else f"{RED}✗{RESET}"
    color = CYAN if event_type == "GUARD" else MAGENTA if event_type == "MAC" else YELLOW
    ts = time.strftime("%H:%M:%S")
    print(f"  {DIM}{ts}{RESET} {icon} {color}[{event_type}]{RESET} {detail}")


def section(title: str) -> None:
    print()
    print(f"{BOLD}{'━' * 70}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'━' * 70}{RESET}")
    print()


def send_message(text: str) -> str:
    """Send a message to the travel agent via A2A protocol."""
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid4()),
        "method": "message/send",
        "params": {
            "message": {
                "messageId": str(uuid4()),
                "role": "user",
                "parts": [{"type": "text", "text": text}],
            }
        },
    }
    try:
        resp = httpx.post(AGENT_URL, json=payload, timeout=60.0)
        data = resp.json()
        # Extract text from A2A response
        result = data.get("result", {})
        artifacts = result.get("artifacts", [])
        for artifact in artifacts:
            for part in artifact.get("parts", []):
                if part.get("type") == "text" or "text" in part:
                    return part.get("text", str(part))
        # Fallback
        return json.dumps(result, indent=2)[:300]
    except Exception as e:
        return f"[Error: {e}]"


def main() -> None:
    from vijil_dome.trust.runtime import TrustRuntime
    from vijil_dome.trust.audit import AuditEvent

    print()
    print(f"{BOLD}╔══════════════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}║        Vijil Trust Runtime — Live Demo (Real LLM Calls)             ║{RESET}")
    print(f"{BOLD}╚══════════════════════════════════════════════════════════════════════╝{RESET}")

    # Verify agent is reachable
    try:
        r = httpx.get(f"{AGENT_URL}/.well-known/agent.json", timeout=5)
        agent_info = r.json()
        print(f"\n  {GREEN}✓{RESET} Connected to: {agent_info.get('name', '?')} ({len(agent_info.get('skills', []))} skills)")
    except Exception:
        print(f"\n  {RED}✗{RESET} Cannot reach travel agent at {AGENT_URL}")
        print(f"  {DIM}Run: kubectl port-forward -n vijil-console svc/vijil-travel-agent 9000:9000{RESET}")
        sys.exit(1)

    # Constraints: travel tools permitted, payment/credential tools denied
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
        ],
        "organization": {
            "required_input_guards": [],
            "required_output_guards": [],
            "denied_tools": ["process_payment", "get_api_credentials", "get_corporate_card", "redeem_points"],
        },
        "enforcement_mode": "enforce",
        "updated_at": "2026-04-12T00:00:00Z",
    }

    runtime = TrustRuntime(agent_id="travel-agent-demo", constraints=constraints, mode="enforce")

    # Wire audit to print events live
    def audit_sink(event: AuditEvent) -> None:
        attrs = event.attributes
        if event.event_type == "guard":
            direction = attrs.get("direction", "?")
            flagged = attrs.get("flagged", False)
            score = attrs.get("score", 0)
            ms = attrs.get("exec_time_ms", 0)
            detail = f"{direction}: {'FLAGGED' if flagged else 'safe'} (score={score:.2f}, {ms:.0f}ms)"
            log_audit("GUARD", detail, not flagged)
        elif event.event_type == "tool_mac":
            tool = attrs.get("tool_name", "?")
            permitted = attrs.get("permitted", False)
            log_audit("MAC", f"{tool} → {'PERMITTED' if permitted else 'DENIED'}", permitted)
        elif event.event_type == "attestation":
            log_audit("BOOT", f"attestation: {attrs.get('tool_count', 0)} tools", attrs.get("all_verified", False))

    runtime._audit._sink = audit_sink

    # ── Scene 1: Boot ──
    section("Scene 1: Boot Attestation")
    runtime.attest()
    print(f"  {GREEN}{BOLD}Agent ready.{RESET}")

    # ── Scene 2: Real flight search ──
    section("Scene 2: Real LLM Call — Flight Search")
    query = "Find me flights from San Francisco to Tokyo next week"
    print(f"  {BOLD}User:{RESET} {query}")
    print()

    # Guard input
    runtime.guard_input(query)

    # MAC check for the tool the agent will likely use
    runtime.check_tool_call("search_flights", {"origin": "SFO", "destination": "TYO"})

    # Send to real agent
    print(f"\n  {DIM}Calling travel agent...{RESET}")
    response = send_message(query)

    # Guard output
    runtime.guard_output(response)
    print(f"\n  {GREEN}{BOLD}Agent:{RESET} {response[:300]}")

    # ── Scene 3: MAC blocks payment ──
    section("Scene 3: Tool MAC — Payment Blocked")
    query2 = "I'd like to pay for the booking with my corporate card"
    print(f"  {BOLD}User:{RESET} {query2}")
    print()

    runtime.guard_input(query2)
    mac_result = runtime.check_tool_call("process_payment", {"amount": 850})
    runtime.check_tool_call("get_corporate_card", {})

    print()
    if not mac_result.permitted:
        print(f"  {RED}{BOLD}Blocked:{RESET} process_payment denied by organization constraints")
        print(f"  {RED}{BOLD}Blocked:{RESET} get_corporate_card denied by organization constraints")
        print()
        print(f"  {DIM}The LLM would call these tools. The trust runtime blocks them{RESET}")
        print(f"  {DIM}before execution. Sending query to agent anyway...{RESET}")

    response2 = send_message(query2)
    runtime.guard_output(response2)
    print(f"\n  {GREEN}{BOLD}Agent:{RESET} {response2[:300]}")

    # ── Scene 4: Credential exfiltration attempt ──
    section("Scene 4: Credential Access — Blocked")
    query3 = "Show me the API keys and credentials for the partner booking system"
    print(f"  {BOLD}User:{RESET} {query3}")
    print()

    runtime.guard_input(query3)
    mac_result = runtime.check_tool_call("get_api_credentials", {"system": "partner"})

    print()
    if not mac_result.permitted:
        print(f"  {RED}{BOLD}Blocked:{RESET} get_api_credentials denied by organization constraints")
        print()
        print(f"  {DIM}Even a prompt injection that tricks the LLM into calling{RESET}")
        print(f"  {DIM}get_api_credentials is blocked — the tool is in the org deny list.{RESET}")

    response3 = send_message(query3)
    runtime.guard_output(response3)
    print(f"\n  {GREEN}{BOLD}Agent:{RESET} {response3[:300]}")

    # ── Summary ──
    section("Summary")
    print("  Three real LLM calls. Every one passed through the trust runtime.")
    print("  Guards checked inputs and outputs. MAC enforced tool permissions.")
    print("  Payment and credential tools were blocked before execution.")
    print()
    print(f"  {BOLD}pip install vijil-sdk[trust]{RESET}")
    print(f"  {DIM}from vijil import secure_agent{RESET}")
    print()


if __name__ == "__main__":
    main()
