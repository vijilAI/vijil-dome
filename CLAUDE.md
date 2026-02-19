# CLAUDE.md - vijil-dome

**Company context:** @../CLAUDE.md

This is Vijil's runtime guardrail library - protects AI agents with input/output guards.

## Quick Links

- **Technical guidance:** @docs/AGENTS.md
- **Development process:** @../DEVELOPMENT-RUNBOOK.md
- **Deployment operations:** @../DEPLOYMENT-RUNBOOK.md

## This Repository

Dome is a **pip-installable Python library** (not a service) providing:

- **Input Guards** - Detect prompt injections, jailbreaks, toxicity
- **Output Guards** - Detect unsafe outputs, mask PII, filter harmful content
- **Detection Methods** - ~20 prebuilt detectors (HuggingFace, LLMs, heuristics)
- **Framework Integrations** - Google ADK, LangChain, MCP, OpenTelemetry

## Current Branch

Check `@../DEPLOYMENT-RUNBOOK.md` for feature branch status.
