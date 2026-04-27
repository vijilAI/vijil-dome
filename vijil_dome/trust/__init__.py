"""Vijil Trust Runtime — signed tool manifests and MAC enforcement."""

from vijil_dome.trust.attestation import AttestationResult, ToolAttestationStatus
from vijil_dome.trust.audit import AuditEmitter, AuditEvent
from vijil_dome.trust.constraints import (
    AgentConstraints,
    DomeGuardConfig,
    OrganizationConstraints,
    ToolPermission,
)
from vijil_dome.trust.guard import DetectorTrace, EnforcementResult, GuardTrace
from vijil_dome.trust.identity import AgentIdentity
from vijil_dome.trust.manifest import ToolEntry, ToolManifest
from vijil_dome.trust.policy import ToolCallResult, ToolPolicy
from vijil_dome.trust.runtime import TrustRuntime

__all__ = [
    "AgentConstraints",
    "AgentIdentity",
    "AttestationResult",
    "AuditEmitter",
    "AuditEvent",
    "DetectorTrace",
    "DomeGuardConfig",
    "EnforcementResult",
    "GuardTrace",
    "OrganizationConstraints",
    "ToolAttestationStatus",
    "ToolCallResult",
    "ToolEntry",
    "ToolManifest",
    "ToolPermission",
    "ToolPolicy",
    "TrustRuntime",
]
