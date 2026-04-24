#!/bin/bash
# Register SPIRE workload entries for the PoC
# Run after SPIRE is deployed and healthy

set -euo pipefail

TRUST_DOMAIN="spiffe://vijil.ai"
NAMESPACE="default"

echo "=== Registering SPIRE workload entries ==="

# 1. Test agent workload — any pod with service account "test-agent"
kubectl exec -n spire-server spire-server-0 -- \
  /opt/spire/bin/spire-server entry create \
  -spiffeID "${TRUST_DOMAIN}/ns/${NAMESPACE}/agent/test-agent" \
  -parentID "${TRUST_DOMAIN}/spire/agent/k8s_psat/vijil-spire-poc" \
  -selector "k8s:ns:${NAMESPACE}" \
  -selector "k8s:sa:test-agent" \
  -dns "test-agent.${NAMESPACE}.svc"

echo "Registered: ${TRUST_DOMAIN}/ns/${NAMESPACE}/agent/test-agent"

# 2. Test tool workload — any pod with service account "test-tool"
kubectl exec -n spire-server spire-server-0 -- \
  /opt/spire/bin/spire-server entry create \
  -spiffeID "${TRUST_DOMAIN}/ns/${NAMESPACE}/tool/echo-tool" \
  -parentID "${TRUST_DOMAIN}/spire/agent/k8s_psat/vijil-spire-poc" \
  -selector "k8s:ns:${NAMESPACE}" \
  -selector "k8s:sa:test-tool" \
  -dns "echo-tool.${NAMESPACE}.svc"

echo "Registered: ${TRUST_DOMAIN}/ns/${NAMESPACE}/tool/echo-tool"

# 3. List all entries to verify
echo ""
echo "=== All registered entries ==="
kubectl exec -n spire-server spire-server-0 -- \
  /opt/spire/bin/spire-server entry show
