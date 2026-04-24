#!/bin/bash
# Set up Vault for the LLM Proxy PoC
# Run after Vault pod is healthy

set -euo pipefail

VAULT_ADDR="http://vault.vijil-proxy.svc:8200"
VAULT_TOKEN="dev-token"
SPIRE_OIDC_URL="http://spire-spiffe-oidc-discovery-provider.spire-server.svc"

echo "=== Vault Setup for LLM Proxy ==="

# Enable KV v2 secrets engine (dev mode has it at secret/)
kubectl exec -n vijil-proxy deploy/vault -- vault secrets list 2>/dev/null || true

# Enable JWT auth method (trusts SPIRE's OIDC Discovery Provider)
echo "1. Enabling JWT auth..."
kubectl exec -n vijil-proxy deploy/vault -- \
  vault auth enable jwt 2>/dev/null || echo "   (already enabled)"

echo "2. Configuring JWT auth to trust SPIRE OIDC..."
kubectl exec -n vijil-proxy deploy/vault -- \
  vault write auth/jwt/config \
    jwks_url="${SPIRE_OIDC_URL}/.well-known/openid-configuration" \
    default_role="llm-proxy"

# Create policy for the proxy
echo "3. Creating llm-proxy policy..."
kubectl exec -n vijil-proxy deploy/vault -- sh -c '
vault policy write llm-proxy - <<EOF
path "secret/data/llm/*" {
  capabilities = ["read"]
}
EOF
'

# Create role bound to the proxy's SPIFFE ID
echo "4. Creating llm-proxy role..."
kubectl exec -n vijil-proxy deploy/vault -- \
  vault write auth/jwt/role/llm-proxy \
    role_type="jwt" \
    bound_subject="spiffe://vijil.ai/ns/vijil-proxy/sa/llm-proxy" \
    bound_audiences="vault.vijil.ai" \
    user_claim="sub" \
    token_policies="llm-proxy" \
    token_ttl="5m"

# Store test API keys
echo "5. Storing API keys..."
kubectl exec -n vijil-proxy deploy/vault -- \
  vault kv put secret/llm/anthropic api_key="${ANTHROPIC_API_KEY:-sk-ant-test-key}"

kubectl exec -n vijil-proxy deploy/vault -- \
  vault kv put secret/llm/openai api_key="${OPENAI_API_KEY:-sk-test-key}"

echo ""
echo "=== Vault setup complete ==="
echo "  JWT auth: enabled (trusts SPIRE OIDC)"
echo "  Policy: llm-proxy (read secret/data/llm/*)"
echo "  Role: llm-proxy (bound to proxy SPIFFE ID)"
echo "  Secrets: anthropic + openai keys stored"
