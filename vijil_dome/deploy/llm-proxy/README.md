# llm-proxy

A thin, SPIFFE-authenticated proxy for LLM API calls. Replaces static API keys stored on disk with transient, attestation-gated credentials fetched from Vault per-request.

## Architecture

```
Agent (X.509-SVID)
  │
  ├── mTLS ──▶  llm-proxy (X.509-SVID + JWT-SVID)
  │                │
  │                ├── JWT-SVID auth ──▶ Vault (returns API key)
  │                ├── injects key ──▶ Upstream LLM (Anthropic/OpenAI/Google)
  │                └── emits telemetry ──▶ OTel Collector
  │
  └── key never on disk, zeroed after each request
```

## Security Properties

- **No secrets on disk**: API keys exist only in memory, scoped to a single request handler
- **Attestation-gated**: The proxy only receives its SVID if the container image hash, service account, and namespace match the SPIRE registration entry
- **Per-agent authorization**: Vault policies determine which agent SPIFFE IDs can access which provider keys
- **Minimal dependencies**: `go-spiffe`, Vault client, Go stdlib — three direct dependencies
- **Distroless container**: No shell, no package manager, no attack surface beyond the binary

## Project Structure

```
├── cmd/proxy/main.go              # Entrypoint: wires SPIRE, Vault, config, telemetry
├── internal/
│   ├── config/config.go           # Route config with file-watch hot reload
│   ├── proxy/handler.go           # Core handler: mTLS → Vault → upstream → response
│   ├── telemetry/telemetry.go     # Structured event emission at proxy boundary
│   └── vault/client.go            # Vault JWT-SVID auth + secret fetch
├── config.example.json            # Example routing configuration
├── Dockerfile                     # Multi-stage: Go build → distroless runtime
├── go.mod
└── README.md
```

## Prerequisites

- Go 1.22+
- SPIRE Server and Agent running (see deployment section)
- HashiCorp Vault with JWT auth method configured
- LLM provider API keys stored in Vault KV v2

## Build

```bash
go mod tidy
go build -o llm-proxy ./cmd/proxy
```

## Configuration

Copy `config.example.json` and customize:

```json
{
  "listen_addr": ":8443",
  "spiffe_socket": "unix:///run/spire/sockets/agent.sock",
  "vault_addr": "https://vault.vijil.ai:8200",
  "vault_auth_mount": "jwt",
  "vault_audience": "vault.vijil.ai",
  "vault_role": "llm-proxy",
  "routes": [
    {
      "agent_id_pattern": "spiffe://vijil.ai/agent/research-*",
      "provider": "anthropic",
      "upstream": "https://api.anthropic.com/v1/messages",
      "vault_path": "secret/data/llm/anthropic",
      "auth_header": "x-api-key",
      "auth_prefix": ""
    }
  ]
}
```

### Route Fields

| Field | Description | Example |
|-------|-------------|---------|
| `agent_id_pattern` | Glob pattern matching caller's SPIFFE ID | `spiffe://vijil.ai/agent/*` |
| `provider` | Provider name (matched from request path) | `anthropic` |
| `upstream` | Full URL of the upstream LLM endpoint | `https://api.anthropic.com/v1/messages` |
| `vault_path` | KV v2 path in Vault (must contain `api_key` field) | `secret/data/llm/anthropic` |
| `auth_header` | HTTP header for the API key | `x-api-key` (Anthropic), `Authorization` (OpenAI) |
| `auth_prefix` | Prefix prepended to the key value | `""` (Anthropic), `"Bearer "` (OpenAI) |

Routes are evaluated in order; first match wins. Use specific patterns before wildcards.

## Run

```bash
./llm-proxy -config /etc/llm-proxy/config.json
```

## Agent Usage

Agents send requests to the proxy instead of directly to LLM providers:

```
POST https://llm-proxy:8443/v1/proxy/anthropic
Content-Type: application/json

{
  "model": "claude-sonnet-4-20250514",
  "max_tokens": 1024,
  "messages": [{"role": "user", "content": "Hello"}]
}
```

The proxy handles authentication (via mTLS), credential injection, and forwarding.

## Docker Compose Deployment

```yaml
services:
  spire-server:
    image: ghcr.io/spiffe/spire-server:1.11
    volumes:
      - spire-data:/run/spire/data
      - ./spire/server.conf:/etc/spire/server.conf

  spire-agent:
    image: ghcr.io/spiffe/spire-agent:1.11
    volumes:
      - spire-sockets:/run/spire/sockets
      - ./spire/agent.conf:/etc/spire/agent.conf
    depends_on: [spire-server]

  vault:
    image: hashicorp/vault:1.17
    environment:
      VAULT_DEV_ROOT_TOKEN_ID: dev-token
    ports: ["8200:8200"]

  llm-proxy:
    build: .
    volumes:
      - spire-sockets:/run/spire/sockets:ro
      - ./config.json:/etc/llm-proxy/config.json:ro
    depends_on: [spire-agent, vault]

  research-agent:
    image: your-adk-agent:latest
    volumes:
      - spire-sockets:/run/spire/sockets:ro
    environment:
      SPIFFE_ENDPOINT_SOCKET: unix:///run/spire/sockets/agent.sock
      LLM_PROXY_URL: https://llm-proxy:8443
    depends_on: [llm-proxy]

volumes:
  spire-data:
  spire-sockets:
```

## Vault Setup

```bash
# Enable JWT auth (trusts SPIRE's OIDC Discovery Provider)
vault auth enable jwt
vault write auth/jwt/config \
  jwks_url="https://spire-oidc-discovery:8008/keys" \
  default_role="llm-proxy"

# Create policy for the proxy
vault policy write llm-proxy - <<EOF
path "secret/data/llm/*" {
  capabilities = ["read"]
}
EOF

# Create role bound to the proxy's SPIFFE ID
vault write auth/jwt/role/llm-proxy \
  role_type="jwt" \
  bound_subject="spiffe://vijil.ai/proxy/llm-gateway" \
  bound_audiences="vault.vijil.ai" \
  user_claim="sub" \
  token_policies="llm-proxy" \
  token_ttl="5m"

# Store API keys
vault kv put secret/llm/anthropic api_key="sk-ant-..."
vault kv put secret/llm/openai api_key="sk-..."
```

## SPIRE Workload Registration

```bash
# Register the proxy
spire-server entry create \
  -spiffeID spiffe://vijil.ai/proxy/llm-gateway \
  -parentID spiffe://vijil.ai/node/k8s-node \
  -selector k8s:ns:infrastructure \
  -selector k8s:sa:llm-proxy

# Register an agent workload
spire-server entry create \
  -spiffeID spiffe://vijil.ai/agent/research-agent \
  -parentID spiffe://vijil.ai/node/k8s-node \
  -selector k8s:ns:agents \
  -selector k8s:sa:research-agent
```

## Telemetry

The proxy emits structured JSON logs for each request:

```json
{
  "level": "INFO",
  "msg": "llm_request",
  "agent_spiffe_id": "spiffe://vijil.ai/agent/research-agent",
  "provider": "anthropic",
  "status_code": 200,
  "proxy_latency_ms": 12,
  "vault_fetch_latency_ms": 8,
  "upstream_latency_ms": 1450,
  "prompt_tokens": 150,
  "completion_tokens": 512,
  "total_tokens": 662
}
```

Configure an OTel Collector with the `filelog` receiver to ingest these into your observability stack.

## Health Check

```
GET https://llm-proxy:8443/healthz
```

Returns `{"status":"ok"}`. Does not require mTLS (for load balancer probes).

## Design Reference

See `SPIFFE-SPIRE-LLM-Proxy-PRD-Design.docx` for the full product requirements document and detailed architecture.
