# Trust Runtime Deployment Guide

Deploy the Vijil Trust Runtime infrastructure on a Kubernetes cluster.

## Prerequisites

- Kubernetes cluster (EKS, GKE, or DOKS) with kubectl access
- Helm 3 installed
- PostgreSQL database accessible from the cluster (shared with Console, Darwin, Swarm)
- `vijil-sdk` PR #1 merged (or branch checked out for the deploy artifacts)

## Architecture

```
Your Kubernetes Cluster
├── spire-server namespace
│   ├── SPIRE Server (StatefulSet) — root of trust, CA, registration
│   ├── SPIRE Agent (DaemonSet) — one per node, attests workloads
│   ├── CSI Driver (DaemonSet) — mounts SPIFFE socket into pods
│   ├── OIDC Discovery Provider — JWKS for Vault JWT auth
│   └── Identity Delegate — JWT-SVID for managed runtimes
│
├── vijil-proxy namespace
│   ├── LLM Proxy (Deployment) — mTLS + Vault credential injection
│   └── Vault (StatefulSet) — stores LLM provider API keys
│
└── your-app namespace
    └── Your Agent (with label vijil.ai/trust-runtime: enabled)
        └── Trust Runtime (in-process via vijil-sdk)
```

## Step 1: Create the SPIRE database

SPIRE uses a PostgreSQL database for registration entries and CA state. Create a new database in the shared managed PostgreSQL instance:

```sql
-- Connect to your managed PostgreSQL
CREATE DATABASE vijil_enterprise_spire;
```

Or if using a shared database with separate schemas:

```sql
CREATE SCHEMA IF NOT EXISTS spire;
```

## Step 2: Install SPIRE

```bash
# Add the SPIRE Helm repo
helm repo add spiffe https://spiffe.github.io/helm-charts-hardened/
helm repo update

# Install CRDs first
helm install spire-crds spiffe/spire-crds \
  --namespace spire-server \
  --create-namespace

# Install SPIRE with production values
helm install spire spiffe/spire \
  -f deploy/spire-values-production.yaml \
  --namespace spire-server \
  --set spire-server.dataStore.sql.connectionString="dbname=vijil_enterprise_spire host=YOUR_PG_HOST port=25060 user=doadmin password=YOUR_PASSWORD sslmode=require"
```

Verify SPIRE is healthy:

```bash
kubectl exec -n spire-server spire-server-0 -c spire-server -- \
  /opt/spire/bin/spire-server healthcheck
# Expected: Server is healthy.

kubectl get pods -n spire-server
# Expected: spire-server, spire-agent (on each node), csi-driver, oidc-discovery-provider
```

## Step 3: Apply automated workload registration

```bash
kubectl apply -f deploy/spire-cluster-spiffeid.yaml
```

This creates `ClusterSPIFFEID` resources that automatically register any pod with:
- `vijil.ai/trust-runtime: enabled` → gets an agent SPIFFE ID
- `vijil.ai/tool-identity: enabled` → gets a tool SPIFFE ID
- `app: llm-proxy` → gets a proxy SPIFFE ID

No manual `spire-server entry create` needed.

## Step 4: Install Vault

```bash
helm repo add hashicorp https://helm.releases.hashicorp.com
helm repo update

helm install vault hashicorp/vault \
  -f deploy/vault-production.yaml \
  --namespace vijil-proxy \
  --create-namespace
```

Initialize and unseal Vault:

```bash
# Initialize (first time only — save the unseal keys securely)
kubectl exec -n vijil-proxy vault-0 -- vault operator init -key-shares=1 -key-threshold=1
# Save the Unseal Key and Root Token

# Unseal
kubectl exec -n vijil-proxy vault-0 -- vault operator unseal YOUR_UNSEAL_KEY
```

Configure JWT auth and store API keys:

```bash
# Set environment
export VAULT_ADDR="http://vault.vijil-proxy.svc:8200"
export VAULT_TOKEN="YOUR_ROOT_TOKEN"

# Run the setup script (adapts to production Vault)
bash deploy/llm-proxy/vault-setup.sh
```

## Step 5: Deploy the LLM Proxy

```bash
# Build and push the proxy image (requires Go or use pre-built)
docker build -t YOUR_REGISTRY/vijil-llm-proxy:latest deploy/llm-proxy/
docker push YOUR_REGISTRY/vijil-llm-proxy:latest

# Update the image in the manifest
kubectl apply -f deploy/llm-proxy/k8s.yaml
```

Verify the proxy acquired its SPIFFE identity:

```bash
kubectl logs -n vijil-proxy deploy/llm-proxy | grep "proxy SVID acquired"
# Expected: proxy SVID acquired spiffe://vijil.ai/ns/vijil-proxy/sa/llm-proxy
```

## Step 6: Deploy the Identity Delegate (optional)

Required only for agents running in managed runtimes (AgentCore) that cannot access the SPIRE Agent socket.

```bash
kubectl apply -f deploy/identity-delegate/k8s.yaml
```

## Step 7: Deploy your agent with trust enforcement

Add the trust runtime label and install the SDK:

```yaml
# In your agent's Deployment:
metadata:
  labels:
    vijil.ai/trust-runtime: "enabled"  # Auto-registers with SPIRE
spec:
  containers:
    - name: agent
      env:
        - name: TRUST_MODE
          value: "enforce"  # or "warn" for development
```

In your agent's Dockerfile:

```dockerfile
RUN pip install vijil-sdk[trust]
```

In your agent's code:

```python
from vijil import secure_agent
app = secure_agent(my_agent, agent_id="travel-agent")
```

## Verification

```bash
# 1. SPIRE healthy
kubectl exec -n spire-server spire-server-0 -c spire-server -- \
  /opt/spire/bin/spire-server healthcheck

# 2. Agent got SVID
kubectl exec -n your-namespace your-agent-pod -- \
  python3 -c "from vijil.trust.identity import AgentIdentity; i = AgentIdentity(); print(i.spiffe_id)"

# 3. Proxy serves mTLS
kubectl logs -n vijil-proxy deploy/llm-proxy | grep "proxy SVID acquired"

# 4. Vault JWT auth works
kubectl logs -n vijil-proxy deploy/llm-proxy | grep "vault authentication successful"
```

## Tear down (PoC only)

```bash
helm uninstall spire --namespace spire-server
helm uninstall spire-crds --namespace spire-server
helm uninstall vault --namespace vijil-proxy
kubectl delete -f deploy/llm-proxy/k8s.yaml
kubectl delete -f deploy/identity-delegate/k8s.yaml
kubectl delete -f deploy/spire-cluster-spiffeid.yaml
kubectl delete namespace spire-server vijil-proxy
```

## Cost estimate

| Component | Resource | Monthly cost |
|---|---|---|
| SPIRE Server | 1 pod, 256MB | ~$5 |
| SPIRE Agent | DaemonSet (per node) | Included in node cost |
| Vault | 1 pod, 512MB + 5GB PVC | ~$10 |
| LLM Proxy | 1 pod, 128MB | ~$3 |
| Identity Delegate | 1 pod, 128MB | ~$3 |
| **Total infrastructure** | | **~$21/month** |

Cluster node cost is separate and depends on your workload.
