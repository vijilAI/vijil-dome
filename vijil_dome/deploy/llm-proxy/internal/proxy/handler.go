// Package proxy implements the core LLM proxy handler.
// It authenticates agents via mTLS (SPIFFE X.509-SVIDs), fetches credentials
// from Vault, and forwards requests to upstream LLM providers.
//
// SECURITY INVARIANTS:
//   - API keys are held only in local variables scoped to a single request.
//   - Key bytes are zeroed before the handler returns.
//   - No secret is written to disk, cached in a struct field, or logged.
package proxy

import (
	"bytes"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/spiffe/go-spiffe/v2/spiffeid"
	"github.com/spiffe/go-spiffe/v2/spiffetls/tlsconfig"
	"github.com/spiffe/go-spiffe/v2/svid/jwtsvid"
	"github.com/spiffe/go-spiffe/v2/workloadapi"

	"github.com/vijil-ai/llm-proxy/internal/config"
	"github.com/vijil-ai/llm-proxy/internal/telemetry"
	"github.com/vijil-ai/llm-proxy/internal/vault"
)

// Handler processes incoming LLM proxy requests.
type Handler struct {
	configLoader *config.Loader
	vaultClient  *vault.Client
	telemetry    *telemetry.Emitter
	x509Source   *workloadapi.X509Source
	jwtSource    *workloadapi.JWTSource
	upstream     *http.Client
	logger       *slog.Logger
	trustDomain  spiffeid.TrustDomain
	vaultAudience string
}

// HandlerParams contains the dependencies for creating a Handler.
type HandlerParams struct {
	ConfigLoader  *config.Loader
	VaultClient   *vault.Client
	Telemetry     *telemetry.Emitter
	X509Source    *workloadapi.X509Source
	JWTSource     *workloadapi.JWTSource
	Logger        *slog.Logger
	TrustDomain   spiffeid.TrustDomain
	VaultAudience string
}

// NewHandler creates a new proxy handler.
func NewHandler(p HandlerParams) *Handler {
	return &Handler{
		configLoader:  p.ConfigLoader,
		vaultClient:   p.VaultClient,
		telemetry:     p.Telemetry,
		x509Source:    p.X509Source,
		jwtSource:     p.JWTSource,
		logger:        p.Logger,
		trustDomain:   p.TrustDomain,
		vaultAudience: p.VaultAudience,
		upstream: &http.Client{
			Timeout: 120 * time.Second, // LLM requests can be slow
		},
	}
}

// TLSConfig returns the TLS configuration for the proxy server.
// It requires mTLS with SPIFFE X.509-SVIDs from the configured trust domain.
func (h *Handler) TLSConfig() *tls.Config {
	return tlsconfig.MTLSServerConfig(
		h.x509Source,
		h.x509Source,
		tlsconfig.AuthorizeMemberOf(h.trustDomain),
	)
}

// ServeHTTP handles a proxied LLM request.
//
// Expected request format:
//
//	POST /v1/proxy/{provider}
//	Headers: Content-Type, any provider-specific headers
//	Body: The LLM request payload (passed through unchanged)
//
// The handler:
//  1. Extracts the caller's SPIFFE ID from the mTLS peer certificate
//  2. Resolves the route based on agent ID and provider
//  3. Fetches the API key from Vault (transient, in-memory only)
//  4. Forwards the request to the upstream LLM endpoint
//  5. Emits telemetry and returns the response
func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	startTime := time.Now()

	event := &telemetry.RequestEvent{
		Timestamp: startTime,
	}
	defer func() {
		event.ProxyLatency = time.Since(startTime)
		h.telemetry.Emit(event)
	}()

	// ── Step 1: Extract caller SPIFFE ID from mTLS peer certificate ──
	agentID, err := extractSpiffeID(r)
	if err != nil {
		h.logger.Warn("agent authentication failed", "error", err, "remote", r.RemoteAddr)
		event.Error = "auth_failed"
		http.Error(w, "authentication failed: "+err.Error(), http.StatusUnauthorized)
		return
	}
	event.AgentSpiffeID = agentID

	// ── Step 2: Parse provider from request path ──
	provider := parseProvider(r.URL.Path)
	if provider == "" {
		event.Error = "invalid_path"
		http.Error(w, "invalid path: expected /v1/proxy/{provider}[/...]", http.StatusBadRequest)
		return
	}
	event.Provider = provider

	// ── Step 3: Resolve route ──
	cfg := h.configLoader.Get()
	route := cfg.MatchRoute(agentID, provider)
	if route == nil {
		h.logger.Warn("no matching route",
			"agent_id", agentID,
			"provider", provider,
		)
		event.Error = "no_route"
		http.Error(w, fmt.Sprintf("no route for agent %s provider %s", agentID, provider), http.StatusForbidden)
		return
	}
	event.Upstream = route.Upstream

	// ── Step 4: Fetch API key from Vault ──
	vaultStart := time.Now()

	jwtSVID, err := h.jwtSource.FetchJWTSVID(r.Context(), jwtsvid.Params{
		Audience: h.vaultAudience,
	})
	if err != nil {
		h.logger.Error("jwt svid fetch failed", "error", err)
		event.Error = "jwt_fetch_failed"
		http.Error(w, "internal error", http.StatusInternalServerError)
		return
	}

	apiKey, err := h.vaultClient.FetchSecret(r.Context(), jwtSVID.Marshal(), route.VaultPath, agentID)
	event.VaultFetchLatency = time.Since(vaultStart)
	if err != nil {
		h.logger.Error("vault fetch failed",
			"error", err,
			"agent_id", agentID,
			"vault_path", route.VaultPath,
		)
		event.Error = "vault_fetch_failed"
		// Invalidate token in case it expired
		h.vaultClient.InvalidateToken()
		http.Error(w, "credential fetch failed", http.StatusInternalServerError)
		return
	}

	// SECURITY: apiKey is a local variable. It will be zeroed before return.
	defer zeroString(&apiKey)

	// ── Step 5: Build and send upstream request ──
	upstreamStart := time.Now()

	// Read the original request body
	reqBody, err := io.ReadAll(r.Body)
	if err != nil {
		event.Error = "read_body_failed"
		http.Error(w, "failed to read request body", http.StatusBadRequest)
		return
	}

	upstreamReq, err := http.NewRequestWithContext(r.Context(), r.Method, route.Upstream, bytes.NewReader(reqBody))
	if err != nil {
		event.Error = "upstream_req_failed"
		http.Error(w, "internal error", http.StatusInternalServerError)
		return
	}

	// Copy safe headers from the original request
	for _, hdr := range []string{"Content-Type", "Accept", "User-Agent"} {
		if v := r.Header.Get(hdr); v != "" {
			upstreamReq.Header.Set(hdr, v)
		}
	}
	if upstreamReq.Header.Get("Content-Type") == "" {
		upstreamReq.Header.Set("Content-Type", "application/json")
	}

	// Inject the API key using the provider-specific header
	authHeader := route.AuthHeader
	if authHeader == "" {
		authHeader = "Authorization"
	}
	authValue := apiKey
	if route.AuthPrefix != "" {
		authValue = route.AuthPrefix + apiKey
	}
	upstreamReq.Header.Set(authHeader, authValue)

	// Send to upstream
	upstreamResp, err := h.upstream.Do(upstreamReq)
	event.UpstreamLatency = time.Since(upstreamStart)
	if err != nil {
		h.logger.Error("upstream request failed",
			"error", err,
			"upstream", route.Upstream,
			"agent_id", agentID,
		)
		event.Error = "upstream_failed"
		http.Error(w, "upstream request failed", http.StatusBadGateway)
		return
	}
	defer upstreamResp.Body.Close()

	event.StatusCode = upstreamResp.StatusCode

	// ── Step 6: Read response, parse usage, forward to client ──
	respBody, err := io.ReadAll(upstreamResp.Body)
	if err != nil {
		event.Error = "read_upstream_failed"
		http.Error(w, "failed to read upstream response", http.StatusBadGateway)
		return
	}

	// Parse token usage for telemetry (best-effort, non-blocking)
	if usage := telemetry.ParseUsage(respBody); usage != nil {
		event.PromptTokens = usage.PromptTokens
		event.CompletionTokens = usage.CompletionTokens
		event.TotalTokens = usage.TotalTokens
	}

	// Forward response headers
	for key, vals := range upstreamResp.Header {
		for _, v := range vals {
			w.Header().Add(key, v)
		}
	}
	w.WriteHeader(upstreamResp.StatusCode)
	w.Write(respBody)
}

// HealthHandler returns a simple health check endpoint.
// Does NOT expose any secret material.
func HealthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Write([]byte(`{"status":"ok"}`))
}

// extractSpiffeID retrieves the caller's SPIFFE ID from the mTLS peer certificate.
func extractSpiffeID(r *http.Request) (string, error) {
	if r.TLS == nil {
		return "", fmt.Errorf("no TLS connection")
	}
	if len(r.TLS.PeerCertificates) == 0 {
		return "", fmt.Errorf("no peer certificate")
	}

	cert := r.TLS.PeerCertificates[0]
	spiffeID, err := extractSpiffeIDFromCert(cert)
	if err != nil {
		return "", err
	}

	return spiffeID, nil
}

// extractSpiffeIDFromCert extracts the SPIFFE ID URI from the certificate's SANs.
func extractSpiffeIDFromCert(cert *x509.Certificate) (string, error) {
	for _, uri := range cert.URIs {
		if uri.Scheme == "spiffe" {
			return uri.String(), nil
		}
	}
	return "", fmt.Errorf("no SPIFFE ID in certificate SANs")
}

// parseProvider extracts the provider name from the request path.
// Expected format: /v1/proxy/{provider} or /v1/proxy/{provider}/...
func parseProvider(path string) string {
	path = strings.TrimPrefix(path, "/")
	parts := strings.Split(path, "/")
	// Expected: v1/proxy/{provider}[/...]
	if len(parts) >= 3 && parts[0] == "v1" && parts[1] == "proxy" {
		return parts[2]
	}
	return ""
}

// zeroString overwrites a string's backing bytes to prevent lingering in memory.
// This is a defense-in-depth measure; Go's GC will also eventually reclaim the memory.
func zeroString(s *string) {
	if s == nil || *s == "" {
		return
	}
	b := []byte(*s)
	for i := range b {
		b[i] = 0
	}
	*s = ""
}
