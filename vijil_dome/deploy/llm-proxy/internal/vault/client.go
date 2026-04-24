// Package vault provides a client for fetching LLM API keys from HashiCorp Vault
// using SPIFFE JWT-SVIDs for authentication. Credentials are held only in memory
// and scoped to the lifetime of a single request.
package vault

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"time"
)

// Client authenticates to Vault using JWT-SVIDs and retrieves secrets.
type Client struct {
	addr      string
	authMount string
	role      string
	httpClient *http.Client
	logger     *slog.Logger

	// Cached Vault token (short-lived, refreshed on expiry)
	mu         sync.RWMutex
	token      string
	tokenExpiry time.Time
}

// NewClient creates a new Vault client.
func NewClient(addr, authMount, role string, logger *slog.Logger) *Client {
	return &Client{
		addr:      strings.TrimRight(addr, "/"),
		authMount: authMount,
		role:      role,
		httpClient: &http.Client{
			Timeout: 10 * time.Second,
		},
		logger: logger,
	}
}

// FetchSecret retrieves the API key from the given Vault KV v2 path.
// jwtSVID is the proxy's JWT-SVID used to authenticate to Vault.
// callerSpiffeID is the originating agent's SPIFFE ID, included as metadata for audit.
// Returns the API key value and any error.
//
// SECURITY: The returned key should be used immediately and not stored beyond
// the scope of the current request handler.
func (c *Client) FetchSecret(ctx context.Context, jwtSVID, vaultPath, callerSpiffeID string) (string, error) {
	// Authenticate to Vault if needed
	token, err := c.ensureToken(ctx, jwtSVID)
	if err != nil {
		return "", fmt.Errorf("vault auth: %w", err)
	}

	// Read the secret
	secret, err := c.readSecret(ctx, token, vaultPath, callerSpiffeID)
	if err != nil {
		return "", fmt.Errorf("vault read %s: %w", vaultPath, err)
	}

	return secret, nil
}

func (c *Client) ensureToken(ctx context.Context, jwtSVID string) (string, error) {
	c.mu.RLock()
	if c.token != "" && time.Now().Before(c.tokenExpiry) {
		tok := c.token
		c.mu.RUnlock()
		return tok, nil
	}
	c.mu.RUnlock()

	return c.authenticate(ctx, jwtSVID)
}

func (c *Client) authenticate(ctx context.Context, jwtSVID string) (string, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Double-check after acquiring write lock
	if c.token != "" && time.Now().Before(c.tokenExpiry) {
		return c.token, nil
	}

	url := fmt.Sprintf("%s/v1/auth/%s/login", c.addr, c.authMount)

	body := map[string]string{
		"jwt":  jwtSVID,
		"role": c.role,
	}
	bodyBytes, err := json.Marshal(body)
	if err != nil {
		return "", fmt.Errorf("marshal auth request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(bodyBytes))
	if err != nil {
		return "", fmt.Errorf("create auth request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("auth request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("auth failed (status %d): %s", resp.StatusCode, string(respBody))
	}

	var authResp vaultAuthResponse
	if err := json.NewDecoder(resp.Body).Decode(&authResp); err != nil {
		return "", fmt.Errorf("decode auth response: %w", err)
	}

	c.token = authResp.Auth.ClientToken
	// Set expiry with a safety margin (renew at 2/3 of lease)
	leaseDuration := time.Duration(authResp.Auth.LeaseDuration) * time.Second
	c.tokenExpiry = time.Now().Add(leaseDuration * 2 / 3)

	c.logger.Info("vault authentication successful",
		"lease_duration", leaseDuration,
		"renewable", authResp.Auth.Renewable,
	)

	return c.token, nil
}

func (c *Client) readSecret(ctx context.Context, token, path, callerSpiffeID string) (string, error) {
	// KV v2 paths: secret/data/... (the "data" is part of the API path)
	url := fmt.Sprintf("%s/v1/%s", c.addr, path)

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return "", fmt.Errorf("create read request: %w", err)
	}
	req.Header.Set("X-Vault-Token", token)
	// Pass caller SPIFFE ID for audit trail
	req.Header.Set("X-Caller-SPIFFE-ID", callerSpiffeID)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("read request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusForbidden {
		return "", fmt.Errorf("access denied for caller %s at path %s", callerSpiffeID, path)
	}
	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("read failed (status %d): %s", resp.StatusCode, string(respBody))
	}

	var secretResp vaultSecretResponse
	if err := json.NewDecoder(resp.Body).Decode(&secretResp); err != nil {
		return "", fmt.Errorf("decode secret response: %w", err)
	}

	apiKey, ok := secretResp.Data.Data["api_key"]
	if !ok {
		return "", fmt.Errorf("secret at %s missing 'api_key' field", path)
	}

	keyStr, ok := apiKey.(string)
	if !ok {
		return "", fmt.Errorf("secret at %s: api_key is not a string", path)
	}

	return keyStr, nil
}

// InvalidateToken clears the cached Vault token, forcing re-authentication
// on the next request. Call this if a request fails with 403.
func (c *Client) InvalidateToken() {
	c.mu.Lock()
	c.token = ""
	c.tokenExpiry = time.Time{}
	c.mu.Unlock()
}

// Vault API response types

type vaultAuthResponse struct {
	Auth struct {
		ClientToken   string `json:"client_token"`
		LeaseDuration int    `json:"lease_duration"`
		Renewable     bool   `json:"renewable"`
	} `json:"auth"`
}

type vaultSecretResponse struct {
	Data struct {
		Data map[string]interface{} `json:"data"`
	} `json:"data"`
}
