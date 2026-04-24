// Package config provides routing configuration for the LLM proxy.
// Routes map agent SPIFFE IDs to upstream LLM providers and Vault secret paths.
// The config file is watched for changes and hot-reloaded without restart.
package config

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// Route defines a mapping from an agent identity pattern to an upstream LLM endpoint.
type Route struct {
	// Match criteria
	AgentIDPattern string `json:"agent_id_pattern"` // glob pattern, e.g. "spiffe://vijil.ai/agent/research-*"
	Provider       string `json:"provider"`         // e.g. "anthropic", "openai", "google"

	// Upstream target
	Upstream  string `json:"upstream"`   // e.g. "https://api.anthropic.com/v1/messages"
	VaultPath string `json:"vault_path"` // e.g. "secret/data/llm/anthropic"

	// Provider-specific header name for the API key
	AuthHeader string `json:"auth_header"` // e.g. "x-api-key" for Anthropic, "Authorization" for OpenAI
	AuthPrefix string `json:"auth_prefix"` // e.g. "Bearer " for OpenAI, "" for Anthropic
}

// Config holds the complete proxy configuration.
type Config struct {
	ListenAddr      string `json:"listen_addr"`       // e.g. ":8443"
	SpiffeSocket    string `json:"spiffe_socket"`      // e.g. "unix:///run/spire/sockets/agent.sock"
	VaultAddr       string `json:"vault_addr"`         // e.g. "https://vault.vijil.ai:8200"
	VaultAuthMount  string `json:"vault_auth_mount"`   // e.g. "jwt" or "spiffe"
	VaultAudience   string `json:"vault_audience"`     // audience for JWT-SVID when authenticating to Vault
	VaultRole       string `json:"vault_role"`          // Vault role for the proxy
	OTELEndpoint    string `json:"otel_endpoint"`      // e.g. "localhost:4317"
	Routes          []Route `json:"routes"`
}

// Loader manages loading and hot-reloading of the config file.
type Loader struct {
	path     string
	mu       sync.RWMutex
	current  *Config
	stopCh   chan struct{}
	logger   *slog.Logger
}

// NewLoader creates a new config loader and performs the initial load.
func NewLoader(path string, logger *slog.Logger) (*Loader, error) {
	l := &Loader{
		path:   path,
		stopCh: make(chan struct{}),
		logger: logger,
	}
	if err := l.load(); err != nil {
		return nil, fmt.Errorf("initial config load: %w", err)
	}
	return l, nil
}

// Get returns the current configuration. Safe for concurrent use.
func (l *Loader) Get() *Config {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return l.current
}

// WatchForChanges polls the config file for modifications and reloads on change.
// Call in a goroutine. Stops when Stop() is called.
func (l *Loader) WatchForChanges(interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	var lastMod time.Time
	if info, err := os.Stat(l.path); err == nil {
		lastMod = info.ModTime()
	}

	for {
		select {
		case <-l.stopCh:
			return
		case <-ticker.C:
			info, err := os.Stat(l.path)
			if err != nil {
				l.logger.Warn("config file stat failed", "error", err)
				continue
			}
			if info.ModTime().After(lastMod) {
				lastMod = info.ModTime()
				if err := l.load(); err != nil {
					l.logger.Error("config reload failed", "error", err)
				} else {
					l.logger.Info("config reloaded", "path", l.path)
				}
			}
		}
	}
}

// Stop halts the file watcher.
func (l *Loader) Stop() {
	close(l.stopCh)
}

func (l *Loader) load() error {
	data, err := os.ReadFile(l.path)
	if err != nil {
		return fmt.Errorf("read config: %w", err)
	}

	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return fmt.Errorf("parse config: %w", err)
	}

	if err := cfg.validate(); err != nil {
		return fmt.Errorf("validate config: %w", err)
	}

	l.mu.Lock()
	l.current = &cfg
	l.mu.Unlock()
	return nil
}

func (c *Config) validate() error {
	if c.ListenAddr == "" {
		return fmt.Errorf("listen_addr is required")
	}
	if c.SpiffeSocket == "" {
		return fmt.Errorf("spiffe_socket is required")
	}
	if c.VaultAddr == "" {
		return fmt.Errorf("vault_addr is required")
	}
	if len(c.Routes) == 0 {
		return fmt.Errorf("at least one route is required")
	}
	for i, r := range c.Routes {
		if r.AgentIDPattern == "" {
			return fmt.Errorf("route %d: agent_id_pattern is required", i)
		}
		if r.Upstream == "" {
			return fmt.Errorf("route %d: upstream is required", i)
		}
		if r.VaultPath == "" {
			return fmt.Errorf("route %d: vault_path is required", i)
		}
	}
	return nil
}

// MatchRoute finds the first route matching the given agent SPIFFE ID and provider.
// Returns nil if no route matches.
func (c *Config) MatchRoute(agentID, provider string) *Route {
	for i := range c.Routes {
		r := &c.Routes[i]
		if provider != "" && r.Provider != provider {
			continue
		}
		if matchGlob(r.AgentIDPattern, agentID) {
			return r
		}
	}
	return nil
}

// matchGlob performs simple glob matching with * as wildcard.
func matchGlob(pattern, s string) bool {
	// Simple implementation: split on * and check prefix/suffix/contains
	if pattern == "*" {
		return true
	}
	if !strings.Contains(pattern, "*") {
		return pattern == s
	}

	parts := strings.Split(pattern, "*")
	if len(parts) == 2 {
		return strings.HasPrefix(s, parts[0]) && strings.HasSuffix(s, parts[1])
	}

	// Multi-wildcard: use filepath.Match as fallback
	matched, _ := filepath.Match(pattern, s)
	return matched
}
