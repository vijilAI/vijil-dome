// Package telemetry provides lightweight, structured telemetry emission at
// the proxy boundary. The proxy emits events; a separate collector handles
// aggregation. This package has no access to secrets.
package telemetry

import (
	"encoding/json"
	"log/slog"
	"time"
)

// RequestEvent captures the telemetry for a single proxied LLM request.
type RequestEvent struct {
	// Identity
	AgentSpiffeID string `json:"agent_spiffe_id"`
	Provider      string `json:"provider"`
	Model         string `json:"model,omitempty"`
	Upstream      string `json:"upstream"`

	// Timing
	Timestamp        time.Time     `json:"timestamp"`
	ProxyLatency     time.Duration `json:"proxy_latency_ms"`
	VaultFetchLatency time.Duration `json:"vault_fetch_latency_ms"`
	UpstreamLatency  time.Duration `json:"upstream_latency_ms"`

	// Response
	StatusCode int `json:"status_code"`

	// Token usage (parsed from LLM response body)
	PromptTokens     int `json:"prompt_tokens,omitempty"`
	CompletionTokens int `json:"completion_tokens,omitempty"`
	TotalTokens      int `json:"total_tokens,omitempty"`

	// Error (if any)
	Error string `json:"error,omitempty"`
}

// Emitter writes structured telemetry events.
type Emitter struct {
	logger *slog.Logger
}

// NewEmitter creates a telemetry emitter that writes to the given logger.
// In production, this logger should be configured to output JSON to stdout,
// which the OTel Collector can ingest via the filelog receiver.
func NewEmitter(logger *slog.Logger) *Emitter {
	return &Emitter{logger: logger}
}

// Emit records a completed request event.
func (e *Emitter) Emit(event *RequestEvent) {
	e.logger.Info("llm_request",
		"agent_spiffe_id", event.AgentSpiffeID,
		"provider", event.Provider,
		"model", event.Model,
		"upstream", event.Upstream,
		"status_code", event.StatusCode,
		"proxy_latency_ms", event.ProxyLatency.Milliseconds(),
		"vault_fetch_latency_ms", event.VaultFetchLatency.Milliseconds(),
		"upstream_latency_ms", event.UpstreamLatency.Milliseconds(),
		"prompt_tokens", event.PromptTokens,
		"completion_tokens", event.CompletionTokens,
		"total_tokens", event.TotalTokens,
		"error", event.Error,
	)
}

// Usage represents token usage from an LLM response body.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ParseUsage attempts to extract token usage from an LLM response body.
// Works for both Anthropic and OpenAI response formats.
func ParseUsage(body []byte) *Usage {
	// Both Anthropic and OpenAI include a top-level "usage" object
	var resp struct {
		Usage *Usage `json:"usage"`
	}
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil
	}

	// Anthropic uses "input_tokens"/"output_tokens" instead
	if resp.Usage == nil {
		var anthropicResp struct {
			Usage *struct {
				InputTokens  int `json:"input_tokens"`
				OutputTokens int `json:"output_tokens"`
			} `json:"usage"`
		}
		if err := json.Unmarshal(body, &anthropicResp); err != nil || anthropicResp.Usage == nil {
			return nil
		}
		return &Usage{
			PromptTokens:     anthropicResp.Usage.InputTokens,
			CompletionTokens: anthropicResp.Usage.OutputTokens,
			TotalTokens:      anthropicResp.Usage.InputTokens + anthropicResp.Usage.OutputTokens,
		}
	}

	return resp.Usage
}
