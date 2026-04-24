// llm-proxy is a thin, SPIFFE-authenticated proxy for LLM API calls.
//
// It accepts mTLS connections from AI agent workloads (authenticated via
// X.509-SVIDs), fetches API keys from Vault using its own JWT-SVID,
// and forwards requests to upstream LLM providers. API keys are held
// only in memory for the duration of a single request.
//
// Usage:
//
//	llm-proxy -config /etc/llm-proxy/config.json
package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/spiffe/go-spiffe/v2/spiffeid"
	"github.com/spiffe/go-spiffe/v2/workloadapi"

	"github.com/vijil-ai/llm-proxy/internal/config"
	"github.com/vijil-ai/llm-proxy/internal/proxy"
	"github.com/vijil-ai/llm-proxy/internal/telemetry"
	"github.com/vijil-ai/llm-proxy/internal/vault"
)

func main() {
	configPath := flag.String("config", "/etc/llm-proxy/config.json", "path to configuration file")
	flag.Parse()

	// Structured JSON logger for OTel Collector ingestion
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))

	if err := run(*configPath, logger); err != nil {
		logger.Error("fatal", "error", err)
		os.Exit(1)
	}
}

func run(configPath string, logger *slog.Logger) error {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// ── Load configuration ──
	cfgLoader, err := config.NewLoader(configPath, logger)
	if err != nil {
		return fmt.Errorf("load config: %w", err)
	}
	go cfgLoader.WatchForChanges(5 * time.Second)
	defer cfgLoader.Stop()

	cfg := cfgLoader.Get()
	logger.Info("configuration loaded",
		"listen_addr", cfg.ListenAddr,
		"vault_addr", cfg.VaultAddr,
		"routes", len(cfg.Routes),
	)

	// ── Connect to SPIFFE Workload API ──
	logger.Info("connecting to SPIFFE Workload API", "socket", cfg.SpiffeSocket)

	x509Source, err := workloadapi.NewX509Source(
		ctx,
		workloadapi.WithClientOptions(workloadapi.WithAddr(cfg.SpiffeSocket)),
	)
	if err != nil {
		return fmt.Errorf("create X509 source: %w", err)
	}
	defer x509Source.Close()

	// Log the proxy's own SPIFFE ID
	svid, err := x509Source.GetX509SVID()
	if err != nil {
		return fmt.Errorf("get proxy SVID: %w", err)
	}
	logger.Info("proxy SVID acquired", "spiffe_id", svid.ID.String())

	// JWT source for Vault authentication
	jwtSource, err := workloadapi.NewJWTSource(
		ctx,
		workloadapi.WithClientOptions(workloadapi.WithAddr(cfg.SpiffeSocket)),
	)
	if err != nil {
		return fmt.Errorf("create JWT source: %w", err)
	}
	defer jwtSource.Close()

	// Parse trust domain from the proxy's own SPIFFE ID
	trustDomain, err := spiffeid.TrustDomainFromString(svid.ID.TrustDomain().String())
	if err != nil {
		return fmt.Errorf("parse trust domain: %w", err)
	}

	// ── Initialize Vault client ──
	vaultClient := vault.NewClient(
		cfg.VaultAddr,
		cfg.VaultAuthMount,
		cfg.VaultRole,
		logger.With("component", "vault"),
	)

	// ── Initialize telemetry ──
	telemetryEmitter := telemetry.NewEmitter(
		logger.With("component", "telemetry"),
	)

	// ── Build handler ──
	handler := proxy.NewHandler(proxy.HandlerParams{
		ConfigLoader:  cfgLoader,
		VaultClient:   vaultClient,
		Telemetry:     telemetryEmitter,
		X509Source:    x509Source,
		JWTSource:     jwtSource,
		Logger:        logger.With("component", "proxy"),
		TrustDomain:   trustDomain,
		VaultAudience: cfg.VaultAudience,
	})

	// ── Set up HTTP server with mTLS ──
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", proxy.HealthHandler)
	mux.Handle("/v1/proxy/", handler)

	server := &http.Server{
		Addr:      cfg.ListenAddr,
		Handler:   mux,
		TLSConfig: handler.TLSConfig(),
		// Timeouts to prevent resource exhaustion
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 180 * time.Second, // Long for LLM streaming responses
		IdleTimeout:  60 * time.Second,
	}

	// ── Graceful shutdown ──
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	errCh := make(chan error, 1)
	go func() {
		logger.Info("starting mTLS server", "addr", cfg.ListenAddr)
		// TLS certs are provided by the SPIFFE X509Source, so we pass empty
		// cert/key file paths. The TLSConfig handles certificate retrieval.
		if err := server.ListenAndServeTLS("", ""); err != nil && err != http.ErrServerClosed {
			errCh <- err
		}
	}()

	select {
	case sig := <-sigCh:
		logger.Info("received shutdown signal", "signal", sig)
	case err := <-errCh:
		return fmt.Errorf("server error: %w", err)
	}

	// Graceful shutdown with timeout
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer shutdownCancel()

	if err := server.Shutdown(shutdownCtx); err != nil {
		return fmt.Errorf("graceful shutdown: %w", err)
	}

	logger.Info("server stopped")
	return nil
}
