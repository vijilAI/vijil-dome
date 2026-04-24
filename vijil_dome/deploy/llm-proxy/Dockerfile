# Build stage: compile static binary
FROM golang:1.22-alpine AS builder

RUN apk add --no-cache ca-certificates

WORKDIR /build
COPY . .
RUN go mod tidy && \
    CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-s -w" -o /llm-proxy ./cmd/proxy

# Runtime stage: distroless for minimal attack surface
# No shell, no package manager, no unnecessary tools.
FROM gcr.io/distroless/static-debian12:nonroot

COPY --from=builder /llm-proxy /llm-proxy

# SPIRE agent socket is mounted at runtime
VOLUME ["/run/spire/sockets"]

# Config is mounted at runtime
VOLUME ["/etc/llm-proxy"]

EXPOSE 8443

ENTRYPOINT ["/llm-proxy"]
CMD ["-config", "/etc/llm-proxy/config.json"]
