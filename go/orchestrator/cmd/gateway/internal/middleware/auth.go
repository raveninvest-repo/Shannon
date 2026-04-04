package middleware

import (
	"context"
	"net/http"
	"os"
	"strings"

	authpkg "github.com/Kocoro-lab/Shannon/go/orchestrator/internal/auth"
	"github.com/google/uuid"
	"go.uber.org/zap"
)

// AuthMiddleware provides authentication middleware
type AuthValidator interface {
	ValidateAPIKey(ctx context.Context, apiKey string) (*authpkg.UserContext, error)
	ValidateAccessToken(ctx context.Context, token string) (*authpkg.UserContext, error)
}

// AuthMiddleware provides authentication middleware
type AuthMiddleware struct {
	authService AuthValidator
	logger      *zap.Logger
}

// NewAuthMiddleware creates a new authentication middleware
func NewAuthMiddleware(authService AuthValidator, logger *zap.Logger) *AuthMiddleware {
	return &AuthMiddleware{
		authService: authService,
		logger:      logger,
	}
}

// Middleware returns the HTTP middleware function
func (m *AuthMiddleware) Middleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Check if auth should be skipped (DEVELOPMENT ONLY - NEVER USE IN PRODUCTION)
		env := strings.TrimSpace(os.Getenv("ENVIRONMENT"))
		skipAuth := strings.TrimSpace(os.Getenv("GATEWAY_SKIP_AUTH"))

		if skipAuth == "1" {
			// Only allow auth skip in development environment
			if env != "development" && env != "dev" && env != "test" {
				m.logger.Error("SECURITY WARNING: GATEWAY_SKIP_AUTH enabled in non-development environment",
					zap.String("environment", env),
					zap.String("path", r.URL.Path),
				)
				m.sendUnauthorized(w, "Authentication required")
				return
			}

			m.logger.Warn("Authentication bypassed (DEVELOPMENT MODE ONLY)",
				zap.String("environment", env),
				zap.String("path", r.URL.Path),
			)

			// In dev mode, respect x-user-id and x-tenant-id headers if provided
			// This allows testing ownership/tenancy isolation without real auth
			userID := uuid.MustParse("00000000-0000-0000-0000-000000000002")   // default
			tenantID := uuid.MustParse("00000000-0000-0000-0000-000000000001") // default

			if headerUserID := r.Header.Get("x-user-id"); headerUserID != "" {
				if parsed, err := uuid.Parse(headerUserID); err == nil {
					userID = parsed
				}
			}
			if headerTenantID := r.Header.Get("x-tenant-id"); headerTenantID != "" {
				if parsed, err := uuid.Parse(headerTenantID); err == nil {
					tenantID = parsed
				}
			}

			userCtx := &authpkg.UserContext{
				UserID:       userID,
				TenantID:     tenantID,
				Username:     "admin",
				Email:        "admin@localhost",
				Role:         "admin",
				IsAPIKey:     true,
				TokenType:    "api_key",
				AuthBypassed: true,
			}
			ctx := context.WithValue(r.Context(), authpkg.UserContextKey, userCtx)
			next.ServeHTTP(w, r.WithContext(ctx))
			return
		}

		// Extract token from headers (preferred) or query params (SSE endpoints only)
		token := m.extractToken(r)
		if token == "" {
			m.sendUnauthorized(w, "Authentication required")
			return
		}

		// Validate JWT access token first, then fall back to API key
		userCtx, err := m.validateToken(r.Context(), token)
		if err != nil {
			m.logger.Debug("Authentication validation failed",
				zap.Error(err),
				zap.String("token_prefix", m.getTokenPrefix(token)),
			)
			m.sendUnauthorized(w, "Invalid token")
			return
		}

		// Add user context to request context
		// IMPORTANT: Use authpkg.UserContextKey (typed ContextKey), not plain string "user"
		// This matches what grpcmeta.go expects when extracting user context
		ctx := context.WithValue(r.Context(), authpkg.UserContextKey, userCtx)

		// Log successful authentication
		m.logger.Debug("Request authenticated",
			zap.String("user_id", userCtx.UserID.String()),
			zap.String("tenant_id", userCtx.TenantID.String()),
			zap.String("path", r.URL.Path),
		)

		// Continue with authenticated request
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

func (m *AuthMiddleware) validateToken(ctx context.Context, token string) (*authpkg.UserContext, error) {
	if userCtx, err := m.authService.ValidateAccessToken(ctx, token); err == nil {
		return userCtx, nil
	}
	return m.authService.ValidateAPIKey(ctx, token)
}

// extractToken extracts an auth token (JWT access token or API key) from the request.
func (m *AuthMiddleware) extractToken(r *http.Request) string {
	// Check X-API-Key header (preferred)
	if apiKey := r.Header.Get("X-API-Key"); apiKey != "" {
		return normalizePotentialAPIKey(apiKey)
	}

	// Check Authorization header with Bearer token (preferred)
	if auth := r.Header.Get("Authorization"); auth != "" {
		parts := strings.Split(auth, " ")
		if len(parts) == 2 && strings.ToLower(parts[0]) == "bearer" {
			return normalizePotentialAPIKey(parts[1])
		}
	}

	// Check query parameter (SSE/WebSocket streaming endpoints only - for EventSource/WebSocket API compatibility)
	// SECURITY WARNING: Query parameters appear in server logs, browser history, and referrer headers.
	// This is only supported because browser EventSource API doesn't allow custom headers.
	// Production deployments should use headers when possible.
	path := r.URL.Path
	isStreamingEndpoint := r.Method == http.MethodGet && (strings.HasPrefix(path, "/api/v1/stream/") || strings.HasPrefix(path, "/v1/ws/"))

	if isStreamingEndpoint {
		// Try api_key query param first
		if apiKey := r.URL.Query().Get("api_key"); apiKey != "" {
			m.logger.Warn("API key provided via query parameter (SSE endpoint)",
				zap.String("path", path),
				zap.String("method", r.Method),
				zap.String("remote_addr", r.RemoteAddr),
				zap.String("token_prefix", m.getTokenPrefix(apiKey)),
				zap.String("warning", "Query params appear in access logs. Use X-API-Key header when possible."),
			)
			return normalizePotentialAPIKey(apiKey)
		}

		// Try token query param (JWT) for SSE endpoints
		if token := r.URL.Query().Get("token"); token != "" {
			m.logger.Warn("JWT token provided via query parameter (SSE endpoint)",
				zap.String("path", path),
				zap.String("method", r.Method),
				zap.String("remote_addr", r.RemoteAddr),
				zap.String("token_prefix", m.getTokenPrefix(token)),
				zap.String("warning", "Query params appear in access logs. Use Authorization header when possible."),
			)
			return normalizePotentialAPIKey(token)
		}
	} else {
		// Non-streaming endpoint - reject query param auth
		if apiKey := r.URL.Query().Get("api_key"); apiKey != "" {
			m.logger.Warn("API key in query parameter rejected (non-SSE endpoint)",
				zap.String("path", path),
				zap.String("method", r.Method),
				zap.String("token_prefix", m.getTokenPrefix(apiKey)),
				zap.String("hint", "Use X-API-Key header or Authorization: Bearer header instead"),
			)
		}
		if token := r.URL.Query().Get("token"); token != "" {
			m.logger.Warn("JWT token in query parameter rejected (non-SSE endpoint)",
				zap.String("path", path),
				zap.String("method", r.Method),
				zap.String("token_prefix", m.getTokenPrefix(token)),
				zap.String("hint", "Use Authorization: Bearer header instead"),
			)
		}
	}

	return ""
}

func normalizePotentialAPIKey(token string) string {
	token = strings.TrimSpace(token)
	if strings.HasPrefix(token, "sk-shannon-") {
		token = strings.TrimPrefix(token, "sk-shannon-")
		if !strings.HasPrefix(token, "sk_") {
			token = "sk_" + token
		}
	}
	return token
}

// getTokenPrefix returns the first few characters of a token for logging.
func (m *AuthMiddleware) getTokenPrefix(token string) string {
	if len(token) > 8 {
		return token[:8] + "..."
	}
	return "***"
}

// sendUnauthorized sends an unauthorized response
func (m *AuthMiddleware) sendUnauthorized(w http.ResponseWriter, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("WWW-Authenticate", `Bearer realm="Shannon API"`)
	w.WriteHeader(http.StatusUnauthorized)
	w.Write([]byte(`{"error":"` + message + `"}`))
}

// ServeHTTP implements http.Handler interface for convenience
func (m *AuthMiddleware) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// This allows the middleware to be used directly as a handler
	// It will reject all requests since there's no next handler
	m.sendUnauthorized(w, "Direct access not allowed")
}
