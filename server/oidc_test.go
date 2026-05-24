package main

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestLooksLikeJWT(t *testing.T) {
	cases := []struct {
		in   string
		want bool
	}{
		{"eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1MSJ9.sig", true},
		{"sk-1234567890abcdef", false},
		{"ak-12345.67890", false}, // only 2 segments
		{"", false},
		{"a.b", false},
		{"a..c", false}, // empty middle
		{".b.c", false},
		{"a.b.", false},
		{"a.b.c.d", false},
	}
	for _, tc := range cases {
		t.Run(tc.in, func(t *testing.T) {
			if got := looksLikeJWT(tc.in); got != tc.want {
				t.Fatalf("looksLikeJWT(%q) = %v, want %v", tc.in, got, tc.want)
			}
		})
	}
}

// newPKCEPair must produce a verifier of 43+ chars (RFC 7636 §4.1) and a
// challenge equal to base64url(SHA256(verifier)).  We additionally check
// that two consecutive calls produce different verifiers — the function
// must use a fresh CSPRNG read per call.
func TestNewPKCEPair(t *testing.T) {
	v1, c1, err := newPKCEPair()
	if err != nil {
		t.Fatalf("newPKCEPair: %v", err)
	}
	if len(v1) < 43 || len(v1) > 128 {
		t.Fatalf("verifier length %d out of RFC 7636 bounds", len(v1))
	}
	h := sha256.Sum256([]byte(v1))
	want := base64.RawURLEncoding.EncodeToString(h[:])
	if c1 != want {
		t.Fatalf("challenge mismatch: got %s want %s", c1, want)
	}
	v2, _, err := newPKCEPair()
	if err != nil {
		t.Fatalf("newPKCEPair 2: %v", err)
	}
	if v1 == v2 {
		t.Fatalf("two PKCE verifiers collided: %s", v1)
	}
}

func TestOIDCClaimsDisplayName(t *testing.T) {
	cases := []struct {
		c    oidcClaims
		want string
	}{
		{oidcClaims{Name: "Ada Lovelace"}, "Ada Lovelace"},
		{oidcClaims{PreferredUsername: "ada"}, "ada"},
		{oidcClaims{Email: "ada@example.com"}, "ada"},
		{oidcClaims{}, "user"},
		{oidcClaims{Name: "   "}, "user"},
	}
	for _, tc := range cases {
		if got := tc.c.displayName(); got != tc.want {
			t.Fatalf("displayName(%+v) = %q want %q", tc.c, got, tc.want)
		}
	}
}

// upsertOIDCUser must insert on first sight (issuer, sub) and return the
// same uid on second sight without inserting a duplicate row.  This is
// the load-bearing invariant for /auth/oidc/callback being idempotent
// across replays.
func TestUpsertOIDCUserIdempotent(t *testing.T) {
	s := newTestServer(t)
	c := oidcClaims{
		Issuer:  "https://dex.example.com",
		Subject: "user-42",
		Name:    "Test User",
		Email:   "tu@example.com",
	}
	uid1, err := s.upsertOIDCUser(c)
	if err != nil {
		t.Fatalf("upsert 1: %v", err)
	}
	uid2, err := s.upsertOIDCUser(c)
	if err != nil {
		t.Fatalf("upsert 2: %v", err)
	}
	if uid1 != uid2 {
		t.Fatalf("expected same uid on second upsert, got %d vs %d", uid1, uid2)
	}

	// Different issuer → different user, even with same sub.
	c2 := c
	c2.Issuer = "https://other.example.com"
	uid3, err := s.upsertOIDCUser(c2)
	if err != nil {
		t.Fatalf("upsert other-issuer: %v", err)
	}
	if uid3 == uid1 {
		t.Fatalf("same uid across different issuers: %d", uid1)
	}

	var n int
	if err := s.db.QueryRow(
		`SELECT COUNT(*) FROM users WHERE oidc_issuer IS NOT NULL`,
	).Scan(&n); err != nil {
		t.Fatalf("count: %v", err)
	}
	if n != 2 {
		t.Fatalf("expected 2 OIDC users, got %d", n)
	}
}

// newOIDCProvider must consume a discovery document at the configured
// issuer URL.  We stub it with httptest and confirm the construction
// succeeds + sets the redirect URL off cfg.publicURL.
func TestNewOIDCProviderDiscovery(t *testing.T) {
	mux := http.NewServeMux()
	var srv *httptest.Server
	mux.HandleFunc("/.well-known/openid-configuration", func(w http.ResponseWriter, r *http.Request) {
		doc := map[string]any{
			"issuer":                 srv.URL,
			"authorization_endpoint": srv.URL + "/auth",
			"token_endpoint":         srv.URL + "/token",
			"jwks_uri":               srv.URL + "/jwks",
			"id_token_signing_alg_values_supported": []string{"RS256"},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(doc)
	})
	srv = httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	cfg := &config{
		oidcIssuer:       srv.URL,
		oidcClientID:     "test-client",
		oidcClientSecret: "test-secret",
		publicURL:        "https://app.example.com",
	}
	p, err := newOIDCProvider(context.Background(), cfg)
	if err != nil {
		t.Fatalf("newOIDCProvider: %v", err)
	}
	if p.audience != "test-client" {
		t.Fatalf("audience default to clientID, got %q", p.audience)
	}
	if p.oauth2.RedirectURL != "https://app.example.com/auth/oidc/callback" {
		t.Fatalf("redirect URL mismatch: %q", p.oauth2.RedirectURL)
	}
	if !strings.HasPrefix(p.oauth2.Endpoint.AuthURL, srv.URL) {
		t.Fatalf("auth URL not derived from discovery: %q", p.oauth2.Endpoint.AuthURL)
	}
}

func TestNewOIDCProviderRequiresClientCreds(t *testing.T) {
	cfg := &config{
		oidcIssuer: "https://dex.example.com",
		publicURL:  "https://app.example.com",
	}
	if _, err := newOIDCProvider(context.Background(), cfg); err == nil {
		t.Fatalf("expected error when client creds missing")
	}
}

// /auth/oidc must return 501 when OIDC is not configured — same shape as
// /auth/github when DIST_GITHUB_CLIENT is unset.  This guards against
// accidentally exposing the route as 200 + empty redirect.
func TestOIDCStartUnconfigured(t *testing.T) {
	s := newTestServer(t)
	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/auth/oidc", nil)
	s.handleOIDCStart(w, r)
	if w.Code != 501 {
		t.Fatalf("want 501 when oidc nil, got %d", w.Code)
	}
}

// /api/auth/status surfaces oidc_configured + oidc_label so the auth
// page can render a third button when the operator has wired dex/hydra.
func TestAuthStatusExposesOIDC(t *testing.T) {
	s := newTestServer(t)
	s.cfg.oidcLabel = "Acme SSO"
	// Without provider configured: oidc_configured=false.
	w := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/api/auth/status", nil)
	s.handleAuthStatus(w, r)
	if w.Code != 200 {
		t.Fatalf("status: %d", w.Code)
	}
	var got map[string]any
	if err := json.NewDecoder(w.Body).Decode(&got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got["oidc_configured"] != false {
		t.Fatalf("oidc_configured should be false, got %v", got["oidc_configured"])
	}
	if got["oidc_label"] != "Acme SSO" {
		t.Fatalf("oidc_label = %v", got["oidc_label"])
	}
	// Quick guard that the JSON encoder didn't drop a key.
	for _, k := range []string{"github_configured", "google_configured", "oidc_configured", "oidc_label", "dev_mode", "user_id"} {
		if _, ok := got[k]; !ok {
			t.Fatalf("status missing key %q: %v", k, got)
		}
	}
	_ = fmt.Sprintf
}
