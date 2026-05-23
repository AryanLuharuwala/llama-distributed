package main

// CORS / redirect / TLS-header test surface.
//
// Covers headers and redirects that browsers and TLS-terminating proxies
// rely on for cross-origin requests and HTTPS enforcement.  These are
// easy to regress (one stray `return` skips a header, one wrong path
// pattern leaks Allow-Origin to cookie surfaces) and the failure is
// silent in dev because curl ignores them.

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// ── CORS ────────────────────────────────────────────────────────────────────

func TestCORS_V1Preflight_ReturnsAllowHeaders(t *testing.T) {
	s := newTestServer(t)
	rr := httptest.NewRecorder()
	req := httptest.NewRequest("OPTIONS", "/v1/chat/completions", nil)
	req.Header.Set("Origin", "https://example.com")
	req.Header.Set("Access-Control-Request-Method", "POST")
	s.router().ServeHTTP(rr, req)

	if rr.Code != http.StatusNoContent {
		t.Errorf("preflight: got %d, want 204", rr.Code)
	}
	if got := rr.Header().Get("Access-Control-Allow-Origin"); got != "*" {
		t.Errorf("Allow-Origin: %q want *", got)
	}
	if !strings.Contains(rr.Header().Get("Access-Control-Allow-Methods"), "POST") {
		t.Errorf("Allow-Methods missing POST: %q", rr.Header().Get("Access-Control-Allow-Methods"))
	}
	if !strings.Contains(rr.Header().Get("Access-Control-Allow-Headers"), "Authorization") {
		t.Errorf("Allow-Headers missing Authorization: %q", rr.Header().Get("Access-Control-Allow-Headers"))
	}
	// We intentionally do NOT send Allow-Credentials — pairing `*` with
	// credentials would be invalid and would let any origin make
	// cookie-authenticated calls.  Lock it in.
	if rr.Header().Get("Access-Control-Allow-Credentials") != "" {
		t.Errorf("CORS Allow-Credentials must stay empty (Bearer-only surface): %q",
			rr.Header().Get("Access-Control-Allow-Credentials"))
	}
}

func TestCORS_V1PoolSlugPreflight(t *testing.T) {
	s := newTestServer(t)
	rr := httptest.NewRecorder()
	req := httptest.NewRequest("OPTIONS", "/v1/some-pool/chat/completions", nil)
	req.Header.Set("Origin", "https://example.com")
	req.Header.Set("Access-Control-Request-Method", "POST")
	s.router().ServeHTTP(rr, req)

	if rr.Code != http.StatusNoContent {
		t.Errorf("slug preflight: got %d, want 204", rr.Code)
	}
	if rr.Header().Get("Access-Control-Allow-Origin") != "*" {
		t.Errorf("slug preflight missing Allow-Origin: %v", rr.Header())
	}
}

// CORS must NOT leak onto cookie-authenticated surfaces (everything
// outside /v1/*).  The middleware is path-gated and this test pins
// that gate in place.
func TestCORS_NonV1_NoAllowOrigin(t *testing.T) {
	s := newTestServer(t)
	for _, path := range []string{"/api/pools", "/auth/github", "/healthz", "/"} {
		rr := httptest.NewRecorder()
		req := httptest.NewRequest("OPTIONS", path, nil)
		req.Header.Set("Origin", "https://evil.example")
		s.router().ServeHTTP(rr, req)

		if rr.Header().Get("Access-Control-Allow-Origin") != "" {
			t.Errorf("%s: Allow-Origin leaked: %q", path, rr.Header().Get("Access-Control-Allow-Origin"))
		}
	}
}

// ── Security headers ────────────────────────────────────────────────────────

func TestSecurityHeaders_AlwaysSet(t *testing.T) {
	s := newTestServer(t)
	rr := httptest.NewRecorder()
	s.router().ServeHTTP(rr, httptest.NewRequest("GET", "/healthz", nil))

	if rr.Header().Get("X-Content-Type-Options") != "nosniff" {
		t.Errorf("nosniff missing: %q", rr.Header().Get("X-Content-Type-Options"))
	}
	if rr.Header().Get("X-Frame-Options") != "DENY" {
		t.Errorf("X-Frame-Options: %q want DENY", rr.Header().Get("X-Frame-Options"))
	}
	if rr.Header().Get("Referrer-Policy") == "" {
		t.Errorf("Referrer-Policy not set")
	}
}

// HSTS is only meaningful under HTTPS — setting it on plain HTTP makes
// curl tests confusing and produces no real security benefit.
// `r.TLS != nil` is the in-process TLS case; `X-Forwarded-Proto: https`
// is the reverse-proxy case.
func TestHSTS_OnlySetUnderTLS(t *testing.T) {
	s := newTestServer(t)

	// Plain HTTP — no HSTS.
	rr := httptest.NewRecorder()
	s.router().ServeHTTP(rr, httptest.NewRequest("GET", "/healthz", nil))
	if rr.Header().Get("Strict-Transport-Security") != "" {
		t.Errorf("HSTS leaked under plain HTTP: %q", rr.Header().Get("Strict-Transport-Security"))
	}

	// Proxy reports the client used TLS.
	rr = httptest.NewRecorder()
	req := httptest.NewRequest("GET", "/healthz", nil)
	req.Header.Set("X-Forwarded-Proto", "https")
	s.router().ServeHTTP(rr, req)
	if hsts := rr.Header().Get("Strict-Transport-Security"); hsts == "" || !strings.Contains(hsts, "max-age=") {
		t.Errorf("HSTS missing under X-Forwarded-Proto=https: %q", hsts)
	}
}

// ── OAuth redirect / CSRF ───────────────────────────────────────────────────

func TestOAuth_NotConfigured_Returns501(t *testing.T) {
	s := newTestServer(t)
	// newTestServer doesn't set githubClient — start handler should 501.
	rr := httptest.NewRecorder()
	s.router().ServeHTTP(rr, httptest.NewRequest("GET", "/auth/github", nil))
	if rr.Code != http.StatusNotImplemented {
		t.Errorf("start: got %d want 501", rr.Code)
	}

	rr = httptest.NewRecorder()
	s.router().ServeHTTP(rr, httptest.NewRequest("GET", "/auth/github/callback?code=x&state=y", nil))
	if rr.Code != http.StatusNotImplemented {
		t.Errorf("callback: got %d want 501", rr.Code)
	}
}

func TestOAuth_StartSetsStateCookieAndRedirects(t *testing.T) {
	s := newTestServer(t)
	s.cfg.githubClient = "test-client-id"
	s.cfg.githubSecret = "test-secret"
	s.cfg.publicURL = "https://pool.example.com"

	rr := httptest.NewRecorder()
	s.router().ServeHTTP(rr, httptest.NewRequest("GET", "/auth/github", nil))

	if rr.Code != http.StatusFound {
		t.Fatalf("got %d want 302", rr.Code)
	}
	loc := rr.Header().Get("Location")
	if !strings.HasPrefix(loc, "https://github.com/login/oauth/authorize") {
		t.Errorf("redirect location wrong: %q", loc)
	}
	if !strings.Contains(loc, "redirect_uri=https%3A%2F%2Fpool.example.com%2Fauth%2Fgithub%2Fcallback") {
		t.Errorf("redirect_uri missing or malformed: %q", loc)
	}
	if !strings.Contains(loc, "state=") {
		t.Errorf("state param missing: %q", loc)
	}

	// State cookie must be set, HttpOnly, SameSite=Lax, path-scoped to /auth.
	var got *http.Cookie
	for _, c := range rr.Result().Cookies() {
		if c.Name == "oauth_state" {
			got = c
		}
	}
	if got == nil {
		t.Fatalf("oauth_state cookie not set")
	}
	if !got.HttpOnly {
		t.Errorf("oauth_state must be HttpOnly")
	}
	if got.SameSite != http.SameSiteLaxMode {
		t.Errorf("oauth_state SameSite=%v want Lax", got.SameSite)
	}
	if got.Path != "/auth" {
		t.Errorf("oauth_state Path=%q want /auth", got.Path)
	}
}

func TestOAuth_Callback_RejectsMissingState(t *testing.T) {
	s := newTestServer(t)
	s.cfg.githubClient = "test-client"
	s.cfg.githubSecret = "test-secret"

	rr := httptest.NewRecorder()
	// No cookie, no matching state — must be rejected as CSRF.
	s.router().ServeHTTP(rr, httptest.NewRequest("GET", "/auth/github/callback?code=abc&state=xyz", nil))
	if rr.Code != http.StatusBadRequest {
		t.Errorf("missing-cookie callback: got %d want 400", rr.Code)
	}
}

func TestOAuth_Callback_RejectsStateMismatch(t *testing.T) {
	s := newTestServer(t)
	s.cfg.githubClient = "test-client"
	s.cfg.githubSecret = "test-secret"

	rr := httptest.NewRecorder()
	req := httptest.NewRequest("GET", "/auth/github/callback?code=abc&state=server-said", nil)
	// Attacker forges a different state value in the cookie.
	req.AddCookie(&http.Cookie{Name: "oauth_state", Value: "attacker-said"})
	s.router().ServeHTTP(rr, req)
	if rr.Code != http.StatusBadRequest {
		t.Errorf("state mismatch: got %d want 400", rr.Code)
	}
}

func TestOAuth_Callback_RejectsMissingCode(t *testing.T) {
	s := newTestServer(t)
	s.cfg.githubClient = "test-client"
	s.cfg.githubSecret = "test-secret"

	rr := httptest.NewRecorder()
	req := httptest.NewRequest("GET", "/auth/github/callback?state=matching", nil)
	req.AddCookie(&http.Cookie{Name: "oauth_state", Value: "matching"})
	s.router().ServeHTTP(rr, req)
	if rr.Code != http.StatusBadRequest {
		t.Errorf("missing code: got %d want 400", rr.Code)
	}
}

// ── Session cookie shape (for completeness — these are also TLS-sensitive) ──

func TestSessionCookie_AfterDevLogin_HasSecureFlags(t *testing.T) {
	s := newTestServer(t)
	s.cfg.devMode = true

	rr := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/auth/dev", strings.NewReader(`{"display_name":"alice"}`))
	req.Header.Set("Content-Type", "application/json")
	s.router().ServeHTTP(rr, req)

	if rr.Code/100 != 2 {
		t.Fatalf("dev login: %d %s", rr.Code, rr.Body.String())
	}
	var sc *http.Cookie
	for _, c := range rr.Result().Cookies() {
		if c.Name == sessionCookieName {
			sc = c
		}
	}
	if sc == nil {
		t.Fatalf("session cookie not set")
	}
	if !sc.HttpOnly {
		t.Errorf("session cookie must be HttpOnly")
	}
	if sc.SameSite != http.SameSiteLaxMode && sc.SameSite != http.SameSiteStrictMode {
		t.Errorf("session cookie SameSite=%v; want Lax or Strict", sc.SameSite)
	}
}
