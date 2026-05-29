package main

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"database/sql"
	"encoding/base64"
	"errors"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/coreos/go-oidc/v3/oidc"
	"golang.org/x/oauth2"
)

// oidcProvider is the live state for the dex/hydra (or any OIDC OP)
// integration.  Built once at boot via newOIDCProvider, then read-only
// for the rest of the process lifetime; the embedded *oidc.Provider
// keeps its own JWKS fresh via the go-oidc library.
//
// Disabled when cfg.oidcIssuer is empty — cfg.oidc is nil and the
// /auth/oidc routes return 501.
type oidcProvider struct {
	issuer   string
	audience string // expected `aud` claim for JWT bearers
	clientID string
	clientSecret string

	prov     *oidc.Provider
	verifier *oidc.IDTokenVerifier
	oauth2   *oauth2.Config
}

// newOIDCProvider runs OIDC discovery against the configured issuer
// and returns a live verifier + oauth2 config.  The redirect URL is
// derived from cfg.publicURL so a single env var (DIST_PUBLIC_URL)
// drives both browser and OP-registered values.
func newOIDCProvider(ctx context.Context, cfg *config) (*oidcProvider, error) {
	if cfg.oidcClientID == "" || cfg.oidcClientSecret == "" {
		return nil, errors.New("DIST_OIDC_CLIENT_ID and DIST_OIDC_CLIENT_SECRET required when DIST_OIDC_ISSUER is set")
	}
	dctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	prov, err := oidc.NewProvider(dctx, cfg.oidcIssuer)
	if err != nil {
		return nil, fmt.Errorf("discovery: %w", err)
	}
	aud := cfg.oidcBearerAudience
	if aud == "" {
		aud = cfg.oidcClientID
	}
	verifier := prov.Verifier(&oidc.Config{ClientID: aud})
	oa := &oauth2.Config{
		ClientID:     cfg.oidcClientID,
		ClientSecret: cfg.oidcClientSecret,
		RedirectURL:  cfg.publicURL + "/auth/oidc/callback",
		Endpoint:     prov.Endpoint(),
		Scopes:       []string{oidc.ScopeOpenID, "profile", "email"},
	}
	return &oidcProvider{
		issuer:       cfg.oidcIssuer,
		audience:     aud,
		clientID:     cfg.oidcClientID,
		clientSecret: cfg.oidcClientSecret,
		prov:         prov,
		verifier:     verifier,
		oauth2:       oa,
	}, nil
}

// verifyBearer authenticates a JWT bearer token.  Returns the parsed
// claims subset the server cares about, or an error suitable for a
// 401 response.  Verification chain matches what the browser callback
// does: signature via JWKS, issuer + audience + expiry checks.
func (p *oidcProvider) verifyBearer(ctx context.Context, raw string) (oidcClaims, error) {
	if p == nil {
		return oidcClaims{}, errors.New("oidc not configured")
	}
	tok, err := p.verifier.Verify(ctx, raw)
	if err != nil {
		return oidcClaims{}, err
	}
	var c oidcClaims
	if err := tok.Claims(&c); err != nil {
		return oidcClaims{}, fmt.Errorf("claims: %w", err)
	}
	c.Issuer = tok.Issuer
	c.Subject = tok.Subject
	return c, nil
}

// oidcClaims is the subset of the ID token / JWT we project into the
// users table.  Anything not in here is dropped — the server's role
// model is built on (issuer, subject), not on the OP's free-form
// claim set.
type oidcClaims struct {
	Issuer            string `json:"iss"`
	Subject           string `json:"sub"`
	Email             string `json:"email"`
	Name              string `json:"name"`
	PreferredUsername string `json:"preferred_username"`
}

// displayName picks a non-empty human-readable label for the user.
// Preference order: explicit "name" claim → preferred_username →
// email local part → "user".
func (c oidcClaims) displayName() string {
	if s := strings.TrimSpace(c.Name); s != "" {
		return s
	}
	if s := strings.TrimSpace(c.PreferredUsername); s != "" {
		return s
	}
	if i := strings.IndexByte(c.Email, '@'); i > 0 {
		return c.Email[:i]
	}
	return "user"
}

// ─── handlers ───────────────────────────────────────────────────────────

// /auth/oidc — kick off the code flow.  Stores PKCE verifier + state +
// optional `next` URL in short-lived cookies, then redirects to the OP.
//
// PKCE is unconditional even though we're a confidential client and
// the OP would accept us without it — it's free defense in depth
// against an attacker who steals an authorization code mid-flight.
func (s *server) handleOIDCStart(w http.ResponseWriter, r *http.Request) {
	if s.cfg.oidc == nil {
		http.Error(w, "OIDC not configured (set DIST_OIDC_ISSUER + CLIENT_ID/SECRET)", 501)
		return
	}
	state := s.mintOAuthState(s.clientIP(r))
	verifier, challenge, err := newPKCEPair()
	if err != nil {
		http.Error(w, "pkce: "+err.Error(), 500)
		return
	}

	secure := s.secureCookies()
	http.SetCookie(w, &http.Cookie{
		Name: "oidc_state", Value: state, Path: "/auth", HttpOnly: true,
		Secure: secure, MaxAge: 600, SameSite: http.SameSiteLaxMode,
	})
	http.SetCookie(w, &http.Cookie{
		Name: "oidc_verifier", Value: verifier, Path: "/auth", HttpOnly: true,
		Secure: secure, MaxAge: 600, SameSite: http.SameSiteLaxMode,
	})
	if next := r.URL.Query().Get("next"); isSafeNext(next) {
		http.SetCookie(w, &http.Cookie{
			Name: "oauth_next", Value: next, Path: "/auth", HttpOnly: true,
			Secure: secure, MaxAge: 600, SameSite: http.SameSiteLaxMode,
		})
	}

	u := s.cfg.oidc.oauth2.AuthCodeURL(state,
		oauth2.SetAuthURLParam("code_challenge", challenge),
		oauth2.SetAuthURLParam("code_challenge_method", "S256"),
	)
	http.Redirect(w, r, u, http.StatusFound)
}

// /auth/oidc/callback — receives the code, exchanges it, verifies the
// ID token, upserts the user, mints a session cookie.
//
// Same shape as handleGithubCallback so a future refactor can collapse
// them — but kept separate for now because OIDC has more verification
// surface (issuer, audience, nonce, JWKS) and tangling that with the
// minimal GitHub flow would obscure both.
func (s *server) handleOIDCCallback(w http.ResponseWriter, r *http.Request) {
	if s.cfg.oidc == nil {
		http.Error(w, "OIDC not configured", 501)
		return
	}
	state := r.URL.Query().Get("state")
	sc, err := r.Cookie("oidc_state")
	if err != nil || sc.Value == "" || sc.Value != state {
		http.Error(w, "bad oidc state", 400)
		return
	}
	if !s.verifyOAuthState(state, s.clientIP(r)) {
		http.Error(w, "oidc state expired or bound to a different client", 400)
		return
	}
	vc, err := r.Cookie("oidc_verifier")
	if err != nil || vc.Value == "" {
		http.Error(w, "missing pkce verifier", 400)
		return
	}
	code := r.URL.Query().Get("code")
	if code == "" {
		http.Error(w, "missing code", 400)
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()
	tok, err := s.cfg.oidc.oauth2.Exchange(ctx, code,
		oauth2.SetAuthURLParam("code_verifier", vc.Value))
	if err != nil {
		http.Error(w, "exchange: "+err.Error(), 502)
		return
	}
	rawID, ok := tok.Extra("id_token").(string)
	if !ok || rawID == "" {
		http.Error(w, "id_token missing from OP response", 502)
		return
	}
	idTok, err := s.cfg.oidc.verifier.Verify(ctx, rawID)
	if err != nil {
		http.Error(w, "verify id_token: "+err.Error(), 502)
		return
	}
	var claims oidcClaims
	if err := idTok.Claims(&claims); err != nil {
		http.Error(w, "claims: "+err.Error(), 502)
		return
	}
	claims.Issuer = idTok.Issuer
	claims.Subject = idTok.Subject
	if claims.Subject == "" {
		http.Error(w, "id_token missing sub claim", 502)
		return
	}

	uid, err := s.upsertOIDCUser(claims)
	if err != nil {
		http.Error(w, "upsert: "+err.Error(), 500)
		return
	}

	sid, exp, err := s.createSession(uid)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	s.setSessionCookie(w, sid, exp)

	dest := "/home"
	if c, err := r.Cookie("oauth_next"); err == nil && isSafeNext(c.Value) {
		dest = c.Value
		http.SetCookie(w, &http.Cookie{
			Name: "oauth_next", Value: "", Path: "/auth", HttpOnly: true,
			Secure: s.secureCookies(), MaxAge: -1,
		})
	}
	http.Redirect(w, r, dest, http.StatusFound)
}

// upsertOIDCUser looks up a user by (issuer, subject) and creates one
// if absent.  Display-name updates from claim drift do NOT overwrite
// a user-edited display_name — the OP is authoritative for identity,
// but the user owns their own name once they're in our system.
func (s *server) upsertOIDCUser(c oidcClaims) (int64, error) {
	var uid int64
	err := s.dbQueryRow(
		`SELECT id FROM users WHERE oidc_issuer = ? AND oidc_subject = ?`,
		c.Issuer, c.Subject,
	).Scan(&uid)
	if err == nil {
		return uid, nil
	}
	if !errors.Is(err, sql.ErrNoRows) {
		return 0, err
	}
	display := c.displayName()
	if clean, err := sanitizeDisplayName(display); err == nil {
		display = clean
	} else {
		display = "user"
	}
	res, err := s.dbExec(
		`INSERT INTO users (oidc_issuer, oidc_subject, display_name, created_at)
		 VALUES (?, ?, ?, ?)`,
		c.Issuer, c.Subject, display, nowUnix(),
	)
	if err != nil {
		return 0, err
	}
	uid, err = res.LastInsertId()
	return uid, err
}

// ─── PKCE helpers ───────────────────────────────────────────────────────

// newPKCEPair returns (verifier, challenge) for an OIDC code flow.
// RFC 7636 §4.1: verifier is 43-128 chars of [A-Z a-z 0-9 - . _ ~];
// we use 32 bytes of CSPRNG entropy base64url-encoded (43 chars).
// Challenge is SHA-256(verifier) base64url-encoded.
func newPKCEPair() (verifier, challenge string, err error) {
	var buf [32]byte
	if _, err := rand.Read(buf[:]); err != nil {
		return "", "", err
	}
	verifier = base64.RawURLEncoding.EncodeToString(buf[:])
	h := sha256.Sum256([]byte(verifier))
	challenge = base64.RawURLEncoding.EncodeToString(h[:])
	return verifier, challenge, nil
}

// userFromOIDCBearer is the JWT-bearer arm of userFromRequest.  Looks
// for `Authorization: Bearer eyJ...` (JWT shape — 3 dot-separated
// base64url segments), verifies it via the configured OIDC provider,
// and resolves to a user row, creating one on first sight if the
// `sub` is new.  This is what lets a corporate CI runner with a
// short-lived JWT call /api endpoints without holding a session
// cookie.
func (s *server) userFromOIDCBearer(r *http.Request) (*user, bool) {
	if s.cfg.oidc == nil {
		return nil, false
	}
	raw := bearerFromRequest(r)
	if raw == "" || !looksLikeJWT(raw) {
		return nil, false
	}
	claims, err := s.cfg.oidc.verifyBearer(r.Context(), raw)
	if err != nil {
		return nil, false
	}
	uid, err := s.upsertOIDCUser(claims)
	if err != nil {
		log.Printf("oidc bearer: upsert failed for %s/%s: %v", claims.Issuer, claims.Subject, err)
		return nil, false
	}
	u := &user{ID: uid, DisplayName: claims.displayName()}
	return u, true
}

// looksLikeJWT is a cheap shape check: a JWT is exactly three
// segments separated by dots, each segment a non-empty base64url
// string.  We use it as a fast-path so an `sk-...` api_key bearer
// doesn't waste a JWKS verification on every request.
func looksLikeJWT(s string) bool {
	parts := strings.Split(s, ".")
	if len(parts) != 3 {
		return false
	}
	for _, p := range parts {
		if p == "" {
			return false
		}
	}
	return true
}

