package main

import (
	"crypto/hmac"
	"crypto/sha256"
	_ "embed"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

// oauthStateMaxAge bounds how long a freshly-minted state token is
// honored on callback.  Long enough for a slow GitHub round-trip on a
// flaky network; short enough to make a stolen state token useless.
const oauthStateMaxAge = int64(600) // 10 minutes

// mintOAuthState returns a state string that binds the calling client
// IP and a freshly-stamped timestamp to a server secret.  The format is
// "<unix-ts>.<base64-hmac>" — both halves are echoed back by the OAuth
// provider, so verify() can recompute the HMAC for the same IP and
// reject mismatches.  This is the companion to the Secure cookie
// hardening done in A2: even if the oauth_state cookie escapes, an
// attacker on a different IP can't drive the callback through to a
// stage where the session is minted.
func (s *server) mintOAuthState(clientIP string) string {
	ts := nowUnix()
	mac := s.computeOAuthStateMAC(ts, clientIP)
	return fmt.Sprintf("%d.%s", ts, base64.RawURLEncoding.EncodeToString(mac))
}

// verifyOAuthState parses the timestamp + HMAC, checks both against the
// server secret and the calling client IP, and confirms the timestamp
// is within oauthStateMaxAge.  Returns true only if everything lines
// up.  Constant-time HMAC compare so attackers can't time-attack the
// secret material.
func (s *server) verifyOAuthState(state, clientIP string) bool {
	dot := strings.IndexByte(state, '.')
	if dot <= 0 || dot == len(state)-1 {
		return false
	}
	ts, err := strconv.ParseInt(state[:dot], 10, 64)
	if err != nil {
		return false
	}
	now := nowUnix()
	if ts > now+5 || now-ts > oauthStateMaxAge {
		return false
	}
	got, err := base64.RawURLEncoding.DecodeString(state[dot+1:])
	if err != nil {
		return false
	}
	want := s.computeOAuthStateMAC(ts, clientIP)
	return hmac.Equal(got, want)
}

func (s *server) computeOAuthStateMAC(ts int64, clientIP string) []byte {
	h := hmac.New(sha256.New, []byte(s.cfg.sessionSecret))
	fmt.Fprintf(h, "oauth-state:%d:%s", ts, clientIP)
	sum := h.Sum(nil)
	// Truncate to 16 bytes — plenty against forgery given the timestamp
	// and IP material are already included.
	return sum[:16]
}

//go:embed assets/auth.html
var authPageHTML []byte

// handleAuthPage serves the unified sign-in / signup landing page.  The
// page works for both first-time and returning users — there is no
// separate signup flow, the OAuth callback upserts on github_id.
// isSafeNext returns true iff `v` is a site-relative path safe to use as a
// post-login redirect target. We reject:
//   - empty / non-rooted ("/foo" required)
//   - protocol-relative ("//evil.com")
//   - backslash anywhere ("/\evil.com" — some browsers normalize \ to /,
//     producing //evil.com cross-origin)
//   - embedded NUL or control chars
func isSafeNext(v string) bool {
	if v == "" || !strings.HasPrefix(v, "/") {
		return false
	}
	if strings.HasPrefix(v, "//") {
		return false
	}
	if strings.ContainsAny(v, "\\\x00") {
		return false
	}
	return true
}

func (s *server) handleAuthPage(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	_, _ = w.Write(authPageHTML)
}

// handleAuthStatus tells the /auth page whether GitHub OAuth is wired,
// whether dev-login is permitted, and whether the caller is already
// signed in (so we can skip the form and redirect).
func (s *server) handleAuthStatus(w http.ResponseWriter, r *http.Request) {
	uid := int64(0)
	if u, ok := s.userFromRequest(r); ok {
		uid = u.ID
	}
	writeJSON(w, 200, map[string]any{
		"github_configured": s.cfg.githubClient != "",
		"google_configured": s.cfg.googleClient != "",
		"oidc_configured":   s.cfg.oidc != nil,
		"oidc_label":        s.cfg.oidcLabel,
		"dev_mode":          s.cfg.devMode,
		"user_id":           uid,
	})
}

// GET /api/meta — small unauthenticated endpoint that the dashboard footer
// hits to show build/host/env.  Returns the build SHA (DIST_BUILD_SHA env)
// or "dev" if unset, the configured public host, and "local"/"cloud" derived
// from the public URL scheme.
func (s *server) handleMeta(w http.ResponseWriter, r *http.Request) {
	build := envOr("DIST_BUILD_SHA", "dev")
	host := s.cfg.publicURL
	if idx := strings.Index(host, "://"); idx != -1 {
		host = host[idx+3:]
	}
	env := "cloud"
	if strings.HasPrefix(s.cfg.publicURL, "http://localhost") ||
		strings.HasPrefix(s.cfg.publicURL, "http://127.") {
		env = "local"
	}
	writeJSON(w, 200, map[string]any{
		"build": build,
		"host":  host,
		"env":   env,
	})
}

// Minimal GitHub OAuth.
//
//   /auth/github            → redirect to github.com/login/oauth/authorize
//   /auth/github/callback   → exchange code, fetch /user, upsert + set cookie
//
// Requires DIST_GITHUB_CLIENT + DIST_GITHUB_SECRET env vars.  If unset, these
// endpoints return 501 so the dev-login endpoint can be used instead.

const githubAuthorizeURL = "https://github.com/login/oauth/authorize"
const githubTokenURL = "https://github.com/login/oauth/access_token"
const githubUserURL = "https://api.github.com/user"

func (s *server) handleGithubStart(w http.ResponseWriter, r *http.Request) {
	if s.cfg.githubClient == "" {
		http.Error(w, "GitHub OAuth not configured (set DIST_GITHUB_CLIENT/SECRET)", 501)
		return
	}
	state := s.mintOAuthState(s.clientIP(r))
	secure := s.secureCookies()
	http.SetCookie(w, &http.Cookie{
		Name:     "oauth_state",
		Value:    state,
		Path:     "/auth",
		HttpOnly: true,
		Secure:   secure,
		MaxAge:   600,
		SameSite: http.SameSiteLaxMode,
	})
	// Stash the post-login landing URL.  We only accept site-relative paths
	// so this can't be turned into an open-redirect gadget.
	if next := r.URL.Query().Get("next"); isSafeNext(next) {
		http.SetCookie(w, &http.Cookie{
			Name:     "oauth_next",
			Value:    next,
			Path:     "/auth",
			HttpOnly: true,
			Secure:   secure,
			MaxAge:   600,
			SameSite: http.SameSiteLaxMode,
		})
	}
	q := url.Values{}
	q.Set("client_id", s.cfg.githubClient)
	q.Set("redirect_uri", s.cfg.publicURL+"/auth/github/callback")
	q.Set("scope", "read:user")
	q.Set("state", state)
	http.Redirect(w, r, githubAuthorizeURL+"?"+q.Encode(), http.StatusFound)
}

func (s *server) handleGithubCallback(w http.ResponseWriter, r *http.Request) {
	if s.cfg.githubClient == "" {
		http.Error(w, "GitHub OAuth not configured", 501)
		return
	}

	code := r.URL.Query().Get("code")
	state := r.URL.Query().Get("state")
	sc, err := r.Cookie("oauth_state")
	if err != nil || sc.Value == "" || sc.Value != state {
		http.Error(w, "bad oauth state", 400)
		return
	}
	if !s.verifyOAuthState(state, s.clientIP(r)) {
		// Cookie matched the state echoed back by GitHub but the HMAC
		// doesn't validate against this client IP or has expired.
		// Refuse — an attacker who pried the cookie loose from a
		// different IP shouldn't be able to complete the flow.
		http.Error(w, "oauth state expired or bound to a different client", 400)
		return
	}
	if code == "" {
		http.Error(w, "missing code", 400)
		return
	}

	// Exchange code for access_token.
	tok, err := exchangeGithubCode(s.cfg.githubClient, s.cfg.githubSecret, code)
	if err != nil {
		http.Error(w, "exchange: "+err.Error(), 502)
		return
	}

	// Fetch user info.
	gh, err := fetchGithubUser(tok)
	if err != nil {
		http.Error(w, "fetch user: "+err.Error(), 502)
		return
	}

	// Upsert.
	var uid int64
	err = s.dbQueryRow(
		`SELECT id FROM users WHERE github_id = ?`, gh.ID,
	).Scan(&uid)
	if err != nil {
		display := gh.Name
		if display == "" {
			display = gh.Login
		}
		// Github profile names are user-controlled and can carry the
		// same hostile shapes (BiDi overrides, overlong, non-NFC) as a
		// dev-login submission.  Fall back to the login on rejection,
		// then to a placeholder so we never persist garbage.
		if clean, err := sanitizeDisplayName(display); err == nil {
			display = clean
		} else if clean, err := sanitizeDisplayName(gh.Login); err == nil {
			display = clean
		} else {
			display = "user"
		}
		res, err := s.dbExec(
			`INSERT INTO users (github_id, github_login, display_name, created_at)
			 VALUES (?, ?, ?, ?)`,
			gh.ID, gh.Login, display, nowUnix(),
		)
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		uid, _ = res.LastInsertId()
	}

	sid, exp, err := s.createSession(uid)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	s.setSessionCookie(w, sid, exp)
	dest := "/console"
	if c, err := r.Cookie("oauth_next"); err == nil && isSafeNext(c.Value) {
		dest = c.Value
		// Clear the cookie so it doesn't sticky across logins.
		http.SetCookie(w, &http.Cookie{
			Name:     "oauth_next",
			Value:    "",
			Path:     "/auth",
			HttpOnly: true,
			Secure:   s.secureCookies(),
			MaxAge:   -1,
		})
	}
	http.Redirect(w, r, dest, http.StatusFound)
}

type githubUser struct {
	ID    int64  `json:"id"`
	Login string `json:"login"`
	Name  string `json:"name"`
}

func exchangeGithubCode(clientID, clientSecret, code string) (string, error) {
	form := url.Values{}
	form.Set("client_id", clientID)
	form.Set("client_secret", clientSecret)
	form.Set("code", code)

	req, _ := http.NewRequest("POST", githubTokenURL,
		strings.NewReader(form.Encode()))
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	c := &http.Client{Timeout: 10 * time.Second}
	resp, err := c.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var out struct {
		AccessToken string `json:"access_token"`
		Error       string `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", err
	}
	if out.AccessToken == "" {
		return "", &oauthErr{"github: " + out.Error}
	}
	return out.AccessToken, nil
}

func fetchGithubUser(accessToken string) (*githubUser, error) {
	req, _ := http.NewRequest("GET", githubUserURL, nil)
	req.Header.Set("Accept", "application/vnd.github+json")
	req.Header.Set("Authorization", "Bearer "+accessToken)
	c := &http.Client{Timeout: 10 * time.Second}
	resp, err := c.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(resp.Body)
		return nil, &oauthErr{"github /user: " + string(b)}
	}
	var u githubUser
	if err := json.NewDecoder(resp.Body).Decode(&u); err != nil {
		return nil, err
	}
	return &u, nil
}

type oauthErr struct{ s string }

func (e *oauthErr) Error() string { return e.s }
