
package main

import (
	_ "embed"
	"encoding/json"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

//go:embed assets/auth.html
var authPageHTML []byte

// handleAuthPage serves the unified sign-in / signup landing page.  The
// page works for both first-time and returning users — there is no
// separate signup flow, the OAuth callback upserts on github_id.
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
		"dev_mode":          s.cfg.devMode,
		"user_id":           uid,
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
const githubTokenURL     = "https://github.com/login/oauth/access_token"
const githubUserURL      = "https://api.github.com/user"

func (s *server) handleGithubStart(w http.ResponseWriter, r *http.Request) {
	if s.cfg.githubClient == "" {
		http.Error(w, "GitHub OAuth not configured (set DIST_GITHUB_CLIENT/SECRET)", 501)
		return
	}
	state := newRandomToken(16)
	http.SetCookie(w, &http.Cookie{
		Name:     "oauth_state",
		Value:    state,
		Path:     "/auth",
		HttpOnly: true,
		MaxAge:   600,
		SameSite: http.SameSiteLaxMode,
	})
	// Stash the post-login landing URL.  We only accept site-relative paths
	// so this can't be turned into an open-redirect gadget.
	if next := r.URL.Query().Get("next"); next != "" && strings.HasPrefix(next, "/") && !strings.HasPrefix(next, "//") {
		http.SetCookie(w, &http.Cookie{
			Name:     "oauth_next",
			Value:    next,
			Path:     "/auth",
			HttpOnly: true,
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
	err = s.db.QueryRow(
		`SELECT id FROM users WHERE github_id = ?`, gh.ID,
	).Scan(&uid)
	if err != nil {
		display := gh.Name
		if display == "" {
			display = gh.Login
		}
		res, err := s.db.Exec(
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
	if c, err := r.Cookie("oauth_next"); err == nil && strings.HasPrefix(c.Value, "/") && !strings.HasPrefix(c.Value, "//") {
		dest = c.Value
		// Clear the cookie so it doesn't sticky across logins.
		http.SetCookie(w, &http.Cookie{Name: "oauth_next", Value: "", Path: "/auth", MaxAge: -1})
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
