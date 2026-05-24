package main

// Google OAuth 2.0 — minimal flow that mirrors the GitHub handler.
//
//   /auth/google            → redirect to accounts.google.com/o/oauth2/v2/auth
//   /auth/google/callback   → exchange code, fetch userinfo, upsert + cookie
//
// Requires DIST_GOOGLE_CLIENT + DIST_GOOGLE_SECRET env vars.  Without them,
// the handlers return 501 so the auth page can hide the button.  The
// redirect_uri sent to Google must be registered on the OAuth client —
// the page tells the user how to add it (publicURL + /auth/google/callback).

import (
	"encoding/json"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

const (
	googleAuthURL     = "https://accounts.google.com/o/oauth2/v2/auth"
	googleTokenURL    = "https://oauth2.googleapis.com/token"
	googleUserinfoURL = "https://openidconnect.googleapis.com/v1/userinfo"
)

type googleUser struct {
	Sub     string `json:"sub"`
	Email   string `json:"email"`
	Name    string `json:"name"`
	Picture string `json:"picture"`
}

func (s *server) handleGoogleStart(w http.ResponseWriter, r *http.Request) {
	if s.cfg.googleClient == "" {
		http.Error(w, "Google OAuth not configured (set DIST_GOOGLE_CLIENT/SECRET)", 501)
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
	if next := r.URL.Query().Get("next"); next != "" && strings.HasPrefix(next, "/") && !strings.HasPrefix(next, "//") {
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
	q.Set("client_id", s.cfg.googleClient)
	q.Set("redirect_uri", s.cfg.publicURL+"/auth/google/callback")
	q.Set("response_type", "code")
	q.Set("scope", "openid email profile")
	q.Set("state", state)
	q.Set("access_type", "online")
	q.Set("prompt", "select_account")
	http.Redirect(w, r, googleAuthURL+"?"+q.Encode(), http.StatusFound)
}

func (s *server) handleGoogleCallback(w http.ResponseWriter, r *http.Request) {
	if s.cfg.googleClient == "" {
		http.Error(w, "Google OAuth not configured", 501)
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
		http.Error(w, "oauth state expired or bound to a different client", 400)
		return
	}
	if code == "" {
		http.Error(w, "missing code", 400)
		return
	}

	tok, err := exchangeGoogleCode(s.cfg.googleClient, s.cfg.googleSecret, s.cfg.publicURL+"/auth/google/callback", code)
	if err != nil {
		http.Error(w, "exchange: "+err.Error(), 502)
		return
	}
	gu, err := fetchGoogleUser(tok)
	if err != nil {
		http.Error(w, "fetch user: "+err.Error(), 502)
		return
	}
	if gu.Sub == "" {
		http.Error(w, "missing sub from userinfo", 502)
		return
	}

	var uid int64
	err = s.db.QueryRow(`SELECT id FROM users WHERE google_id = ?`, gu.Sub).Scan(&uid)
	if err != nil {
		display := gu.Name
		if display == "" {
			display = strings.SplitN(gu.Email, "@", 2)[0]
		}
		if clean, err := sanitizeDisplayName(display); err == nil {
			display = clean
		} else {
			display = "user"
		}
		res, err := s.db.Exec(
			`INSERT INTO users (google_id, google_email, display_name, created_at)
			 VALUES (?, ?, ?, ?)`,
			gu.Sub, gu.Email, display, nowUnix(),
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
	dest := "/nexus"
	if c, err := r.Cookie("oauth_next"); err == nil && strings.HasPrefix(c.Value, "/") && !strings.HasPrefix(c.Value, "//") {
		dest = c.Value
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

func exchangeGoogleCode(clientID, clientSecret, redirectURI, code string) (string, error) {
	form := url.Values{}
	form.Set("client_id", clientID)
	form.Set("client_secret", clientSecret)
	form.Set("code", code)
	form.Set("grant_type", "authorization_code")
	form.Set("redirect_uri", redirectURI)

	req, _ := http.NewRequest("POST", googleTokenURL, strings.NewReader(form.Encode()))
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
		ErrorDesc   string `json:"error_description"`
		ErrorCode   string `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", err
	}
	if out.AccessToken == "" {
		msg := out.ErrorDesc
		if msg == "" {
			msg = out.ErrorCode
		}
		return "", &oauthErr{"google: " + msg}
	}
	return out.AccessToken, nil
}

func fetchGoogleUser(accessToken string) (*googleUser, error) {
	req, _ := http.NewRequest("GET", googleUserinfoURL, nil)
	req.Header.Set("Authorization", "Bearer "+accessToken)
	req.Header.Set("Accept", "application/json")
	c := &http.Client{Timeout: 10 * time.Second}
	resp, err := c.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(resp.Body)
		return nil, &oauthErr{"google userinfo: " + string(b)}
	}
	var u googleUser
	if err := json.NewDecoder(resp.Body).Decode(&u); err != nil {
		return nil, err
	}
	return &u, nil
}
