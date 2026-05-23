package main

import (
	"net/http/httptest"
	"testing"
	"time"
)

func TestSessionCookieSecureFlagHTTPS(t *testing.T) {
	s := &server{cfg: config{publicURL: "https://pool.example.com"}}
	w := httptest.NewRecorder()
	s.setSessionCookie(w, "sid-abc", time.Unix(2000000000, 0))
	cookies := w.Result().Cookies()
	if len(cookies) != 1 || cookies[0].Name != sessionCookieName {
		t.Fatalf("expected 1 session cookie, got %+v", cookies)
	}
	if !cookies[0].Secure {
		t.Errorf("Secure flag must be set when publicURL is https://")
	}
	if !cookies[0].HttpOnly {
		t.Errorf("HttpOnly must always be set")
	}
}

func TestSessionCookieSecureFlagHTTP(t *testing.T) {
	s := &server{cfg: config{publicURL: "http://localhost:8080"}}
	w := httptest.NewRecorder()
	s.setSessionCookie(w, "sid-abc", time.Unix(2000000000, 0))
	cookies := w.Result().Cookies()
	if len(cookies) != 1 {
		t.Fatalf("expected 1 cookie, got %d", len(cookies))
	}
	if cookies[0].Secure {
		t.Errorf("Secure flag must NOT be set on plain-HTTP deployment "+
			"(browser would refuse to send it back); got %+v", cookies[0])
	}
}

func TestClearSessionCookieSecureFlag(t *testing.T) {
	s := &server{cfg: config{publicURL: "HTTPS://Pool.Example.com"}}
	w := httptest.NewRecorder()
	s.clearSessionCookie(w)
	cookies := w.Result().Cookies()
	if len(cookies) != 1 {
		t.Fatalf("expected 1 cookie, got %d", len(cookies))
	}
	if !cookies[0].Secure {
		t.Errorf("clear-cookie must mirror Secure flag so browsers actually accept the expiry")
	}
	if cookies[0].MaxAge != -1 {
		t.Errorf("clear cookie should have MaxAge=-1, got %d", cookies[0].MaxAge)
	}
}

func TestSecureCookiesMethod(t *testing.T) {
	cases := []struct {
		url  string
		want bool
	}{
		{"https://pool.example.com", true},
		{"HTTPS://Pool.Example.com", true},
		{"http://localhost:8080", false},
		{"", false},
		{"ws://x", false},
	}
	for _, c := range cases {
		s := &server{cfg: config{publicURL: c.url}}
		if got := s.secureCookies(); got != c.want {
			t.Errorf("secureCookies(%q) = %v want %v", c.url, got, c.want)
		}
	}
}
