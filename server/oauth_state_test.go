package main

import (
	"encoding/base64"
	"fmt"
	"strings"
	"testing"
)

func formatStateForTest(ts int64, mac []byte) string {
	return fmt.Sprintf("%d.%s", ts, base64.RawURLEncoding.EncodeToString(mac))
}

func TestOAuthStateRoundTrip(t *testing.T) {
	s := &server{cfg: config{sessionSecret: "test-secret-1234567890"}}
	ip := "198.51.100.7"
	state := s.mintOAuthState(ip)
	if state == "" || !strings.Contains(state, ".") {
		t.Fatalf("mintOAuthState produced bad shape: %q", state)
	}
	if !s.verifyOAuthState(state, ip) {
		t.Errorf("freshly-minted state must verify against the same IP+secret")
	}
}

func TestOAuthStateRejectsDifferentIP(t *testing.T) {
	s := &server{cfg: config{sessionSecret: "test-secret"}}
	state := s.mintOAuthState("203.0.113.1")
	if s.verifyOAuthState(state, "203.0.113.99") {
		t.Errorf("state from one IP must not verify on another — that's the whole point of A7")
	}
}

func TestOAuthStateRejectsDifferentSecret(t *testing.T) {
	s := &server{cfg: config{sessionSecret: "secret-a"}}
	state := s.mintOAuthState("198.51.100.7")
	s2 := &server{cfg: config{sessionSecret: "secret-b"}}
	if s2.verifyOAuthState(state, "198.51.100.7") {
		t.Errorf("state minted by one secret must not validate under another")
	}
}

func TestOAuthStateRejectsTampered(t *testing.T) {
	s := &server{cfg: config{sessionSecret: "test-secret"}}
	ip := "198.51.100.7"
	state := s.mintOAuthState(ip)

	// Flip a character in the HMAC portion.
	dot := strings.IndexByte(state, '.')
	if dot <= 0 {
		t.Fatalf("unexpected state shape: %q", state)
	}
	// Mutate the first HMAC byte.
	b := []byte(state)
	if b[dot+1] == 'A' {
		b[dot+1] = 'B'
	} else {
		b[dot+1] = 'A'
	}
	if s.verifyOAuthState(string(b), ip) {
		t.Errorf("tampered HMAC must not validate")
	}
}

func TestOAuthStateRejectsExpired(t *testing.T) {
	s := &server{cfg: config{sessionSecret: "test-secret"}}
	ip := "198.51.100.7"
	// Compose a state with a timestamp that's older than oauthStateMaxAge.
	old := nowUnix() - oauthStateMaxAge - 60
	mac := s.computeOAuthStateMAC(old, ip)
	state := formatStateForTest(old, mac)
	if s.verifyOAuthState(state, ip) {
		t.Errorf("expired state must not validate")
	}
}

func TestOAuthStateRejectsMalformed(t *testing.T) {
	s := &server{cfg: config{sessionSecret: "test-secret"}}
	cases := []string{
		"",
		".",
		"abc",
		"1234567890",
		".abc",
		"abc.",
		"notanumber.abc",
	}
	for _, c := range cases {
		if s.verifyOAuthState(c, "x") {
			t.Errorf("malformed state %q must not validate", c)
		}
	}
}
