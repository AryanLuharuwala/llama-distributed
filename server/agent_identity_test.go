package main

import (
	"crypto/ed25519"
	"crypto/rand"
	"encoding/base64"
	"testing"
	"time"
)

func TestVerifyAgentSig_RoundTrip(t *testing.T) {
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	agentID := "rig-a:1234"
	nonce, ts, err := mintChallenge()
	if err != nil {
		t.Fatal(err)
	}
	msg := signedChallengeInput(agentID, nonce, ts)
	sig := ed25519.Sign(priv, msg)

	if err := verifyAgentSig(pub, agentID, nonce, ts, sig); err != nil {
		t.Fatalf("expected verify ok, got %v", err)
	}
}

func TestVerifyAgentSig_RejectsTampering(t *testing.T) {
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)
	nonce, ts, _ := mintChallenge()
	sig := ed25519.Sign(priv, signedChallengeInput("real-agent", nonce, ts))

	// Verify with the WRONG agent id — domain-separator prefix should fail.
	if err := verifyAgentSig(pub, "evil-agent", nonce, ts, sig); err == nil {
		t.Fatal("expected verify to fail for swapped agent id")
	}
}

func TestVerifyAgentSig_RejectsStaleTS(t *testing.T) {
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)
	agentID := "rig-x"
	// Forge a ts 5 minutes ago — well outside the 30s window.
	ts := time.Now().Add(-5 * time.Minute).Unix()
	nonce, _, _ := mintChallenge()
	sig := ed25519.Sign(priv, signedChallengeInput(agentID, nonce, ts))

	if err := verifyAgentSig(pub, agentID, nonce, ts, sig); err == nil {
		t.Fatal("expected verify to fail for stale ts")
	}
}

func TestVerifyAgentSig_RejectsFutureTS(t *testing.T) {
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)
	agentID := "rig-x"
	ts := time.Now().Add(1 * time.Minute).Unix()
	nonce, _, _ := mintChallenge()
	sig := ed25519.Sign(priv, signedChallengeInput(agentID, nonce, ts))

	if err := verifyAgentSig(pub, agentID, nonce, ts, sig); err == nil {
		t.Fatal("expected verify to fail for future ts")
	}
}

func TestVerifyAgentSig_RejectsWrongKey(t *testing.T) {
	_, priv, _ := ed25519.GenerateKey(rand.Reader)
	otherPub, _, _ := ed25519.GenerateKey(rand.Reader)

	agentID := "rig-x"
	nonce, ts, _ := mintChallenge()
	sig := ed25519.Sign(priv, signedChallengeInput(agentID, nonce, ts))

	if err := verifyAgentSig(otherPub, agentID, nonce, ts, sig); err == nil {
		t.Fatal("expected verify to fail when checked against wrong pubkey")
	}
}

func TestDecodePubkeyB64_AcceptsAllEncodings(t *testing.T) {
	pub, _, _ := ed25519.GenerateKey(rand.Reader)
	for _, enc := range []*base64.Encoding{
		base64.RawURLEncoding, base64.URLEncoding,
		base64.RawStdEncoding, base64.StdEncoding,
	} {
		s := enc.EncodeToString(pub)
		got, err := decodePubkeyB64(s)
		if err != nil {
			t.Fatalf("encoding %v: %v", enc, err)
		}
		if string(got) != string(pub) {
			t.Fatalf("encoding %v: pubkey roundtrip mismatch", enc)
		}
	}
}

func TestDecodePubkeyB64_RejectsGarbage(t *testing.T) {
	if _, err := decodePubkeyB64("not-base64-at-all-/&^%"); err == nil {
		t.Fatal("expected error on garbage input")
	}
	// Wrong-size base64 (e.g. only 16 bytes encoded).
	if _, err := decodePubkeyB64(base64.RawURLEncoding.EncodeToString(make([]byte, 16))); err == nil {
		t.Fatal("expected error on short input")
	}
}
