package main

import (
	"strconv"
	"strings"
	"testing"
	"time"
)

// newMacaroonTestServer builds a minimal server with just the fields the
// macaroon helpers reach for (sessionSecret + publicURL).  Avoids the
// full newServer() path so this test stays a unit test.
func newMacaroonTestServer(t *testing.T) *server {
	t.Helper()
	return &server{
		cfg: config{
			sessionSecret: "test-secret-do-not-use-in-prod-aaaaaaaaaaaaaa",
			publicURL:     "https://example.test",
		},
	}
}

// TestMintAndVerifyShardCap — happy path: a cap minted for (model,file)
// verifies, and the same cap rejected if any matched value differs.
func TestMintAndVerifyShardCap(t *testing.T) {
	s := newMacaroonTestServer(t)
	tok, err := s.mintShardCap(42, "stage-0.gguf", 5*time.Minute)
	if err != nil {
		t.Fatalf("mintShardCap: %v", err)
	}
	if err := s.verifyShardCap(tok, 42, "stage-0.gguf"); err != nil {
		t.Fatalf("verify good: %v", err)
	}
	// Wrong model
	if err := s.verifyShardCap(tok, 43, "stage-0.gguf"); err == nil {
		t.Fatalf("expected model mismatch rejection")
	}
	// Wrong file
	if err := s.verifyShardCap(tok, 42, "stage-1.gguf"); err == nil {
		t.Fatalf("expected file mismatch rejection")
	}
}

// TestShardCapExpiry — a cap with a -1s TTL (i.e. already expired) must
// reject; one with a generous TTL must accept.
func TestShardCapExpiry(t *testing.T) {
	s := newMacaroonTestServer(t)
	tok, err := s.mintShardCap(1, "f.gguf", -1*time.Second)
	if err != nil {
		t.Fatalf("mint: %v", err)
	}
	if err := s.verifyShardCap(tok, 1, "f.gguf"); err == nil {
		t.Fatalf("expected expired cap rejection")
	}
}

// TestCapPathNamespaceIsolation — a shard cap MUST NOT verify as a
// comfy cap and vice versa.  This is the property that makes the
// path= caveat load-bearing.
func TestCapPathNamespaceIsolation(t *testing.T) {
	s := newMacaroonTestServer(t)
	shardTok, _ := s.mintShardCap(7, "x", time.Hour)
	comfyTok, _ := s.mintComfyOutputCap(11, 7, "x", time.Hour)
	if err := s.verifyComfyOutputCap(shardTok, 11, 7, "x"); err == nil {
		t.Fatalf("shard cap must not verify as comfy cap")
	}
	if err := s.verifyShardCap(comfyTok, 7, "x"); err == nil {
		t.Fatalf("comfy cap must not verify as shard cap")
	}
}

// TestCapKeyRotation — caps minted under secret A must not verify under
// secret B.  This is the property that lets an operator rotate
// sessionSecret to invalidate every outstanding cap in one move.
func TestCapKeyRotation(t *testing.T) {
	s := newMacaroonTestServer(t)
	tok, _ := s.mintShardCap(1, "f", time.Hour)
	s2 := &server{cfg: config{sessionSecret: "different-secret-xxxxxxxxxxxxxxxxx"}}
	if err := s2.verifyShardCap(tok, 1, "f"); err == nil {
		t.Fatalf("cap must not verify under rotated key")
	}
}

// TestCapTamperRejection — flipping a byte in the encoded token must
// produce a verify failure.  Macaroon HMAC chain guarantees this; the
// test pins the property so a future refactor can't accidentally trade
// it away.
//
// Implementation note: we flip a *middle* char rather than the last
// one.  Go's base64.RawURLEncoding is non-strict by default — it
// ignores trailing padding bits during decode — so toggling the lowest
// bit of the trailing char can decode to the same bytes when the
// encoded payload length isn't divisible by 3.  Middle chars always
// occupy canonical (non-padding) bit ranges, so a flip there is
// guaranteed to change the decoded byte stream and thus the HMAC.
func TestCapTamperRejection(t *testing.T) {
	s := newMacaroonTestServer(t)
	tok, _ := s.mintShardCap(1, "f", time.Hour)
	tampered := []byte(tok)
	mid := len(tampered) / 2
	if tampered[mid] == 'A' {
		tampered[mid] = 'B'
	} else {
		tampered[mid] = 'A'
	}
	if err := s.verifyShardCap(string(tampered), 1, "f"); err == nil {
		t.Fatalf("expected tamper rejection")
	}
}

// TestComfyOutputCapBindsUID — a cap minted for uid=7 must not verify
// as a cap for uid=8.  This is the property that A11 introduced and
// P7 inherits.
func TestComfyOutputCapBindsUID(t *testing.T) {
	s := newMacaroonTestServer(t)
	tok, _ := s.mintComfyOutputCap(7, 42, "img.png", time.Hour)
	if err := s.verifyComfyOutputCap(tok, 7, 42, "img.png"); err != nil {
		t.Fatalf("uid 7 verify: %v", err)
	}
	if err := s.verifyComfyOutputCap(tok, 8, 42, "img.png"); err == nil {
		t.Fatalf("uid 8 must reject")
	}
	if err := s.verifyComfyOutputCap(tok, 7, 43, "img.png"); err == nil {
		t.Fatalf("job 43 must reject")
	}
	if err := s.verifyComfyOutputCap(tok, 7, 42, "OTHER.png"); err == nil {
		t.Fatalf("file mismatch must reject")
	}
}

// TestMintShardURLCap — sanity that mintShardURLCap returns a URL with
// the cap= query param and no exp/sig parameters (P7 path).
func TestMintShardURLCap(t *testing.T) {
	s := newMacaroonTestServer(t)
	u := s.mintShardURLCap(7, "stage-0.gguf", time.Hour)
	if !strings.Contains(u, "cap=") {
		t.Fatalf("URL missing cap=: %s", u)
	}
	if strings.Contains(u, "exp=") || strings.Contains(u, "sig=") {
		t.Fatalf("URL must not carry legacy HMAC params: %s", u)
	}
}

// TestUnexpectedCaveatRejected — a caveat with a key the verifier
// doesn't know about MUST fail.  Macaroons are attenuating-only: only
// further-restricting caveats may be added by holders, never expanding.
// We enforce strict key whitelist; a stray caveat = misuse, fail loud.
func TestUnexpectedCaveatRejected(t *testing.T) {
	s := newMacaroonTestServer(t)
	tok, err := s.mintCap(
		"path=shard",
		"model=1",
		"file=f",
		"exp="+strconv.FormatInt(time.Now().Add(time.Hour).Unix(), 10),
		"bogus=value",
	)
	if err != nil {
		t.Fatalf("mintCap: %v", err)
	}
	if err := s.verifyShardCap(tok, 1, "f"); err == nil {
		t.Fatalf("expected unexpected-caveat rejection")
	}
}

// TestEmptyCapRejected — verifyCap on an empty token must reject.
func TestEmptyCapRejected(t *testing.T) {
	s := newMacaroonTestServer(t)
	if err := s.verifyShardCap("", 1, "f"); err == nil {
		t.Fatalf("expected empty cap rejection")
	}
}
