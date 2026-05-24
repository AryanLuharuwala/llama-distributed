package main

// P7: Macaroon-based URL capabilities.
//
// Replaces the bespoke HMAC URL signing scheme used for shard downloads
// and comfy outputs with a proper macaroon (RFC-style bearer token with
// first-party caveats).  The win over plain HMAC is composability: the
// minter declares the exact scope (path, uid, resource, filename, expiry)
// as caveats inside the token, and the verifier walks them — no per-URL
// query parameters fan-out, no key-by-key handler code that drifts away
// from the minter as new fields ship.
//
// On the wire the token rides as a single `cap=<base64>` query param.
// The existing `exp=` / `sig=` / `v=` / `uid=` URL shapes stay verifiable
// for one mintHMACGraceWindow after boot so URLs minted just before a
// rolling deploy don't 401 mid-fetch.
//
// Caveat language (first-party only — third-party caveats are out of
// scope for now; we don't have a discharge service).  Each caveat is a
// `key=value` string:
//
//   path=<shard|comfy-out>     — namespaces tokens; a shard cap can't
//                                 unlock a comfy output and vice versa.
//   uid=<int64>                — bind to a specific user (comfy path).
//   model=<int64>              — bind to a model row (shard path).
//   job=<int64>                — bind to a comfy_jobs row (comfy path).
//   file=<filename>            — exact filename, case-sensitive.
//   exp=<unix-seconds>         — token expires at this wall-clock time.
//
// The macaroon's HMAC root key is the server's sessionSecret — same key
// we used for the legacy HMAC URLs, so existing deployments don't need
// a key rotation on top of the rollout.  A constant-time ID prefix
// (`distpool/v1`) lets future versions discriminate.

import (
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"fmt"
	"strconv"
	"strings"
	"time"

	"gopkg.in/macaroon.v2"
)

const (
	macaroonIDPrefix = "distpool/v1"
	macaroonLocation = "distpool"
	mintHMACGraceWindow = 24 * time.Hour
)

// macaroonRootKey derives the HMAC key from the session secret.  We hash
// rather than using the secret directly so that even if the macaroon
// scheme is later forked / cloned in a different binary, the keying is
// domain-separated from any other HMAC use of sessionSecret.
func (s *server) macaroonRootKey() []byte {
	h := sha256.Sum256([]byte("distpool-macaroon-v1\x00" + s.cfg.sessionSecret))
	return h[:]
}

// mintCap builds a macaroon with the given first-party caveats, encodes
// it as base64, and returns the token.  Caveats are added in the order
// passed — order is not security-significant but does affect token bytes
// (callers should pass a deterministic order if they want stable tokens
// for the same input).
func (s *server) mintCap(caveats ...string) (string, error) {
	m, err := macaroon.New(s.macaroonRootKey(), []byte(macaroonIDPrefix), macaroonLocation, macaroon.V2)
	if err != nil {
		return "", fmt.Errorf("macaroon.New: %w", err)
	}
	for _, cv := range caveats {
		if err := m.AddFirstPartyCaveat([]byte(cv)); err != nil {
			return "", fmt.Errorf("AddFirstPartyCaveat(%q): %w", cv, err)
		}
	}
	b, err := m.MarshalBinary()
	if err != nil {
		return "", fmt.Errorf("MarshalBinary: %w", err)
	}
	return base64.RawURLEncoding.EncodeToString(b), nil
}

// verifyCap decodes the token, verifies its signature against the
// server's root key, and runs the caveats through the supplied checker.
// Returns nil on success.  The checker receives one caveat at a time
// (the verifier walks the caveat list and returns the first error).
//
// `require` is the set of caveat keys that MUST appear in the macaroon
// (always including "exp"). After verification succeeds, we cross-check
// that every required key was seen — without this, a macaroon minted
// without (e.g.) the `file` caveat would verify against any file path.
// F-AUTH-01: rejecting unknown caveats is not enough; we must also
// reject missing-expected caveats.
func (s *server) verifyCap(token string, check func(caveat string) error, require map[string]bool) error {
	if token == "" {
		return fmt.Errorf("empty cap")
	}
	raw, err := base64.RawURLEncoding.DecodeString(token)
	if err != nil {
		return fmt.Errorf("base64 decode: %w", err)
	}
	var m macaroon.Macaroon
	if err := m.UnmarshalBinary(raw); err != nil {
		return fmt.Errorf("unmarshal: %w", err)
	}
	// Strict round-trip: the macaroon library's UnmarshalBinary is
	// lenient about trailing bytes, which would let an attacker append
	// junk to a token and have it still verify.  Re-marshal and compare
	// in constant time so the timing of mismatch detection doesn't leak
	// position-of-first-difference.
	round, err := m.MarshalBinary()
	if err != nil || len(round) != len(raw) {
		return fmt.Errorf("token length mismatch (trailing-junk attack?)")
	}
	if subtle.ConstantTimeCompare(round, raw) != 1 {
		return fmt.Errorf("token bytes mismatch")
	}
	// Wrap `check` so we observe each caveat key as the verifier walks
	// the list. After Verify returns success, we assert every required
	// key was visited.
	seen := make(map[string]bool, len(require))
	wrapped := func(cv string) error {
		if eq := strings.IndexByte(cv, '='); eq >= 0 {
			seen[cv[:eq]] = true
		}
		return check(cv)
	}
	// macaroon.v2.Verify takes the root key + a check function + any
	// discharge macaroons (none, since we only use first-party caveats).
	if err := m.Verify(s.macaroonRootKey(), wrapped, nil); err != nil {
		return fmt.Errorf("verify: %w", err)
	}
	for k := range require {
		if !seen[k] {
			return fmt.Errorf("required caveat %q missing", k)
		}
	}
	return nil
}

// capChecker returns a checker function for `verifyCap` that matches the
// expected (key, value) pairs.  `exp` is special-cased: an exp= caveat
// is accepted iff the encoded unix time is strictly greater than `now`.
// All other expected keys must match exactly.  Caveats whose keys are
// not in `expected` cause an error (defense against expanded scope:
// adding a caveat to a macaroon can only narrow the grant, but an
// unexpected key signals a misuse and we'd rather fail loud).
func capChecker(now time.Time, expected map[string]string) func(string) error {
	return func(cv string) error {
		eq := strings.IndexByte(cv, '=')
		if eq < 0 {
			return fmt.Errorf("malformed caveat %q", cv)
		}
		k, v := cv[:eq], cv[eq+1:]
		if k == "exp" {
			ts, err := strconv.ParseInt(v, 10, 64)
			if err != nil {
				return fmt.Errorf("bad exp %q: %w", v, err)
			}
			if now.Unix() > ts {
				return fmt.Errorf("cap expired at %d (now=%d)", ts, now.Unix())
			}
			return nil
		}
		want, ok := expected[k]
		if !ok {
			return fmt.Errorf("unexpected caveat key %q", k)
		}
		if want != v {
			return fmt.Errorf("caveat %s=%s mismatch (want %s)", k, v, want)
		}
		return nil
	}
}

// ─── Shard download capability ─────────────────────────────────────────────

func (s *server) mintShardCap(modelID int64, file string, ttl time.Duration) (string, error) {
	return s.mintCap(
		"path=shard",
		"model="+strconv.FormatInt(modelID, 10),
		"file="+file,
		"exp="+strconv.FormatInt(time.Now().Add(ttl).Unix(), 10),
	)
}

func (s *server) verifyShardCap(token string, modelID int64, file string) error {
	expected := map[string]string{
		"path":  "shard",
		"model": strconv.FormatInt(modelID, 10),
		"file":  file,
	}
	require := map[string]bool{"path": true, "model": true, "file": true, "exp": true}
	return s.verifyCap(token, capChecker(time.Now(), expected), require)
}

func (s *server) mintShardURLCap(modelID int64, file string, ttl time.Duration) string {
	tok, err := s.mintShardCap(modelID, file, ttl)
	if err != nil {
		return ""
	}
	return fmt.Sprintf("%s/models/%d/shards/%s?cap=%s",
		strings.TrimRight(s.cfg.publicURL, "/"), modelID, file, tok)
}

// ─── Comfy output capability ───────────────────────────────────────────────

func (s *server) mintComfyOutputCap(uid, jobID int64, file string, ttl time.Duration) (string, error) {
	return s.mintCap(
		"path=comfy-out",
		"uid="+strconv.FormatInt(uid, 10),
		"job="+strconv.FormatInt(jobID, 10),
		"file="+file,
		"exp="+strconv.FormatInt(time.Now().Add(ttl).Unix(), 10),
	)
}

func (s *server) verifyComfyOutputCap(token string, uid, jobID int64, file string) error {
	expected := map[string]string{
		"path": "comfy-out",
		"uid":  strconv.FormatInt(uid, 10),
		"job":  strconv.FormatInt(jobID, 10),
		"file": file,
	}
	require := map[string]bool{"path": true, "uid": true, "job": true, "file": true, "exp": true}
	return s.verifyCap(token, capChecker(time.Now(), expected), require)
}

func (s *server) mintComfyOutputURLCap(uid, jobID int64, file string, ttl time.Duration) string {
	tok, err := s.mintComfyOutputCap(uid, jobID, file, ttl)
	if err != nil {
		return ""
	}
	return fmt.Sprintf("%s/comfy/out/%d/%s?cap=%s",
		strings.TrimRight(s.cfg.publicURL, "/"), jobID, file, tok)
}
