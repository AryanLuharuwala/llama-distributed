package main

import (
	"crypto/ed25519"
	"crypto/rand"
	"encoding/base64"
	"errors"
	"fmt"
	"strconv"
	"time"
)

// Signed agent identities.
//
// On first pair the rig generates an ed25519 keypair and uploads the public
// key as part of the hello frame:
//
//   {"kind":"hello", "token":"…", "agent_id":"…", "pubkey":"<b64>", …}
//
// The server stores the pubkey on the rigs row.  On every resume the rig
// signs a (server-provided) challenge nonce with its private key:
//
//   1. server sends   {"kind":"challenge", "nonce":"<b64>", "ts":<epoch>}
//   2. rig signs      sigInput = "agent-id|nonce|ts"  (utf-8)
//   3. rig sends      {"kind":"sig", "sig":"<b64>"}
//
// The server verifies the signature against the stored pubkey before
// accepting the resume.  If pubkey is absent (rig was paired before this
// feature shipped) the server logs a warning but accepts the resume on
// agent_key alone — gives existing fleets time to roll over.

const (
	// 32 bytes — enough entropy to make replay infeasible.
	challengeNonceBytes = 32
	// Sig must arrive within this window or the server rejects the resume
	// (defends against an attacker capturing a (nonce, sig) pair and
	// replaying it later against a new connection).
	challengeWindowSec = 30
)

// mintChallenge produces (nonce_b64, ts) for the server → rig "challenge"
// frame.  Returns base64-url-no-padding strings so they're safe to embed
// in JSON without escaping.
func mintChallenge() (string, int64, error) {
	b := make([]byte, challengeNonceBytes)
	if _, err := rand.Read(b); err != nil {
		return "", 0, err
	}
	return base64.RawURLEncoding.EncodeToString(b), nowUnix(), nil
}

// signedChallengeInput builds the exact bytes the rig signs.  Server and
// rig must agree on this format down to the byte; format is:
//
//   "dist-agent-v1|" + agentID + "|" + nonceB64 + "|" + ts
//
// (Domain-separator prefix prevents an attacker from getting the rig to
// sign an attacker-chosen value that the server interprets as something
// else, e.g. a JWT or another protocol's auth blob.)
func signedChallengeInput(agentID, nonceB64 string, ts int64) []byte {
	return []byte("dist-agent-v1|" + agentID + "|" + nonceB64 + "|" + strconv.FormatInt(ts, 10))
}

// verifyAgentSig returns nil if `sig` is a valid ed25519 signature over
// signedChallengeInput(agentID, nonce, ts) using `pubkey`.
//
// Returns a plain error on any mismatch — callers should treat any error
// as "reject this resume".
func verifyAgentSig(pubkey []byte, agentID, nonceB64 string, ts int64, sig []byte) error {
	if len(pubkey) != ed25519.PublicKeySize {
		return fmt.Errorf("pubkey: expected %d bytes, got %d", ed25519.PublicKeySize, len(pubkey))
	}
	if len(sig) != ed25519.SignatureSize {
		return fmt.Errorf("sig: expected %d bytes, got %d", ed25519.SignatureSize, len(sig))
	}
	// Window check — reject ancient sigs.
	now := nowUnix()
	if ts > now+5 {
		return errors.New("sig: ts is in the future")
	}
	if now-ts > challengeWindowSec {
		return errors.New("sig: ts is too old")
	}
	msg := signedChallengeInput(agentID, nonceB64, ts)
	if !ed25519.Verify(ed25519.PublicKey(pubkey), msg, sig) {
		return errors.New("sig: ed25519 verify failed")
	}
	return nil
}

// decodePubkeyB64 parses a base64-encoded ed25519 public key — accepts
// both standard and URL-safe encodings, with or without padding.
func decodePubkeyB64(s string) ([]byte, error) {
	for _, enc := range []*base64.Encoding{
		base64.RawURLEncoding, base64.URLEncoding,
		base64.RawStdEncoding, base64.StdEncoding,
	} {
		if b, err := enc.DecodeString(s); err == nil {
			if len(b) == ed25519.PublicKeySize {
				return b, nil
			}
		}
	}
	return nil, errors.New("pubkey: not a valid base64 ed25519 public key")
}

// decodeSigB64 is the same shape as decodePubkeyB64 but for signatures.
func decodeSigB64(s string) ([]byte, error) {
	for _, enc := range []*base64.Encoding{
		base64.RawURLEncoding, base64.URLEncoding,
		base64.RawStdEncoding, base64.StdEncoding,
	} {
		if b, err := enc.DecodeString(s); err == nil {
			if len(b) == ed25519.SignatureSize {
				return b, nil
			}
		}
	}
	return nil, errors.New("sig: not a valid base64 ed25519 signature")
}

// challengeTimeoutCtx is the timeout we wait for the rig to send its
// signature back after the challenge.  Generous because some rigs sit
// behind high-latency tunnels.
const challengeTimeout = 10 * time.Second
