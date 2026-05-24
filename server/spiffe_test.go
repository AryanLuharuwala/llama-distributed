package main

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"math/big"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/go-jose/go-jose/v4"
	"github.com/go-jose/go-jose/v4/jwt"
	"github.com/spiffe/go-spiffe/v2/spiffeid"
)

// TestSpiffeIDFromXFCC walks the XFCC parser through the four wire
// shapes we expect from envoy: bare URI, semicolon-separated fields,
// quoted Subject containing a comma, and a missing-URI degenerate case.
func TestSpiffeIDFromXFCC(t *testing.T) {
	cases := []struct {
		name string
		hdr  string
		want string
		ok   bool
	}{
		{
			name: "bare uri",
			hdr:  "By=spiffe://prod.local/server;URI=spiffe://prod.local/rig/abc",
			want: "spiffe://prod.local/rig/abc",
			ok:   true,
		},
		{
			name: "quoted subject with comma",
			hdr:  `By=spiffe://prod.local/server;Hash=abc;Subject="CN=rig,OU=fleet";URI=spiffe://prod.local/rig/xyz`,
			want: "spiffe://prod.local/rig/xyz",
			ok:   true,
		},
		{
			name: "no uri field",
			hdr:  "By=spiffe://prod.local/server;Hash=abc",
			ok:   false,
		},
		{
			name: "empty",
			hdr:  "",
			ok:   false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			id, err := spiffeIDFromXFCC(tc.hdr)
			if tc.ok && err != nil {
				t.Fatalf("expected ok, got err: %v", err)
			}
			if !tc.ok && err == nil {
				t.Fatalf("expected error, got id=%s", id)
			}
			if tc.ok && id.String() != tc.want {
				t.Fatalf("id mismatch: got %s want %s", id, tc.want)
			}
		})
	}
}

// TestSpiffeIDFromCert exercises the X.509 SAN parser via a cert we
// generate inline.  The cert is self-signed because we only need the
// SAN URI extraction, not chain validation.
func TestSpiffeIDFromCert(t *testing.T) {
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf("gen key: %v", err)
	}
	u, _ := url.Parse("spiffe://prod.local/rig/test-1")
	tmpl := &x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject:      pkix.Name{CommonName: "rig"},
		NotBefore:    time.Now(),
		NotAfter:     time.Now().Add(time.Hour),
		URIs:         []*url.URL{u},
	}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &key.PublicKey, key)
	if err != nil {
		t.Fatalf("create cert: %v", err)
	}
	cert, err := x509.ParseCertificate(der)
	if err != nil {
		t.Fatalf("parse cert: %v", err)
	}
	id, err := spiffeIDFromCert(cert)
	if err != nil {
		t.Fatalf("spiffeIDFromCert: %v", err)
	}
	if id.String() != "spiffe://prod.local/rig/test-1" {
		t.Fatalf("id mismatch: %s", id)
	}

	// And a cert with no SPIFFE URI should error.
	tmpl.URIs = nil
	der2, _ := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &key.PublicKey, key)
	cert2, _ := x509.ParseCertificate(der2)
	if _, err := spiffeIDFromCert(cert2); err == nil {
		t.Fatalf("expected error for cert with no SPIFFE URI")
	}
	_ = pem.EncodeToMemory // touch import
}

// TestVerifyJWTSVID stands up a fake JWKS server, generates a JWT-SVID
// signed by the published key, and checks both the happy path and a
// wrong-audience rejection.  This is the load-bearing test for the
// /ws/agent SPIFFE path.
func TestVerifyJWTSVID(t *testing.T) {
	td := spiffeid.RequireTrustDomainFromString("prod.local")
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf("gen key: %v", err)
	}

	// JWKS document — one entry, ES256.
	jwks := jose.JSONWebKeySet{Keys: []jose.JSONWebKey{{
		Key:       &key.PublicKey,
		KeyID:     "test-1",
		Algorithm: "ES256",
		Use:       "sig",
	}}}
	mux := http.NewServeMux()
	mux.HandleFunc("/keys", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(jwks)
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	cfg := &config{
		spiffeTrustDomain: td.String(),
		spiffeJWKSURL:     srv.URL + "/keys",
		spiffeAudience:    "https://dist-server.test",
		publicURL:         "https://dist-server.test",
	}
	p, err := newSPIFFEProvider(context.Background(), cfg)
	if err != nil {
		t.Fatalf("newSPIFFEProvider: %v", err)
	}

	// Mint a JWT-SVID.
	signer, err := jose.NewSigner(
		jose.SigningKey{Algorithm: jose.ES256, Key: key},
		(&jose.SignerOptions{}).WithType("JWT").WithHeader("kid", "test-1"),
	)
	if err != nil {
		t.Fatalf("signer: %v", err)
	}
	mint := func(aud string, sub string) string {
		claims := jwt.Claims{
			Subject:  sub,
			Audience: jwt.Audience{aud},
			Expiry:   jwt.NewNumericDate(time.Now().Add(time.Hour)),
			IssuedAt: jwt.NewNumericDate(time.Now()),
		}
		s, err := jwt.Signed(signer).Claims(claims).Serialize()
		if err != nil {
			t.Fatalf("sign: %v", err)
		}
		return s
	}

	good := mint("https://dist-server.test", "spiffe://prod.local/rig/east-1")
	id, err := p.verifyJWTSVID(good)
	if err != nil {
		t.Fatalf("verify good: %v", err)
	}
	if id.String() != "spiffe://prod.local/rig/east-1" {
		t.Fatalf("id mismatch: %s", id)
	}

	// Wrong audience must reject.
	bad := mint("https://other.test", "spiffe://prod.local/rig/east-1")
	if _, err := p.verifyJWTSVID(bad); err == nil {
		t.Fatalf("expected audience rejection")
	}

	// SVID signed for a different trust domain must reject (the `sub`
	// claim's trust domain doesn't match the configured one).
	wrongTD := mint("https://dist-server.test", "spiffe://other.local/rig/east-1")
	if _, err := p.verifyJWTSVID(wrongTD); err == nil {
		t.Fatalf("expected trust-domain rejection")
	}
}

// TestSpiffeProviderDisabled — verifyJWTSVID on a nil provider must
// fail cleanly so the caller (ws.go) doesn't have to nil-check.
func TestSpiffeProviderDisabled(t *testing.T) {
	var p *spiffeProvider
	if _, err := p.verifyJWTSVID("anything"); err == nil {
		t.Fatalf("expected error from nil provider")
	}
}

// TestXFCCFieldParsing exercises the per-entry parser directly so a
// regression in the quote/escape handling shows up before it hits the
// SPIFFE ID extraction.
func TestXFCCFieldParsing(t *testing.T) {
	cases := []struct {
		entry string
		key   string
		want  string
	}{
		{"By=x;URI=spiffe://x/y", "URI", "spiffe://x/y"},
		{`By=x;Subject="CN=rig,OU=fleet";URI=spiffe://x/y`, "Subject", "CN=rig,OU=fleet"},
		{"By=x", "URI", ""},
		{`By="quoted ""value"`, "By", "quoted ", /* lexer-level; not load-bearing */},
	}
	for _, tc := range cases {
		got := xfccField(tc.entry, tc.key)
		// First three cases are precise.  The fourth is a tolerance
		// test: we don't crash on a malformed quote.
		if tc.entry == cases[3].entry {
			continue
		}
		if got != tc.want {
			t.Errorf("xfccField(%q, %q) = %q want %q", tc.entry, tc.key, got, tc.want)
		}
	}
	_ = strings.Builder{} // touch import
	_ = fmt.Sprint        // touch import
}
