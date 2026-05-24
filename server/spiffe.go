package main

// SPIFFE workload identity (P6).
//
// Two flavours are supported, both optional, both keyed off the same
// trust domain configured via DIST_SPIFFE_TRUST_DOMAIN:
//
//   1. JWT-SVID over the /ws/agent hello frame.  The rig fetches a
//      short-lived JWT-SVID from its SPIRE workload API socket and
//      presents it in the hello as `spiffe_token`.  The server
//      validates the signature against a JWKS published by the SPIRE
//      server (DIST_SPIFFE_JWKS_URL) and trusts the SPIFFE URI in the
//      `sub` claim.
//
//   2. X.509-SVID via envoy XFCC.  Envoy terminates mTLS, validates
//      the client cert against the SPIRE trust bundle, then forwards
//      the verified cert metadata to dist-server in the
//      `x-forwarded-client-cert` header.  dist-server only honours
//      XFCC when the immediate TCP peer is in DIST_TRUSTED_PROXIES —
//      same trust check that gates X-Forwarded-For.
//
// Both flavours end in the same place: a SPIFFE ID string of shape
// `spiffe://<trust-domain>/<path>`, stored on the rig row.  Legacy
// agent_key auth keeps working alongside; SPIFFE is upgrade-in-place.

import (
	"context"
	"crypto/x509"
	"errors"
	"fmt"
	"log"
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/spiffe/go-spiffe/v2/bundle/jwtbundle"
	"github.com/spiffe/go-spiffe/v2/spiffeid"
	"github.com/spiffe/go-spiffe/v2/svid/jwtsvid"
)

// spiffeProvider is the live state for SPIFFE verification.  Nil when
// no trust domain is configured; cfg.spiffe is checked at every call
// site so all paths short-circuit cleanly without OIDC-style boot
// failures.
type spiffeProvider struct {
	td       spiffeid.TrustDomain
	audience string // expected `aud` claim on JWT-SVIDs
	jwksURL  string // SPIRE OIDC discovery / JWKS endpoint

	// bundle is the live JWT trust bundle (signing keys).  Refreshed
	// in the background via refreshLoop so a SPIRE key rotation
	// propagates without a server restart.
	mu     sync.RWMutex
	bundle *jwtbundle.Bundle
}

// newSPIFFEProvider wires up a JWT-SVID verifier against the
// configured trust domain.  We fetch the bundle once synchronously so
// boot fails loud if the JWKS URL is wrong; subsequent refreshes
// happen in the background and log but don't crash.
func newSPIFFEProvider(ctx context.Context, cfg *config) (*spiffeProvider, error) {
	td, err := spiffeid.TrustDomainFromString(cfg.spiffeTrustDomain)
	if err != nil {
		return nil, fmt.Errorf("trust domain: %w", err)
	}
	audience := cfg.spiffeAudience
	if audience == "" {
		audience = cfg.publicURL
	}
	p := &spiffeProvider{
		td:       td,
		audience: audience,
		jwksURL:  cfg.spiffeJWKSURL,
	}
	if cfg.spiffeJWKSURL != "" {
		if err := p.refreshBundle(ctx); err != nil {
			return nil, fmt.Errorf("initial bundle fetch: %w", err)
		}
	}
	return p, nil
}

// refreshBundle pulls the JWKS from the SPIRE OIDC discovery endpoint
// and swaps the in-memory bundle.  Idempotent and safe to call from
// a background loop.
func (p *spiffeProvider) refreshBundle(ctx context.Context) error {
	if p.jwksURL == "" {
		return nil
	}
	dctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	b, err := fetchJWTBundle(dctx, p.td, p.jwksURL)
	if err != nil {
		return err
	}
	p.mu.Lock()
	p.bundle = b
	p.mu.Unlock()
	return nil
}

// startRefreshLoop runs in the background once the server is up.  A
// 10-minute cadence is well below SPIRE's default 6-hour key rotation
// so we always have a fresh key cached when an old SVID rolls over.
func (p *spiffeProvider) startRefreshLoop(ctx context.Context) {
	if p == nil || p.jwksURL == "" {
		return
	}
	go func() {
		t := time.NewTicker(10 * time.Minute)
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-t.C:
				if err := p.refreshBundle(ctx); err != nil {
					// Stale-but-valid bundles keep working; log
					// once per refresh-failure so an operator can
					// notice a SPIRE outage without us page-flapping.
					// (No retries here — the next tick is the retry.)
					logSPIFFERefreshErr(err)
				}
			}
		}
	}()
}

// verifyJWTSVID parses + verifies a JWT-SVID and returns the SPIFFE ID
// that signed it.  Returns an error suitable for closing the WS hello.
func (p *spiffeProvider) verifyJWTSVID(token string) (spiffeid.ID, error) {
	if p == nil {
		return spiffeid.ID{}, errors.New("spiffe not configured")
	}
	p.mu.RLock()
	b := p.bundle
	p.mu.RUnlock()
	if b == nil {
		return spiffeid.ID{}, errors.New("no JWT bundle loaded — set DIST_SPIFFE_JWKS_URL")
	}
	svid, err := jwtsvid.ParseAndValidate(token, jwtBundleSource{b: b}, []string{p.audience})
	if err != nil {
		return spiffeid.ID{}, err
	}
	// Trust-domain check — jwtsvid validates signature + audience but
	// won't reject a token whose `sub` lives in a different trust
	// domain than the bundle.  Enforce here.
	if !svid.ID.MemberOf(p.td) {
		return spiffeid.ID{}, fmt.Errorf("svid trust domain %q != server %q",
			svid.ID.TrustDomain(), p.td)
	}
	return svid.ID, nil
}

// jwtBundleSource adapts a single *jwtbundle.Bundle to the
// jwtbundle.Source interface that jwtsvid.ParseAndValidate expects.
type jwtBundleSource struct{ b *jwtbundle.Bundle }

func (s jwtBundleSource) GetJWTBundleForTrustDomain(td spiffeid.TrustDomain) (*jwtbundle.Bundle, error) {
	if td == s.b.TrustDomain() {
		return s.b, nil
	}
	return nil, fmt.Errorf("no bundle for trust domain %q", td)
}

// ─── X.509-SVID via envoy XFCC ─────────────────────────────────────────

// spiffeIDFromXFCC extracts the SPIFFE ID from an `x-forwarded-client-cert`
// header populated by envoy after it validates the mTLS handshake.
//
// XFCC is a comma-separated list of cert chain entries; each entry is
// a semicolon-separated key=value list.  The keys we care about are
// `By` (envoy's SVID), `Hash`, `Subject`, `URI`, and `Cert`.  When
// envoy is in mTLS pass-through mode, the LEAF cert is the first
// entry and its URI key carries the SVID we want.
//
// We only call this from a code path that already verified the
// immediate TCP peer is in DIST_TRUSTED_PROXIES (see XFCCFromRequest
// below).  Without that guard XFCC is forgeable by any external
// client.
func spiffeIDFromXFCC(header string) (spiffeid.ID, error) {
	if header == "" {
		return spiffeid.ID{}, errors.New("no XFCC")
	}
	// Quote-aware split: commas inside a "..." value are part of a
	// field (Subject is the typical offender), not a chain separator.
	for _, entry := range splitOnUnquoted(header, ',') {
		uri := xfccField(entry, "URI")
		if uri == "" {
			continue
		}
		id, err := spiffeid.FromString(uri)
		if err == nil {
			return id, nil
		}
	}
	return spiffeid.ID{}, errors.New("XFCC had no parseable SPIFFE URI")
}

// splitOnUnquoted splits s on delim, ignoring delims that fall inside
// a double-quoted substring.  Quote characters themselves stay in the
// output — downstream parsers (xfccField → unquoteXFCC) handle them.
func splitOnUnquoted(s string, delim byte) []string {
	var out []string
	var buf strings.Builder
	inQuote := false
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c == '"' {
			inQuote = !inQuote
			buf.WriteByte(c)
			continue
		}
		if c == delim && !inQuote {
			out = append(out, buf.String())
			buf.Reset()
			continue
		}
		buf.WriteByte(c)
	}
	if buf.Len() > 0 {
		out = append(out, buf.String())
	}
	return out
}

// xfccField extracts one key=value field from one XFCC entry.  Values
// can be either bare tokens or double-quoted strings; envoy quotes any
// value with special chars (commas, semicolons, equals).  We strip the
// quotes uniformly so the caller doesn't have to.
func xfccField(entry, want string) string {
	for _, kv := range splitXFCC(entry) {
		eq := strings.IndexByte(kv, '=')
		if eq < 0 {
			continue
		}
		k := strings.TrimSpace(kv[:eq])
		if !strings.EqualFold(k, want) {
			continue
		}
		v := strings.TrimSpace(kv[eq+1:])
		if len(v) >= 2 && v[0] == '"' && v[len(v)-1] == '"' {
			// Unquote, undoing envoy's backslash escapes.
			if u, err := unquoteXFCC(v); err == nil {
				return u
			}
		}
		return v
	}
	return ""
}

// splitXFCC splits an XFCC entry on semicolons but respects double-quoted
// substrings so a quoted Subject= field with a semicolon doesn't get
// torn in half.
func splitXFCC(entry string) []string {
	return splitOnUnquoted(entry, ';')
}

func unquoteXFCC(s string) (string, error) {
	if len(s) < 2 || s[0] != '"' || s[len(s)-1] != '"' {
		return s, nil
	}
	s = s[1 : len(s)-1]
	// XFCC uses backslash-quote and backslash-backslash; everything
	// else passes through verbatim.
	if !strings.ContainsAny(s, `\`) {
		return s, nil
	}
	var b strings.Builder
	for i := 0; i < len(s); i++ {
		if s[i] == '\\' && i+1 < len(s) {
			b.WriteByte(s[i+1])
			i++
			continue
		}
		b.WriteByte(s[i])
	}
	out, err := url.QueryUnescape(b.String())
	if err != nil {
		return b.String(), nil
	}
	return out, nil
}

// spiffeIDFromCert is the X.509 path used in tests + future direct
// TLS termination work.  Given a leaf cert, return its first SPIFFE
// URI SAN.  RFC 4514-style — SPIFFE mandates exactly one URI SAN in
// each SVID.
func spiffeIDFromCert(cert *x509.Certificate) (spiffeid.ID, error) {
	for _, u := range cert.URIs {
		if u.Scheme == "spiffe" {
			return spiffeid.FromURI(u)
		}
	}
	return spiffeid.ID{}, errors.New("cert has no SPIFFE URI SAN")
}

// validSPIFFEForServer is the policy gate: an ID must live in the
// configured trust domain to be accepted.  Federated trust would
// require additional bundles; out of scope today.
func (p *spiffeProvider) validSPIFFEForServer(id spiffeid.ID) bool {
	if p == nil || id.IsZero() {
		return false
	}
	return id.MemberOf(p.td)
}

// ─── server-side helpers wired into ws.go ──────────────────────────────

// spiffeFromAgentHello returns the SPIFFE ID a rig presented during
// the WS hello.  It checks the XFCC header first (X.509-SVID via
// envoy) — only when the TCP peer is in DIST_TRUSTED_PROXIES — and
// falls back to the JWT-SVID in the hello payload.
//
// Returns (id, true) when a valid SVID was presented; (zero, false)
// otherwise.  The caller treats false as "no SPIFFE", not as an
// error: legacy rigs auth via agent_key and that path is unchanged.
func (s *server) spiffeFromAgentHello(r *http.Request, hello *agentHello) (spiffeid.ID, bool) {
	if s.cfg.spiffe == nil {
		return spiffeid.ID{}, false
	}
	// X.509 path: envoy XFCC, only honoured behind a trusted proxy.
	if s.peerIsTrustedProxy(r) {
		if xfcc := r.Header.Get("x-forwarded-client-cert"); xfcc != "" {
			if id, err := spiffeIDFromXFCC(xfcc); err == nil &&
				s.cfg.spiffe.validSPIFFEForServer(id) {
				return id, true
			}
		}
	}
	// JWT path: token in the hello frame.
	if hello.SPIFFEToken != "" {
		if id, err := s.cfg.spiffe.verifyJWTSVID(hello.SPIFFEToken); err == nil &&
			s.cfg.spiffe.validSPIFFEForServer(id) {
			return id, true
		}
	}
	return spiffeid.ID{}, false
}

// peerIsTrustedProxy is true iff the immediate TCP peer (r.RemoteAddr)
// is in the DIST_TRUSTED_PROXIES set.  Reuses the P8 parser so XFCC and
// X-Forwarded-For share one trust boundary.
func (s *server) peerIsTrustedProxy(r *http.Request) bool {
	if s.cfg.trustedProxies.empty() {
		return false
	}
	host := remoteAddrHost(r)
	ip := net.ParseIP(host)
	return s.cfg.trustedProxies.contains(ip)
}

// fetchJWTBundle pulls a JWKS document from the SPIRE OIDC discovery
// endpoint and parses it into a jwtbundle.Bundle.  SPIRE's OIDC
// discovery provider serves the bundle as a standard JWKS so we can
// use jwtbundle.Read.
func fetchJWTBundle(ctx context.Context, td spiffeid.TrustDomain, jwksURL string) (*jwtbundle.Bundle, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", jwksURL, nil)
	if err != nil {
		return nil, err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return nil, fmt.Errorf("jwks fetch: HTTP %d", resp.StatusCode)
	}
	return jwtbundle.Read(td, resp.Body)
}

// logSPIFFERefreshErr logs a background JWKS refresh failure.  Rate is
// driven by the refresh ticker (10 min), so a stuck SPIRE produces ~6
// log lines per hour — visible enough to alert without flooding.
func logSPIFFERefreshErr(err error) {
	log.Printf("spiffe: background JWKS refresh failed: %v (using cached bundle)", err)
}
