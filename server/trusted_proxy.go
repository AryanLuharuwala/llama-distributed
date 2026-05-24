package main

// Trusted-proxy XFF resolver (prod branch).
//
// On the main branch, clientIP() and remoteIPForRateLimit() trust the
// left-most entry of X-Forwarded-For verbatim.  That is correct exactly
// when the server sits behind a proxy that strips any incoming XFF and
// prepends its own — and dangerously wrong otherwise: any client can set
// XFF: 1.2.3.4 to attribute traffic (rate-limit buckets, audit logs,
// device-code ratchet) to whatever IP they choose.
//
// This file installs a trust-aware resolver.  Operators configure the
// set of CIDRs that may be legitimately speaking on the upstream side of
// our front-proxy (typically the proxy's own pod IP / the local docker
// network).  The resolver then walks XFF right-to-left, skipping hops in
// the trust set, returning the first untrusted address — the actual
// client.  If the trust set is empty, XFF is ignored entirely and we
// fall back to r.RemoteAddr.
//
// Configuration:
//   DIST_TRUSTED_PROXIES="127.0.0.1/32,10.0.0.0/8"
//
// Pair this with envoy / nginx configured to: (a) strip any inbound XFF
// from clients on its public listener, (b) append its own
// X-Forwarded-For with the real client, (c) listen on an upstream IP
// that's in DIST_TRUSTED_PROXIES.

import (
	"net"
	"net/http"
	"strings"
)

// trustedProxySet is parsed once at boot and read concurrently.  Empty
// means "don't trust XFF at all".
type trustedProxySet struct {
	nets []*net.IPNet
}

// parseTrustedProxies accepts a comma-separated list of CIDRs and bare
// IPs.  Invalid entries are logged and skipped — we never want a typo
// in this env var to silently flip the server into "trust everything"
// or "trust nothing" mode without telling the operator.
func parseTrustedProxies(spec string) (*trustedProxySet, []string) {
	set := &trustedProxySet{}
	var bad []string
	if spec == "" {
		return set, nil
	}
	for _, raw := range strings.Split(spec, ",") {
		s := strings.TrimSpace(raw)
		if s == "" {
			continue
		}
		// Bare IP → /32 or /128.  Saves operators from having to write
		// the mask when they just want to whitelist a single proxy IP.
		if !strings.Contains(s, "/") {
			if ip := net.ParseIP(s); ip != nil {
				if ip.To4() != nil {
					s += "/32"
				} else {
					s += "/128"
				}
			}
		}
		_, n, err := net.ParseCIDR(s)
		if err != nil {
			bad = append(bad, raw)
			continue
		}
		set.nets = append(set.nets, n)
	}
	return set, bad
}

// contains reports whether ip is within any of the trusted CIDRs.
func (t *trustedProxySet) contains(ip net.IP) bool {
	if t == nil || len(t.nets) == 0 || ip == nil {
		return false
	}
	for _, n := range t.nets {
		if n.Contains(ip) {
			return true
		}
	}
	return false
}

// empty is true when no trust entries are configured.  Callers use this
// to short-circuit XFF parsing entirely.
func (t *trustedProxySet) empty() bool {
	return t == nil || len(t.nets) == 0
}

// trustedClientIP returns the originating client IP for an HTTP request,
// walking XFF right-to-left and stopping at the first untrusted hop.
//
// Rationale for right-to-left:
//   - r.RemoteAddr is the only IP we know for certain (the TCP peer).
//   - Each proxy in the chain appends the IP it saw on its inbound
//     socket to XFF.  So the right-most XFF entry is the IP seen by the
//     final proxy (the one talking to us); the left-most is whatever
//     the first proxy was told — which is exactly the part an attacker
//     can spoof if any proxy in the chain accepts client-supplied XFF.
//   - Walking right-to-left and skipping trusted hops gives us the
//     left-most IP that was supplied by a *trusted* proxy.  Past that
//     point we don't know if the value is genuine, so we stop.
//
// trustSet.empty() ⇒ XFF is ignored; the function returns the IP from
// r.RemoteAddr (the only thing the kernel saw).
func trustedClientIP(r *http.Request, trust *trustedProxySet) string {
	remote := remoteAddrHost(r)
	if trust.empty() {
		return remote
	}
	// Start with the TCP peer.  If it's not itself trusted, the only IP
	// we can attribute traffic to is the peer — we don't trust whatever
	// XFF it sent.
	peerIP := net.ParseIP(remote)
	if peerIP == nil || !trust.contains(peerIP) {
		return remote
	}
	// Peer is a trusted proxy.  Walk XFF right-to-left until we find an
	// IP outside the trust set; that's the client.  If every entry is
	// trusted (uncommon — only happens when an internal service hops
	// through several of our own proxies), we return the left-most
	// entry, which is the closest-to-client we can identify.
	xff := r.Header.Get("X-Forwarded-For")
	if xff == "" {
		// No XFF but the peer is a trusted proxy — best we can do is
		// surface the peer.  This shouldn't happen in practice; a
		// trusted proxy that doesn't set XFF is misconfigured.
		return remote
	}
	parts := strings.Split(xff, ",")
	for i := len(parts) - 1; i >= 0; i-- {
		hop := strings.TrimSpace(parts[i])
		if hop == "" {
			continue
		}
		ip := net.ParseIP(hop)
		if ip == nil {
			// Malformed entry — treat as the boundary.  Don't walk
			// past garbage to find a "good" IP further left, because
			// at that point we have no chain integrity.
			return hop
		}
		if !trust.contains(ip) {
			return hop
		}
	}
	// Every XFF entry was a trusted proxy.  Return the left-most as the
	// closest-to-client identity we have.
	return strings.TrimSpace(parts[0])
}

// remoteAddrHost extracts the host portion of r.RemoteAddr.  Returns
// the full RemoteAddr if it doesn't split (e.g. unix socket).
func remoteAddrHost(r *http.Request) string {
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return host
}
