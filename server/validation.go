package main

// Input validation for agent-supplied fields.
//
// Anything an agent puts on the wire is hostile-by-default — a misbehaving
// or malicious rig could claim impossible specs, spoof its public IP to
// redirect TURN traffic, or stuff absurd byte counts into reputation
// updates.  These helpers exist so the parsing code can clamp values to
// sane ranges in one place and the rest of the codebase can trust what
// it reads from agentConn.live.

import (
	"net"
	"strings"
)

// Hard caps for agent-reported telemetry.  Sized generously so they never
// trip a legitimate rig — these are sanity bounds, not policy.
const (
	maxNGPUs        = 64
	maxVRAMBytes    = int64(2) << 40 // 2 TiB
	maxTokensPS     = 10000.0        // 10k tok/s is absurd; gpt-3.5 peaks ~150
	maxUptimeSec    = int64(10) * 365 * 24 * 3600
	maxRolesHeld    = 32
	maxModelsHeld   = 256
	maxModelNameLen = 256

	// Ports below 1024 are privileged and we explicitly forbid them to
	// stop a rig from claiming port=22/25/53/80/443 etc and convincing
	// the server to hand clients TURN URLs pointing at those services.
	minTURNPort = 1024
	maxTURNPort = 65535

	// Per-session relay bytes — capping each side at 64 GiB.  A single
	// inference session pushing more than that is implausible and probably
	// a rig trying to inflate its tiebreaker score.
	maxRelayBytesPerSession = int64(64) << 30

	// Hard cap on parallel inference slots a rig can advertise.  llama.cpp
	// `--parallel` realistically tops out around 8–16 on consumer GPUs; we
	// allow a generous 32 so future tensor-parallel rigs aren't artificially
	// capped, and clamp anything beyond to keep the dispatch map bounded.
	maxAdvertisedConcurrent = 32
)

// validateAndClampStatus mutates `st` in place to enforce the caps above.
// remoteIP is the actual WS source — used for public_ip validation.  We
// don't return an error: a misbehaving rig that overflows a field gets
// its value reset to zero/empty, and the rest of the flow proceeds with
// safe defaults.  Logging happens at the call site if needed.
func validateAndClampStatus(st *agentStatus, remoteIP string) {
	if st == nil {
		return
	}
	if st.NGPUs < 0 || st.NGPUs > maxNGPUs {
		st.NGPUs = 0
	}
	if st.VRAMTotal < 0 || st.VRAMTotal > maxVRAMBytes {
		st.VRAMTotal = 0
	}
	if st.VRAMFree < 0 || st.VRAMFree > st.VRAMTotal {
		st.VRAMFree = 0
	}
	if st.TokensPS < 0 || st.TokensPS > maxTokensPS {
		st.TokensPS = 0
	}
	if st.UptimeSec < 0 || st.UptimeSec > maxUptimeSec {
		st.UptimeSec = 0
	}
	if len(st.RolesHeld) > maxRolesHeld {
		st.RolesHeld = st.RolesHeld[:maxRolesHeld]
	}
	if len(st.ModelsHeld) > maxModelsHeld {
		st.ModelsHeld = st.ModelsHeld[:maxModelsHeld]
	}
	for i, m := range st.ModelsHeld {
		if len(m) > maxModelNameLen {
			st.ModelsHeld[i] = m[:maxModelNameLen]
		}
	}
	// GPUModel is a free-form string but it's surfaced on the swarm
	// dashboard — clamp so a rig can't cram a 10 MiB string into the UI.
	if len(st.GPUModel) > 128 {
		st.GPUModel = st.GPUModel[:128]
	}
	if len(st.LastError) > 512 {
		st.LastError = st.LastError[:512]
	}
	switch st.NATType {
	case "", "open", "cone", "symmetric", "blocked", "unknown":
		// allowed
	default:
		st.NATType = "unknown"
	}
	if st.CoturnPort != 0 && (st.CoturnPort < minTURNPort || st.CoturnPort > maxTURNPort) {
		// Reject privileged ports and out-of-range nonsense.  Setting to
		// zero disables the TURN sidecar advertisement for this rig
		// until it sends a corrected value.
		st.CoturnPort = 0
	}
	if st.PublicIP != "" && !isAllowedPublicIP(st.PublicIP, remoteIP) {
		st.PublicIP = ""
	}
	if st.MaxConcurrent < 0 || st.MaxConcurrent > maxAdvertisedConcurrent {
		// Negative / huge values would inflate routing decisions and let
		// a malicious rig hog the request queue.  Reset to "legacy
		// single-slot" (0 → server treats as 1) rather than dropping the
		// status frame entirely.
		st.MaxConcurrent = 0
	}
}

// isAllowedPublicIP returns true iff the rig's claimed public IP looks
// real and is consistent with where its WS connection arrived from.
//
// Two acceptable shapes:
//   1. claimed is a public IPv4 (not RFC1918, not loopback, not link-local,
//      not multicast).  We don't require it to match remoteIP exactly —
//      CGNAT and asymmetric routing can both make the STUN-discovered IP
//      legitimately differ from the TCP source — but it MUST be public.
//   2. claimed equals remoteIP exactly.  This covers private-LAN deploys
//      where everyone is on RFC1918 and the rig genuinely has no public IP
//      yet still wants to advertise its LAN address for other rigs in the
//      same subnet.
//
// Anything else — claiming a public IP that doesn't match the source and
// isn't a credible STUN srflx — gets rejected.  Mitigates the "rig spoofs
// its public IP to redirect TURN traffic at a victim" attack.
func isAllowedPublicIP(claimed, remoteIP string) bool {
	if claimed == "" {
		return false
	}
	// Reject obvious garbage early.
	if len(claimed) > 45 { // longest legit IPv6 string
		return false
	}
	ip := net.ParseIP(strings.TrimSpace(claimed))
	if ip == nil {
		return false
	}
	if ip.IsLoopback() || ip.IsMulticast() || ip.IsUnspecified() || ip.IsLinkLocalUnicast() {
		return false
	}
	// Same-IP fallback for LAN/dev deployments.
	if claimed == remoteIP {
		return true
	}
	// Otherwise must be a public-routable IP.
	return !isPrivateOrCGN(ip)
}

// isPrivateOrCGN returns true for RFC1918, CGNAT (100.64.0.0/10), and
// loopback ranges.  These are all addresses that can't be the source of
// a STUN srflx response in practice; rigs claiming them are either
// confused or hostile.
func isPrivateOrCGN(ip net.IP) bool {
	if ip4 := ip.To4(); ip4 != nil {
		switch {
		case ip4[0] == 10:
			return true
		case ip4[0] == 127:
			return true
		case ip4[0] == 172 && ip4[1] >= 16 && ip4[1] <= 31:
			return true
		case ip4[0] == 192 && ip4[1] == 168:
			return true
		case ip4[0] == 100 && ip4[1] >= 64 && ip4[1] <= 127: // CGNAT
			return true
		}
		return false
	}
	// IPv6: fc00::/7 is ULA, ::1 loopback (already caught), fe80::/10
	// link-local (already caught).
	return len(ip) == 16 && (ip[0]&0xfe) == 0xfc
}

// clampRelayBytes bounds the byte count a rig can claim per session.
// Returns the clamped value plus whether clamping actually happened so
// the caller can log suspicious activity.
func clampRelayBytes(v int64) (int64, bool) {
	if v < 0 {
		return 0, true
	}
	if v > maxRelayBytesPerSession {
		return maxRelayBytesPerSession, true
	}
	return v, false
}
