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

// isPrivateOrCGN returns true for any address that can't be the source
// of a STUN srflx response from the public Internet.  Rigs claiming
// these ranges are either confused or actively trying to redirect
// TURN traffic at a victim.
//
// Covers:
//   IPv4: RFC1918 (10/8, 172.16/12, 192.168/16), loopback (127/8),
//         CGNAT (100.64/10), link-local (169.254/16 — also caught by
//         net.IP.IsLinkLocalUnicast at the call site), benchmarking
//         (198.18/15), documentation (192.0.2/24, 198.51.100/24,
//         203.0.113/24), 0/8, 240/4 future-reserved, and broadcast.
//   IPv6: ULA (fc00::/7), 6to4 (2002::/16 — the encapsulated v4
//         address would let a rig tunnel through to private space),
//         Teredo (2001::/32), discard (100::/64), documentation
//         (2001:db8::/32), 6bone (deprecated, 3ffe::/16).
func isPrivateOrCGN(ip net.IP) bool {
	if ip4 := ip.To4(); ip4 != nil {
		switch {
		case ip4[0] == 0:
			return true
		case ip4[0] == 10:
			return true
		case ip4[0] == 127:
			return true
		case ip4[0] == 169 && ip4[1] == 254:
			return true
		case ip4[0] == 172 && ip4[1] >= 16 && ip4[1] <= 31:
			return true
		case ip4[0] == 192 && ip4[1] == 0 && ip4[2] == 0:
			return true
		case ip4[0] == 192 && ip4[1] == 0 && ip4[2] == 2:
			return true
		case ip4[0] == 192 && ip4[1] == 168:
			return true
		case ip4[0] == 198 && (ip4[1] == 18 || ip4[1] == 19):
			return true
		case ip4[0] == 198 && ip4[1] == 51 && ip4[2] == 100:
			return true
		case ip4[0] == 203 && ip4[1] == 0 && ip4[2] == 113:
			return true
		case ip4[0] == 100 && ip4[1] >= 64 && ip4[1] <= 127:
			return true
		case ip4[0] >= 224: // 224/4 multicast (caught upstream) + 240/4 reserved + 255.255.255.255
			return true
		}
		return false
	}
	if len(ip) != 16 {
		return false
	}
	// ULA: fc00::/7 — top 7 bits are 1111110.
	if (ip[0] & 0xfe) == 0xfc {
		return true
	}
	// 6to4: 2002::/16 — embeds an IPv4 in the next 32 bits.  Lets a
	// rig tunnel a claim through to private IPv4 space.  Reject the
	// whole block; legitimate public-IPv6 rigs use native 2000::/3.
	if ip[0] == 0x20 && ip[1] == 0x02 {
		return true
	}
	// Teredo: 2001::/32.  Same reasoning — encapsulated v4.
	if ip[0] == 0x20 && ip[1] == 0x01 && ip[2] == 0x00 && ip[3] == 0x00 {
		return true
	}
	// Discard prefix: 100::/64.
	if ip[0] == 0x01 && ip[1] == 0x00 && ip[2] == 0x00 && ip[3] == 0x00 &&
		ip[4] == 0x00 && ip[5] == 0x00 && ip[6] == 0x00 && ip[7] == 0x00 {
		return true
	}
	// IPv6 documentation: 2001:db8::/32.
	if ip[0] == 0x20 && ip[1] == 0x01 && ip[2] == 0x0d && ip[3] == 0xb8 {
		return true
	}
	// Deprecated 6bone: 3ffe::/16.
	if ip[0] == 0x3f && ip[1] == 0xfe {
		return true
	}
	// IPv4-mapped (::ffff:0:0/96).  Re-check as IPv4 — defensive,
	// net.IP.To4() above should already have caught this.
	if ip[0] == 0 && ip[1] == 0 && ip[2] == 0 && ip[3] == 0 &&
		ip[4] == 0 && ip[5] == 0 && ip[6] == 0 && ip[7] == 0 &&
		ip[8] == 0 && ip[9] == 0 && ip[10] == 0xff && ip[11] == 0xff {
		return isPrivateOrCGN(net.IPv4(ip[12], ip[13], ip[14], ip[15]))
	}
	return false
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

// maxRelayBitsPerSec is the upper bound on plausible throughput per
// direction on a relay link.  A residential 10 GbE uplink is unrealistic
// for nearly every operator but we don't want to penalize that one shop
// with a dark-fiber drop.  Anything beyond is a rig trying to inflate
// its tiebreaker score by reporting bytes that physically could not
// have transited within the session window.
//
// At 10 Gbps, a 60-second session can plausibly move ~75 GB — well
// above the per-session 64 GiB hard cap, so for short sessions the
// per-session cap still binds.  This second cap kicks in for sessions
// that release a few seconds after assign with a giant byte report.
const maxRelayBitsPerSec = int64(10) << 30 // 10 Gbps

// clampRelayBytesByElapsed bounds the reported byte count against the
// physical wall-clock window the session occupied.  A relay rig that
// claims 64 GiB across a 1-second session is lying — the link couldn't
// carry it.  We grant a 1-second grace so freshly-assigned sessions
// that release immediately don't divide-by-zero.
//
// Returns the clamped value plus whether clamping happened so the
// caller can log.
func clampRelayBytesByElapsed(reported, startedAt, nowSec int64) (int64, bool) {
	if reported <= 0 {
		// Pass through (clampRelayBytes already normalized negatives).
		if reported < 0 {
			return 0, true
		}
		return 0, false
	}
	elapsed := nowSec - startedAt
	if elapsed < 1 {
		elapsed = 1
	}
	// max bytes = elapsed_sec * (max_bps / 8)
	limit := elapsed * (maxRelayBitsPerSec / 8)
	if limit < 0 || reported > limit {
		return limit, true
	}
	return reported, false
}
