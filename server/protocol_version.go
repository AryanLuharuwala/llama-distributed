package main

// Wire-protocol versioning between server and gpunet-node.
//
// The protocol-version handshake exists so the next time we need to make
// an incompatible wire change (new ACTV frame layout, mandatory E2E
// encryption, signed-only auth), we can refuse out-of-band rigs cleanly
// rather than hitting weird mid-session parse errors.
//
// Today every supported wire is version 1, so this code is effectively a
// no-op for live rigs — but the handshake gives us a single place to
// rev the protocol later without scattering version checks across
// signal.go, pp_route.go, dpp_route.go and ws.go.
//
// Rules:
//   - A rig that sends `protocol_version: 0` (or omits the field) is
//     treated as v1 — old rigs predate the field and we don't break
//     them.
//   - A rig that sends a version below serverProtocolMin is rejected
//     with a clear error so the user can upgrade.
//   - A rig that sends a version above serverProtocolMax is accepted at
//     serverProtocolMax (the rig knows newer features but the server
//     will only use the common subset).
//
// The chosen version is stored on agentConn.protocolVersion so handlers
// can branch on it without re-parsing the hello.

const (
	// Bump these together with any incompatible wire change.  Document
	// what changed in the commit so older rigs surface a useful error.
	serverProtocolMin = 1
	serverProtocolMax = 1
)

// negotiateProtocol returns (chosen, ok).  chosen is the effective wire
// version the connection will use; ok=false means the rig is too old to
// talk to us.
func negotiateProtocol(rigVersion int) (int, bool) {
	if rigVersion == 0 {
		// Legacy rig — treat as v1.
		return 1, serverProtocolMin <= 1
	}
	if rigVersion < serverProtocolMin {
		return 0, false
	}
	if rigVersion > serverProtocolMax {
		return serverProtocolMax, true
	}
	return rigVersion, true
}
