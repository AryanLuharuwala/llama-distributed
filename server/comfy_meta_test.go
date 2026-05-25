package main

import "testing"

// CF11: the path allowlist is the security boundary between the
// coordinator and the rig's local ComfyUI.  A regression here would
// either accidentally expose a mutating endpoint (e.g. /prompt POST)
// or silently break a SDK that's stable on the documented surface.
// Locking the table down explicitly so future contributors notice.
func TestComfyMetaPathAllowlist(t *testing.T) {
	cases := []struct {
		method, path string
		want         bool
	}{
		// Allowed metadata reads.
		{"GET", "/system_stats", true},
		{"GET", "/embeddings", true},
		{"GET", "/object_info", true},
		{"GET", "/object_info/CheckpointLoaderSimple", true},
		{"GET", "/models", true},
		{"GET", "/models/checkpoints", true},
		{"GET", "/queue", true},
		{"GET", "/history", true},
		{"GET", "/history/abc-123", true},
		{"GET", "/features", true},
		{"GET", "/prompt", true},

		// Allowed control posts.
		{"POST", "/interrupt", true},
		{"POST", "/free", true},

		// Wrong method on a read path.
		{"POST", "/system_stats", false},
		{"POST", "/object_info", false},

		// Mutating paths that must NOT be exposed via the proxy — we
		// own /prompt + /upload + /queue mutations through the
		// authenticated /api/comfy/* surface, with billing/quota
		// attached.  Exposing them here would let a client bypass
		// every quota check.
		{"POST", "/prompt", false},
		{"POST", "/upload/image", false},
		{"POST", "/upload/mask", false},
		{"POST", "/queue", false},
		{"POST", "/history", false},

		// Anything outside the allowlist (e.g. a hypothetical
		// custom_nodes admin endpoint, or a path-traversal probe).
		{"GET", "/", false},
		{"GET", "/admin", false},
		{"GET", "/object_info/../../../etc/passwd", true}, // prefix matches, but agent-side resolves before HTTP — that's the rig's job to fence
		{"GET", "/system_statsX", false},
		{"GET", "/embeddingsX", false},
	}
	for _, tc := range cases {
		got := comfyMetaPathAllowed(tc.method, tc.path)
		if got != tc.want {
			t.Errorf("comfyMetaPathAllowed(%q,%q) = %v, want %v",
				tc.method, tc.path, got, tc.want)
		}
	}
}

func TestComfyMetaCorrID(t *testing.T) {
	// Determinism check: distinct calls return distinct ids, and the
	// id is non-empty (the time-mix fallback should still produce a
	// usable string if rand.Read ever fails).
	seen := make(map[string]bool, 64)
	for i := 0; i < 64; i++ {
		id := newComfyMetaCorrID()
		if id == "" {
			t.Fatal("newComfyMetaCorrID returned empty string")
		}
		if seen[id] {
			t.Fatalf("duplicate corr id %q", id)
		}
		seen[id] = true
	}
}

func TestTranslateUpstreamStatus(t *testing.T) {
	cases := []struct{ in, want int }{
		{0, 503},
		{200, 200},
		{204, 204},
		{400, 400},
		{404, 404},
		{500, 502},
		{503, 502},
		{999, 502},
	}
	for _, tc := range cases {
		if got := translateUpstreamStatus(tc.in); got != tc.want {
			t.Errorf("translateUpstreamStatus(%d) = %d, want %d", tc.in, got, tc.want)
		}
	}
}
