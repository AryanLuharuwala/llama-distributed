package main

import (
	"strings"
	"testing"
)

// Each branch of validateAndClampStatus — fuzz-style "shove garbage in,
// expect sane output" coverage so the hostile-by-default contract holds.
func TestValidateAndClampStatus_Caps(t *testing.T) {
	st := &agentStatus{
		NGPUs:      9999,
		VRAMTotal:  int64(1) << 50,
		VRAMFree:   int64(1) << 50,
		TokensPS:   1e9,
		UptimeSec:  int64(100) * 365 * 24 * 3600,
		RolesHeld:  make([]string, 1000),
		ModelsHeld: []string{strings.Repeat("x", 1024)},
		GPUModel:   strings.Repeat("y", 1024),
		LastError:  strings.Repeat("z", 4096),
		NATType:    "exotic",
		CoturnPort: 22,
		PublicIP:   "10.0.0.5",
	}
	validateAndClampStatus(st, "1.2.3.4")

	if st.NGPUs != 0 {
		t.Errorf("NGPUs should reset; got %d", st.NGPUs)
	}
	if st.VRAMTotal != 0 {
		t.Errorf("VRAMTotal should reset; got %d", st.VRAMTotal)
	}
	if st.VRAMFree != 0 {
		t.Errorf("VRAMFree should reset when > VRAMTotal; got %d", st.VRAMFree)
	}
	if st.TokensPS != 0 {
		t.Errorf("TokensPS should reset; got %f", st.TokensPS)
	}
	if st.UptimeSec != 0 {
		t.Errorf("UptimeSec should reset; got %d", st.UptimeSec)
	}
	if len(st.RolesHeld) != maxRolesHeld {
		t.Errorf("RolesHeld should clamp to %d; got %d", maxRolesHeld, len(st.RolesHeld))
	}
	if len(st.ModelsHeld[0]) != maxModelNameLen {
		t.Errorf("ModelsHeld[0] should clamp; got %d", len(st.ModelsHeld[0]))
	}
	if len(st.GPUModel) != 128 {
		t.Errorf("GPUModel should clamp to 128; got %d", len(st.GPUModel))
	}
	if len(st.LastError) != 512 {
		t.Errorf("LastError should clamp to 512; got %d", len(st.LastError))
	}
	if st.NATType != "unknown" {
		t.Errorf("NATType 'exotic' should normalize to 'unknown'; got %q", st.NATType)
	}
	if st.CoturnPort != 0 {
		t.Errorf("CoturnPort=22 (privileged) should reset; got %d", st.CoturnPort)
	}
	if st.PublicIP != "" {
		t.Errorf("PublicIP RFC1918 with non-matching source should reset; got %q", st.PublicIP)
	}
}

func TestValidateAndClampStatus_NegativesReset(t *testing.T) {
	st := &agentStatus{
		NGPUs:     -1,
		VRAMTotal: -1,
		VRAMFree:  -1,
		TokensPS:  -1,
		UptimeSec: -1,
	}
	validateAndClampStatus(st, "1.2.3.4")
	if st.NGPUs != 0 || st.VRAMTotal != 0 || st.VRAMFree != 0 || st.TokensPS != 0 || st.UptimeSec != 0 {
		t.Errorf("negative inputs should reset to 0; got %+v", st)
	}
}

func TestValidateAndClampStatus_NilSafe(t *testing.T) {
	// Must not panic.
	validateAndClampStatus(nil, "")
}

func TestValidateAndClampStatus_AllowedValues(t *testing.T) {
	st := &agentStatus{
		NGPUs:        2,
		VRAMTotal:    int64(8) << 30,
		VRAMFree:     int64(4) << 30,
		TokensPS:     45.0,
		UptimeSec:    3600,
		NATType:      "cone",
		RelayCapable: true,
		CoturnPort:   3478,
		PublicIP:     "8.8.8.8",
	}
	validateAndClampStatus(st, "9.9.9.9")
	if st.NGPUs != 2 || st.VRAMTotal == 0 || st.VRAMFree == 0 || st.TokensPS != 45.0 {
		t.Errorf("legitimate values should pass through; got %+v", st)
	}
	if st.NATType != "cone" {
		t.Errorf("NATType 'cone' should stay; got %q", st.NATType)
	}
	if st.CoturnPort != 3478 {
		t.Errorf("CoturnPort 3478 should stay; got %d", st.CoturnPort)
	}
	if st.PublicIP != "8.8.8.8" {
		t.Errorf("public IP 8.8.8.8 should stay; got %q", st.PublicIP)
	}
}

func TestValidateAndClampStatus_LANSameIP(t *testing.T) {
	// Private-LAN deploy: claimed == remoteIP, both RFC1918 → allowed.
	st := &agentStatus{PublicIP: "192.168.1.50"}
	validateAndClampStatus(st, "192.168.1.50")
	if st.PublicIP != "192.168.1.50" {
		t.Errorf("LAN same-IP should be allowed; got %q", st.PublicIP)
	}
}

func TestIsAllowedPublicIP(t *testing.T) {
	cases := []struct {
		name             string
		claimed, remote  string
		want             bool
	}{
		{"public IPv4", "8.8.8.8", "1.2.3.4", true},
		{"same private IP (LAN)", "192.168.1.10", "192.168.1.10", true},
		{"RFC1918 ≠ source", "10.0.0.1", "1.2.3.4", false},
		{"CGNAT ≠ source", "100.64.0.1", "1.2.3.4", false},
		{"loopback", "127.0.0.1", "127.0.0.1", false},
		{"multicast", "224.0.0.1", "1.2.3.4", false},
		{"link-local", "169.254.1.1", "1.2.3.4", false},
		{"unspecified", "0.0.0.0", "1.2.3.4", false},
		{"empty", "", "1.2.3.4", false},
		{"junk", "not-an-ip", "1.2.3.4", false},
		{"oversize", strings.Repeat("a", 100), "1.2.3.4", false},
		{"public IPv6", "2001:4860:4860::8888", "1.2.3.4", true},
		{"ULA IPv6", "fc00::1", "1.2.3.4", false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := isAllowedPublicIP(c.claimed, c.remote)
			if got != c.want {
				t.Errorf("isAllowedPublicIP(%q, %q) = %v, want %v",
					c.claimed, c.remote, got, c.want)
			}
		})
	}
}

func TestClampRelayBytes(t *testing.T) {
	cases := []struct {
		in       int64
		out      int64
		clamped  bool
	}{
		{0, 0, false},
		{1024, 1024, false},
		{maxRelayBytesPerSession, maxRelayBytesPerSession, false},
		{maxRelayBytesPerSession + 1, maxRelayBytesPerSession, true},
		{-1, 0, true},
		{-1 << 62, 0, true},
		{1 << 62, maxRelayBytesPerSession, true},
	}
	for _, c := range cases {
		out, clamped := clampRelayBytes(c.in)
		if out != c.out || clamped != c.clamped {
			t.Errorf("clampRelayBytes(%d) = (%d, %v); want (%d, %v)",
				c.in, out, clamped, c.out, c.clamped)
		}
	}
}

func TestValidateAndClampStatus_CoturnPortRanges(t *testing.T) {
	cases := []struct {
		in, want int
	}{
		{0, 0},          // unset stays unset
		{22, 0},         // privileged → reset
		{1023, 0},       // just below min
		{1024, 1024},    // boundary
		{3478, 3478},    // typical TURN
		{65535, 65535},  // max
		{65536, 0},      // out of range
		{-1, 0},
	}
	for _, c := range cases {
		st := &agentStatus{CoturnPort: c.in}
		validateAndClampStatus(st, "1.2.3.4")
		if st.CoturnPort != c.want {
			t.Errorf("CoturnPort %d → %d; want %d", c.in, st.CoturnPort, c.want)
		}
	}
}
