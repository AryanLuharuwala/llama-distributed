package main

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// TestHTTPToWS pins surd's http→ws rewrite (the simpler stringy form
// the bootstrap uses, distinct from dist-turn's url.Parse-based one).
// Anything non-http(s) passes through — surd's caller already trims
// trailing slashes, so we only verify scheme rewriting.
func TestHTTPToWS(t *testing.T) {
	cases := []struct{ in, want string }{
		{"http://pool.example.com", "ws://pool.example.com"},
		{"https://pool.example.com", "wss://pool.example.com"},
		{"https://pool.example.com/api", "wss://pool.example.com/api"},
		{"already://something", "already://something"},
		{"", ""},
	}
	for _, c := range cases {
		if got := httpToWS(c.in); got != c.want {
			t.Errorf("httpToWS(%q) = %q want %q", c.in, got, c.want)
		}
	}
}

// TestCredsRoundtrip exercises save → load through a temp dir,
// catching JSON drift or permission regressions.  The env override
// (SURD_CREDS) makes this hermetic so we don't trample a real
// developer login.
func TestCredsRoundtrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "credentials.json")
	t.Setenv("SURD_CREDS", path)

	want := &credentials{
		Server:   "https://pool.example.com",
		AgentID:  "agent-xyz",
		AgentKey: "key-abc",
	}
	if err := saveCreds(want); err != nil {
		t.Fatalf("saveCreds: %v", err)
	}

	// File must exist at the override path, not the default.
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("creds file missing at override path: %v", err)
	}

	// Mode 0600 — sensitive token, no shared-user readability.  Skip
	// on Windows where Unix permission bits don't apply.
	if runtime.GOOS != "windows" {
		st, _ := os.Stat(path)
		if mode := st.Mode().Perm(); mode != 0o600 {
			t.Errorf("creds perm = %o, want 0600", mode)
		}
	}

	got, err := loadCreds()
	if err != nil {
		t.Fatalf("loadCreds: %v", err)
	}
	if *got != *want {
		t.Errorf("roundtrip mismatch:\n got=%+v\nwant=%+v", *got, *want)
	}
}

// TestLoadCredsMissingFile — loadCreds on an absent path should
// surface a not-exist error rather than a parse error, so callers
// can distinguish "first run" from "corrupted creds".
func TestLoadCredsMissingFile(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("SURD_CREDS", filepath.Join(dir, "no-such-creds.json"))
	if _, err := loadCreds(); err == nil {
		t.Fatal("expected error for missing creds, got nil")
	} else if !os.IsNotExist(err) {
		t.Errorf("expected ErrNotExist, got %v", err)
	}
}

// TestLoadCredsCorrupted — bad JSON must error with a path-tagged
// message so the user can find the file to delete it.
func TestLoadCredsCorrupted(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "creds.json")
	t.Setenv("SURD_CREDS", path)
	if err := os.WriteFile(path, []byte("{not json"), 0o600); err != nil {
		t.Fatal(err)
	}
	_, err := loadCreds()
	if err == nil {
		t.Fatal("expected parse error, got nil")
	}
	if !strings.Contains(err.Error(), path) {
		t.Errorf("error should include creds path so user can find it; got: %v", err)
	}
}

// TestControlPIDPath — the connect.pid is always a sibling of
// credentials.json so `surd stop` and the widget look in the same
// place.  Asserting the sibling property locks the contract.
func TestControlPIDPath(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("SURD_CREDS", filepath.Join(dir, "credentials.json"))
	got := controlPIDPath()
	if got == "" {
		t.Fatal("controlPIDPath returned empty")
	}
	if filepath.Dir(got) != dir {
		t.Errorf("connect.pid not sibling of credentials.json: dir=%q want=%q",
			filepath.Dir(got), dir)
	}
	if filepath.Base(got) != "connect.pid" {
		t.Errorf("connect.pid filename = %q", filepath.Base(got))
	}
}

// TestProcessAliveSelf — sanity that processAlive returns true for
// our own PID.  Catches signal/exec API regressions on the Unix path.
// On Windows the function always returns true for any FindProcess
// success, which is intentional and tested by the same assertion.
func TestProcessAliveSelf(t *testing.T) {
	if !processAlive(os.Getpid()) {
		t.Errorf("processAlive(self) = false")
	}
}

// TestProcessAliveBogus — extremely high PID must report false on
// Unix.  Windows path always reports true (see processAlive doc), so
// skip there.
func TestProcessAliveBogus(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Windows FindProcess always succeeds")
	}
	if processAlive(2_000_000_000) {
		t.Errorf("processAlive(2e9) = true; expected false (pid almost certainly does not exist)")
	}
}

// TestReadControlPID_Stale — a pid file containing garbage must NOT
// produce a (positive, true) reading.  Otherwise `surd stop` would
// try to signal nonsense.
func TestReadControlPID_Stale(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("SURD_CREDS", filepath.Join(dir, "credentials.json"))
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(controlPIDPath(), []byte("not-a-number\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	pid, ok := readControlPID()
	if ok || pid != 0 {
		t.Errorf("garbage pid file returned (pid=%d, ok=%v); want (0, false)", pid, ok)
	}
}

// TestDefaultServer_Env — SURD_SERVER takes precedence over saved
// creds so `surd login --server=…` and CI overrides work without
// editing the creds file.
func TestDefaultServer_Env(t *testing.T) {
	t.Setenv("SURD_SERVER", "https://env-override.example")
	if got := defaultServer(); got != "https://env-override.example" {
		t.Errorf("defaultServer = %q, want env override", got)
	}
}

// TestDefaultServer_FromCreds — with SURD_SERVER unset, defaultServer
// falls back to the persisted credentials so a returning user doesn't
// have to retype the URL.
func TestDefaultServer_FromCreds(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("SURD_SERVER", "")
	t.Setenv("SURD_CREDS", filepath.Join(dir, "credentials.json"))
	c := &credentials{Server: "https://from-creds.example", AgentID: "a", AgentKey: "k"}
	if err := saveCreds(c); err != nil {
		t.Fatal(err)
	}
	if got := defaultServer(); got != "https://from-creds.example" {
		t.Errorf("defaultServer = %q, want fall-through to creds", got)
	}
}

// TestCmdLogoutIdempotent — running logout twice must not error on
// the second call (creds already gone is a normal user state).
func TestCmdLogoutIdempotent(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("SURD_CREDS", filepath.Join(dir, "credentials.json"))
	c := &credentials{Server: "https://x", AgentID: "a", AgentKey: "k"}
	if err := saveCreds(c); err != nil {
		t.Fatal(err)
	}
	if err := cmdLogout(nil); err != nil {
		t.Fatalf("first logout: %v", err)
	}
	if err := cmdLogout(nil); err != nil {
		t.Fatalf("second logout (already gone) must be no-op: %v", err)
	}
}
