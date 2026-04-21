package main

import (
	"bytes"
	"embed"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"runtime/debug"
	"strings"
	"sync"
	"time"
)

//go:generate sh -c "cp ../scripts/install/install.sh  assets/install.sh && cp ../scripts/install/install.ps1 assets/install.ps1 && cp ../scripts/install/build.sh assets/build.sh && cp ../scripts/install/build.ps1 assets/build.ps1"

//go:embed assets/install.sh assets/install.ps1 assets/build.sh assets/build.ps1
var installFS embed.FS

// gitSHA can be injected at build time via:
//   go build -ldflags "-X main.gitSHA=<sha>"
// Falls back to runtime/debug.ReadBuildInfo() (populated automatically when
// the binary is built from a clean git worktree), then to "unknown".
var gitSHA string

// serverRef returns the git commit the server was built from, used to pin
// source-build installs to the exact protocol version this server speaks.
func serverRef() string {
	if gitSHA != "" {
		return gitSHA
	}
	if bi, ok := debug.ReadBuildInfo(); ok {
		for _, s := range bi.Settings {
			if s.Key == "vcs.revision" && s.Value != "" {
				return s.Value
			}
		}
	}
	// Last-ditch: env override for dev shells.
	if v := os.Getenv("DIST_GIT_SHA"); v != "" {
		return v
	}
	return "unknown"
}

func (s *server) handleInstallSh(w http.ResponseWriter, _ *http.Request) {
	b, err := installFS.ReadFile("assets/install.sh")
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	writeScript(w, "text/x-shellscript; charset=utf-8", b)
}

func (s *server) handleInstallPs1(w http.ResponseWriter, _ *http.Request) {
	b, err := installFS.ReadFile("assets/install.ps1")
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	writeScript(w, "text/plain; charset=utf-8", b)
}

// handleBuildSh serves the source-build bootstrapper.  We inject the server's
// own git SHA as the default --ref so client+server stay in lockstep without
// the user needing to know what commit we run.
func (s *server) handleBuildSh(w http.ResponseWriter, _ *http.Request) {
	b, err := installFS.ReadFile("assets/build.sh")
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	b = injectShellDefault(b, "REF", serverRef())
	writeScript(w, "text/x-shellscript; charset=utf-8", b)
}

func (s *server) handleBuildPs1(w http.ResponseWriter, _ *http.Request) {
	b, err := installFS.ReadFile("assets/build.ps1")
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	writeScript(w, "text/plain; charset=utf-8", b)
}

func writeScript(w http.ResponseWriter, ct string, body []byte) {
	w.Header().Set("Content-Type", ct)
	w.Header().Set("Cache-Control", "no-store")
	_, _ = w.Write(body)
}

// injectShellDefault replaces the FIRST line of the form
//    VAR="${VAR:-...}"
// with VAR="${VAR:-<value>}" so the server's preferred default wins over the
// script's baked-in fallback.  No-op if the variable isn't found.
func injectShellDefault(script []byte, name, value string) []byte {
	needle := []byte(name + `="${` + name + `:-`)
	idx := bytes.Index(script, needle)
	if idx < 0 {
		return script
	}
	// find end of that line
	end := bytes.IndexByte(script[idx:], '\n')
	if end < 0 {
		return script
	}
	line := []byte(name + `="${` + name + `:-` + value + `}"`)
	var out bytes.Buffer
	out.Grow(len(script) + len(value))
	out.Write(script[:idx])
	out.Write(line)
	out.Write(script[idx+end:])
	return out.Bytes()
}

// handleInstallCommand mints a pair token and returns ready-to-paste
// one-liners for prebuilt and source-build flows, both OSes.
func (s *server) handleInstallCommand(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}

	// Optional body — clients can narrow to one (os, accel) to get a single
	// "primary" command field they can paste verbatim.
	var body struct {
		OS    string `json:"os"`    // linux|macos|windows
		Accel string `json:"accel"` // cpu|cuda|rocm|metal|vulkan|auto
		Mode  string `json:"mode"`  // prebuilt|build|auto
	}
	_ = json.NewDecoder(io.LimitReader(r.Body, 2048)).Decode(&body)

	token := newRandomToken(20)
	// 30-min TTL for install-command tokens — longer than the raw pairing
	// flow because the user may walk to a second machine between clicks.
	expires := time.Now().Add(30 * time.Minute)
	if _, err := s.db.Exec(
		`INSERT INTO pair_tokens (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)`,
		token, u.ID, nowUnix(), expires.Unix(),
	); err != nil {
		writeErr(w, 500, err.Error())
		return
	}

	base := strings.TrimRight(s.cfg.publicURL, "/")
	wsURL := httpToWS(base) + "/ws/agent"
	deepLink := "distpool://pair?token=" + token + "&server=" + wsURL

	ref := serverRef()
	accelArg := body.Accel
	if accelArg == "" || accelArg == "auto" {
		accelArg = "" // let the script auto-detect
	}

	bashPrebuilt := "curl -fsSL " + base + "/install.sh | sh -s -- --pair '" + deepLink + "'"
	bashBuild := "curl -fsSL " + base + "/build.sh | bash -s -- --pair '" + deepLink + "' --ref " + ref
	if accelArg != "" {
		bashBuild += " --accel " + accelArg
	}

	pwshPrebuilt := "$p='" + deepLink + "'; " +
		"iwr -useb " + base + "/install.ps1 -OutFile $env:TEMP\\install.ps1; " +
		"powershell -ExecutionPolicy Bypass -File $env:TEMP\\install.ps1 -Pair $p"
	pwshBuild := "$p='" + deepLink + "'; " +
		"iwr -useb " + base + "/build.ps1 -OutFile $env:TEMP\\build.ps1; " +
		"powershell -ExecutionPolicy Bypass -File $env:TEMP\\build.ps1 -Pair $p -Ref " + ref

	// Pick the "primary" command — the one the UI highlights for copy.
	primaryBash, primaryPwsh := bashPrebuilt, pwshPrebuilt
	wantBuild := body.Mode == "build" || !targetIsPrebuilt(s, body.OS, body.Accel)
	if body.Mode == "auto" {
		wantBuild = !targetIsPrebuilt(s, body.OS, body.Accel)
	}
	if wantBuild {
		primaryBash, primaryPwsh = bashBuild, pwshBuild
	}

	writeJSON(w, 200, map[string]any{
		"token":         token,
		"deep_link":     deepLink,
		"expires_at":    expires.Unix(),
		"server_ref":    ref,
		"bash":          primaryBash,
		"pwsh":          primaryPwsh,
		"bash_prebuilt": bashPrebuilt,
		"bash_build":    bashBuild,
		"pwsh_prebuilt": pwshPrebuilt,
		"pwsh_build":    pwshBuild,
	})
}

// ─── Install targets matrix ────────────────────────────────────────────────

type installTarget struct {
	OS       string `json:"os"`
	Arch     string `json:"arch"`
	Accel    string `json:"accel"`
	Prebuilt bool   `json:"prebuilt"` // a matching release asset exists
	Build    bool   `json:"build"`    // source-build is supported
	Asset    string `json:"asset,omitempty"`
}

// Static matrix of what the project supports. Prebuilt flags get overwritten
// from the GitHub release probe below; the rest is compile-time knowledge.
var baseTargets = []installTarget{
	{OS: "linux", Arch: "x86_64", Accel: "cpu", Build: true},
	{OS: "linux", Arch: "x86_64", Accel: "cuda", Build: true},
	{OS: "linux", Arch: "x86_64", Accel: "rocm", Build: true},
	{OS: "linux", Arch: "x86_64", Accel: "vulkan", Build: true},
	{OS: "linux", Arch: "aarch64", Accel: "cpu", Build: true},
	{OS: "macos", Arch: "arm64", Accel: "metal", Build: true},
	{OS: "macos", Arch: "arm64", Accel: "cpu", Build: true},
	{OS: "macos", Arch: "x86_64", Accel: "cpu", Build: true},
	{OS: "windows", Arch: "x86_64", Accel: "cpu", Build: false},
	{OS: "windows", Arch: "x86_64", Accel: "cuda", Build: false},
}

type targetsCache struct {
	mu        sync.RWMutex
	targets   []installTarget
	ref       string
	fetchedAt time.Time
}

var gTargets targetsCache

// targetIsPrebuilt is called from handleInstallCommand to pick the default
// install mode.  OS/accel may be empty → we conservatively say "yes prebuilt"
// so the UI keeps the cheaper default.
func targetIsPrebuilt(_ *server, os, accel string) bool {
	if os == "" || accel == "" {
		return true
	}
	gTargets.mu.RLock()
	defer gTargets.mu.RUnlock()
	for _, t := range gTargets.targets {
		if t.OS == os && t.Accel == accel {
			return t.Prebuilt
		}
	}
	return false
}

// handleInstallTargets returns the targets matrix + the server's git SHA.
// Both the UI and the CLI installers read from this one source of truth.
//
// GET /api/install_targets
func (s *server) handleInstallTargets(w http.ResponseWriter, _ *http.Request) {
	gTargets.mu.RLock()
	targets := gTargets.targets
	ref := gTargets.ref
	age := time.Since(gTargets.fetchedAt)
	gTargets.mu.RUnlock()

	if len(targets) == 0 || age > 15*time.Minute {
		// Fire-and-forget refresh; serve whatever we had (even if stale).
		go refreshTargets(s)
	}
	if len(targets) == 0 {
		targets = withBuildOnly(baseTargets)
	}
	if ref == "" {
		ref = serverRef()
	}
	writeJSON(w, 200, map[string]any{
		"server_ref": ref,
		"targets":    targets,
	})
}

func withBuildOnly(in []installTarget) []installTarget {
	out := make([]installTarget, len(in))
	copy(out, in)
	for i := range out {
		out[i].Prebuilt = false
		out[i].Asset = ""
	}
	return out
}

// refreshTargets probes GitHub for the newest release (including prereleases)
// and flips Prebuilt=true on every matching matrix entry.  Best-effort: any
// error leaves the old cache in place.
func refreshTargets(s *server) {
	// Snapshot the base matrix so we don't partially mutate global state.
	targets := make([]installTarget, len(baseTargets))
	copy(targets, baseTargets)

	repo := "AryanLuharuwala/llama-distributed"
	if v := os.Getenv("DIST_REPO"); v != "" {
		repo = v
	}

	client := &http.Client{Timeout: 10 * time.Second}
	// Prefer /releases/latest; fall back to newest-including-prereleases.
	var rel struct {
		TagName string `json:"tag_name"`
		Assets  []struct {
			Name string `json:"name"`
		} `json:"assets"`
	}
	ok := fetchJSON(client, "https://api.github.com/repos/"+repo+"/releases/latest", &rel)
	if !ok || rel.TagName == "" {
		var list []struct {
			TagName string `json:"tag_name"`
			Assets  []struct {
				Name string `json:"name"`
			} `json:"assets"`
		}
		if fetchJSON(client, "https://api.github.com/repos/"+repo+"/releases?per_page=1", &list) && len(list) > 0 {
			rel.TagName = list[0].TagName
			rel.Assets = list[0].Assets
		}
	}
	if rel.TagName == "" {
		// Couldn't reach GitHub — keep old cache (or seed build-only).
		gTargets.mu.Lock()
		if len(gTargets.targets) == 0 {
			gTargets.targets = withBuildOnly(targets)
			gTargets.ref = serverRef()
			gTargets.fetchedAt = time.Now()
		}
		gTargets.mu.Unlock()
		return
	}

	// Map each asset name back to a target.
	for i := range targets {
		t := &targets[i]
		// matches names like  llama-distributed-v0.0.0-dev-linux-x86_64-cuda.tar.gz
		suffixes := []string{
			t.OS + "-" + t.Arch + "-" + t.Accel + ".tar.gz",
			t.OS + "-" + t.Arch + "-" + t.Accel + ".zip",
		}
		for _, a := range rel.Assets {
			for _, sfx := range suffixes {
				if strings.HasSuffix(a.Name, sfx) {
					t.Prebuilt = true
					t.Asset = a.Name
					break
				}
			}
			if t.Prebuilt {
				break
			}
		}
	}

	gTargets.mu.Lock()
	gTargets.targets = targets
	gTargets.ref = serverRef()
	gTargets.fetchedAt = time.Now()
	gTargets.mu.Unlock()
}

func fetchJSON(c *http.Client, url string, v any) bool {
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return false
	}
	req.Header.Set("Accept", "application/vnd.github+json")
	resp, err := c.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return false
	}
	return json.NewDecoder(resp.Body).Decode(v) == nil
}
