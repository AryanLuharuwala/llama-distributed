package main

import (
	"bytes"
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"runtime/debug"
	"strings"
	"sync"
	"time"
)

//go:generate sh -c "cp ../scripts/install/install.sh  assets/install.sh && cp ../scripts/install/install.ps1 assets/install.ps1 && cp ../scripts/install/build.sh assets/build.sh && cp ../scripts/install/build.ps1 assets/build.ps1 && cp ../scripts/install/setup.sh assets/setup.sh && cp ../scripts/install/setup.ps1 assets/setup.ps1 && cp ../scripts/install/setup.zsh assets/setup.zsh"

//go:embed assets/install.sh assets/install.ps1 assets/build.sh assets/build.ps1 assets/setup.sh assets/setup.ps1 assets/setup.zsh
var installFS embed.FS

//go:embed assets/install.html
var installPageHTML []byte

// handleInstallPage renders the OS picker + one-liner generator UI.  This
// is the page the dashboard links to from "add a rig"; it works without
// being logged in (the API endpoint /api/install/oneliner is anonymous).
func (s *server) handleInstallPage(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	_, _ = w.Write(installPageHTML)
}

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
	// Point the rig at our /releases proxy so it doesn't hit the GitHub API.
	// The user can still override by setting DIST_SERVER= on the command line.
	b = injectShellDefault(b, "DIST_SERVER", strings.TrimRight(s.cfg.publicURL, "/"))
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

// handleSetupSh serves the lightweight `surd`-only POSIX installer.  We
// inject DIST_SERVER so the rig fetches the binary from us, not GitHub.
func (s *server) handleSetupSh(w http.ResponseWriter, _ *http.Request) {
	b, err := installFS.ReadFile("assets/setup.sh")
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	b = injectShellDefault(b, "DIST_SERVER", strings.TrimRight(s.cfg.publicURL, "/"))
	writeScript(w, "text/x-shellscript; charset=utf-8", b)
}

// handleSetupZsh is the zsh-flavoured variant.  Same DIST_SERVER injection.
func (s *server) handleSetupZsh(w http.ResponseWriter, _ *http.Request) {
	b, err := installFS.ReadFile("assets/setup.zsh")
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	b = injectShellDefault(b, "DIST_SERVER", strings.TrimRight(s.cfg.publicURL, "/"))
	writeScript(w, "text/x-shellscript; charset=utf-8", b)
}

// handleSetupPs1 serves the PowerShell installer.  The PS variant gates on
// `${env:DIST_SERVER}`; we patch the fallback line so it works when piped
// through `iex` (which loses the calling-shell environment).
func (s *server) handleSetupPs1(w http.ResponseWriter, _ *http.Request) {
	b, err := installFS.ReadFile("assets/setup.ps1")
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	b = injectPsDefault(b, "DistServer", strings.TrimRight(s.cfg.publicURL, "/"))
	writeScript(w, "text/plain; charset=utf-8", b)
}

// injectPsDefault rewrites the FIRST line of the form
//    $Name = "${env:NAME}"
// to bake the server's public URL in as the post-env fallback.  The script
// already prefers the env var when set, so this is safe under `iex` piping.
func injectPsDefault(script []byte, varName, value string) []byte {
	needle := []byte("$" + varName + " = \"${env:DIST_SERVER}\"")
	idx := bytes.Index(script, needle)
	if idx < 0 {
		return script
	}
	end := bytes.IndexByte(script[idx:], '\n')
	if end < 0 {
		return script
	}
	replacement := []byte("$" + varName + " = if ($env:DIST_SERVER) { $env:DIST_SERVER } else { \"" + value + "\" }")
	var out bytes.Buffer
	out.Grow(len(script) + len(value))
	out.Write(script[:idx])
	out.Write(replacement)
	out.Write(script[idx+end:])
	return out.Bytes()
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
	// "primary" command field they can paste verbatim, and optionally pin
	// the pair token to a pool so the rig auto-joins on first connect.
	var body struct {
		OS     string `json:"os"`     // linux|macos|windows
		Accel  string `json:"accel"`  // cpu|cuda|rocm|metal|vulkan|auto
		Mode   string `json:"mode"`   // prebuilt|build|auto
		PoolID int64  `json:"pool_id"` // optional: auto-attach on pair
	}
	_ = json.NewDecoder(io.LimitReader(r.Body, 2048)).Decode(&body)

	// Validate pool membership up front so we don't mint a token that points
	// at a pool the user cannot join.  PoolID=0 means "don't auto-attach".
	var poolID any
	if body.PoolID != 0 {
		if _, isMember := s.userIsMember(body.PoolID, u.ID); !isMember {
			writeErr(w, 403, "not a member of the selected pool")
			return
		}
		poolID = body.PoolID
	}

	token := newRandomToken(20)
	// 30-min TTL for install-command tokens — longer than the raw pairing
	// flow because the user may walk to a second machine between clicks.
	expires := time.Now().Add(30 * time.Minute)
	if _, err := s.dbExec(
		`INSERT INTO pair_tokens (token, user_id, created_at, expires_at, pool_id) VALUES (?, ?, ?, ?, ?)`,
		token, u.ID, nowUnix(), expires.Unix(), poolID,
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

// githubAuth adds Authorization: Bearer when GITHUB_TOKEN is set.  Raises the
// anonymous 60/hr rate limit to 5000/hr so burst installs (a whole lab
// spinning up at once) don't drain the quota.
func githubAuth(req *http.Request) {
	if t := os.Getenv("GITHUB_TOKEN"); t != "" {
		req.Header.Set("Authorization", "Bearer "+t)
	}
}

func fetchJSON(c *http.Client, url string, v any) bool {
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return false
	}
	req.Header.Set("Accept", "application/vnd.github+json")
	githubAuth(req)
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

// ─── /releases proxy ──────────────────────────────────────────────────────
//
// The installer on a rig asks us for <target>.tar.gz (e.g. linux-x86_64-cuda.tar.gz)
// rather than the full versioned GitHub asset name.  We look the target up in
// the cached GitHub release probe, fetch the tarball on first miss, and serve
// subsequent requests straight from disk.  This means:
//   - rigs don't need to hit api.github.com (LAN-only installs work)
//   - a single GitHub egress bills the server instead of every rig
//   - the cache keys by the actual asset name so a tag bump invalidates.

// releaseFetchLock serialises concurrent fetches of the same asset so we
// don't download the same tarball ten times when ten rigs come up at once.
var releaseFetchLock sync.Map // assetName (string) -> *sync.Mutex

var errNoReleaseAsset = errors.New("no matching release asset")

// shortTargetOK matches installer-style short target names we'll accept:
//   linux-x86_64-cpu.tar.gz, macos-arm64-metal.tar.gz, windows-x86_64-cuda.zip, …
// Returns (os, arch, accel, ext, ok).
func parseShortTarget(name string) (string, string, string, string, bool) {
	var ext string
	switch {
	case strings.HasSuffix(name, ".tar.gz"):
		ext = ".tar.gz"
	case strings.HasSuffix(name, ".zip"):
		ext = ".zip"
	default:
		return "", "", "", "", false
	}
	stem := strings.TrimSuffix(name, ext)
	parts := strings.Split(stem, "-")
	if len(parts) != 3 {
		return "", "", "", "", false
	}
	return parts[0], parts[1], parts[2], ext, true
}

// findRelease returns the full GitHub asset name for a short target (or "" if
// we haven't seen a release that carries it).  Also returns the release tag.
func findRelease(os_, arch, accel, ext string) (string, string) {
	gTargets.mu.RLock()
	defer gTargets.mu.RUnlock()
	for _, t := range gTargets.targets {
		if t.OS == os_ && t.Arch == arch && t.Accel == accel && t.Prebuilt {
			// Confidence check: the asset extension must match what we claim.
			if ext == ".tar.gz" && !strings.HasSuffix(t.Asset, ".tar.gz") {
				continue
			}
			if ext == ".zip" && !strings.HasSuffix(t.Asset, ".zip") {
				continue
			}
			return t.Asset, gTargets.ref
		}
	}
	return "", ""
}

// cachedPathFor returns the absolute path we use for a given GitHub asset
// name.  Flat dir, filename is the asset name verbatim.
func (s *server) cachedPathFor(assetName string) string {
	return filepath.Join(s.cfg.releasesDir, assetName)
}

// ensureReleaseCached downloads `assetName` from GitHub Releases into the
// cache dir if it isn't already there.  Concurrent calls for the same asset
// block on a per-asset mutex.
func (s *server) ensureReleaseCached(assetName string) (string, error) {
	dest := s.cachedPathFor(assetName)
	if fi, err := os.Stat(dest); err == nil && fi.Size() > 0 {
		return dest, nil
	}

	muAny, _ := releaseFetchLock.LoadOrStore(assetName, &sync.Mutex{})
	mu := muAny.(*sync.Mutex)
	mu.Lock()
	// Cleanup: whoever holds the mutex at the end of the function deletes it
	// from the map.  A late concurrent caller will allocate a fresh mutex via
	// LoadOrStore; correct because only file-system state, not the mutex,
	// gates the fetch, and the disk double-check below catches that case.
	defer func() {
		releaseFetchLock.Delete(assetName)
		mu.Unlock()
	}()
	// Double-check after taking the lock — a sibling goroutine may have
	// finished while we were waiting.
	if fi, err := os.Stat(dest); err == nil && fi.Size() > 0 {
		return dest, nil
	}

	if err := os.MkdirAll(s.cfg.releasesDir, 0o755); err != nil {
		return "", err
	}

	repo := "AryanLuharuwala/llama-distributed"
	if v := os.Getenv("DIST_REPO"); v != "" {
		repo = v
	}
	// We need the tag — `gTargets.ref` holds the server's git SHA, not the
	// release tag.  Re-derive from the targets cache via the asset name.
	tag := ""
	gTargets.mu.RLock()
	for _, t := range gTargets.targets {
		if t.Asset == assetName {
			// We stored the tag as gTargets.ref separately; that's the
			// server git SHA, not the release tag.  The tag name isn't
			// kept on each target today; re-probe is cheap but wasteful,
			// so we store the tag globally in refreshTargets via a new
			// field.  For now, fall back to pulling the assets list.
			break
		}
	}
	gTargets.mu.RUnlock()

	_ = tag // filled below

	// Ask GitHub for the current release (prerelease-aware) and pull the
	// matching asset's browser_download_url.  We don't rely on asset IDs so
	// the URL stays stable across re-uploads.
	client := &http.Client{Timeout: 30 * time.Second}
	var rel struct {
		TagName string `json:"tag_name"`
		Assets  []struct {
			Name string `json:"name"`
			URL  string `json:"browser_download_url"`
		} `json:"assets"`
	}
	if !fetchJSON(client, "https://api.github.com/repos/"+repo+"/releases/latest", &rel) || rel.TagName == "" {
		var list []struct {
			TagName string `json:"tag_name"`
			Assets  []struct {
				Name string `json:"name"`
				URL  string `json:"browser_download_url"`
			} `json:"assets"`
		}
		if !fetchJSON(client, "https://api.github.com/repos/"+repo+"/releases?per_page=1", &list) || len(list) == 0 {
			return "", errNoReleaseAsset
		}
		rel.TagName = list[0].TagName
		rel.Assets = list[0].Assets
	}

	var dlURL string
	for _, a := range rel.Assets {
		if a.Name == assetName {
			dlURL = a.URL
			break
		}
	}
	if dlURL == "" {
		return "", errNoReleaseAsset
	}

	// Download to a sibling .part file, then atomic rename.  Avoids serving
	// a half-written tarball if we crash mid-fetch.
	part := dest + ".part"
	dlReq, err := http.NewRequest("GET", dlURL, nil)
	if err != nil {
		return "", err
	}
	// Bearer on the download too — required for private repos, harmless on
	// public S3 URLs.
	githubAuth(dlReq)
	resp, err := client.Do(dlReq)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return "", fmt.Errorf("github returned %d", resp.StatusCode)
	}
	f, err := os.Create(part)
	if err != nil {
		return "", err
	}
	if _, err := io.Copy(f, resp.Body); err != nil {
		_ = f.Close()
		_ = os.Remove(part)
		return "", err
	}
	if err := f.Close(); err != nil {
		return "", err
	}
	if err := os.Rename(part, dest); err != nil {
		return "", err
	}
	log.Printf("cached release asset %s (%d bytes)", assetName, fileSize(dest))
	return dest, nil
}

// validSurdBinName matches `surd-<os>-<arch>` and `surd-windows-<arch>.exe`.
// Returns the cleaned filename (or "" if rejected).  Refuses anything with
// a path separator or `..` to prevent traversal.
func validSurdBinName(name string) string {
	if name == "" || strings.ContainsAny(name, "/\\") || strings.Contains(name, "..") {
		return ""
	}
	// gpunet rename: the CLI artifact is now gpunet-<os>-<arch>; accept the
	// legacy surd-<os>-<arch> too during the transition.
	if !strings.HasPrefix(name, "gpunet-") && !strings.HasPrefix(name, "surd-") {
		return ""
	}
	stem := strings.TrimSuffix(name, ".exe")
	parts := strings.Split(stem, "-")
	if len(parts) != 3 {
		return ""
	}
	switch parts[1] {
	case "linux", "darwin", "windows":
	default:
		return ""
	}
	switch parts[2] {
	case "amd64", "arm64":
	default:
		return ""
	}
	if parts[1] == "windows" && !strings.HasSuffix(name, ".exe") {
		// We canonicalise: windows builds always end .exe.
		return name + ".exe"
	}
	return name
}

// handleSurdBinary serves built `surd` CLI binaries.
//
//   GET /releases/surd-linux-amd64
//   GET /releases/surd-darwin-arm64
//   GET /releases/surd-windows-amd64.exe
//
// Binaries are read from <releasesDir>/surd/<name>.  Operators drop builds
// there during deploy (e.g. via `go build -o releases-cache/surd/...`).
func (s *server) handleSurdBinary(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	cleaned := validSurdBinName(name)
	if cleaned == "" {
		writeErr(w, 400, "bad cli binary name; expected gpunet-<os>-<arch>[.exe]")
		return
	}
	// Serve from <releasesDir>/cli (new) or the legacy surd/ dir. Operators
	// drop builds in either during the rename transition.
	path := filepath.Join(s.cfg.releasesDir, "cli", cleaned)
	if fi, err := os.Stat(path); err != nil || fi.IsDir() || fi.Size() == 0 {
		legacy := filepath.Join(s.cfg.releasesDir, "surd", cleaned)
		if fi2, err2 := os.Stat(legacy); err2 == nil && !fi2.IsDir() && fi2.Size() > 0 {
			path = legacy
		} else {
			writeErr(w, 404, "binary not published: "+cleaned)
			return
		}
	}
	w.Header().Set("Content-Type", "application/octet-stream")
	w.Header().Set("Cache-Control", "public, max-age=300")
	http.ServeFile(w, r, path)
}

func fileSize(p string) int64 {
	if fi, err := os.Stat(p); err == nil {
		return fi.Size()
	}
	return 0
}

// handleReleaseAsset serves short-name tarballs from the control plane.
// GET /releases/{name}  e.g. /releases/linux-x86_64-cuda.tar.gz
func (s *server) handleReleaseAsset(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	// surd CLI binaries are served from a separate flat dir without going
	// through the GitHub asset matrix.
	if strings.HasPrefix(name, "surd-") {
		s.handleSurdBinary(w, r)
		return
	}
	os_, arch, accel, ext, ok := parseShortTarget(name)
	if !ok {
		writeErr(w, 400, "bad asset name; expected <os>-<arch>-<accel>.tar.gz|.zip")
		return
	}
	assetName, _ := findRelease(os_, arch, accel, ext)
	if assetName == "" {
		// Targets cache may be cold — kick a refresh and retry once.
		refreshTargets(s)
		assetName, _ = findRelease(os_, arch, accel, ext)
	}
	if assetName == "" {
		writeErr(w, 404, "no release asset for "+name)
		return
	}

	path, err := s.ensureReleaseCached(assetName)
	if err != nil {
		log.Printf("release fetch %s: %v", assetName, err)
		writeErr(w, 502, "upstream fetch failed")
		return
	}
	// Serve with ETag == asset name so rigs cache by asset identity; a new
	// release (different asset name) always bypasses the cache.
	w.Header().Set("Content-Type", "application/octet-stream")
	w.Header().Set("ETag", `"`+assetName+`"`)
	w.Header().Set("Cache-Control", "public, max-age=3600")
	http.ServeFile(w, r, path)
}
