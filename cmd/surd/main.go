// surd — the user-facing CLI for the distributed inference network.
//
//	curl -fsSL https://<host>/install.sh | sh   # one-line installer
//	surd login                                  # browser-paired device code
//	surd connect                                # bring the rig online
//
// All commands accept --json so the desktop widget can parse them
// programmatically.  No daemon required for the common case: `surd connect`
// runs dist-node in the foreground until Ctrl-C; daemon mode is opt-in via
// `--daemon` (Phase 3, separate code path).
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"time"
)

const version = "0.1.0"

// usage is the top-level help.  Mirrors `git`/`docker` style: a single
// dispatch table that points each subcommand at its own flag set.
const usage = `surd — distributed inference rig control

USAGE
  surd <command> [flags]

COMMANDS
  login      Pair this machine with your account (device-code in browser)
  connect    Bring the rig online (runs dist-node in foreground)
  status     Show local rig state
  stop       Tell a running surd connect to shut down cleanly
  pools      List pools you're a member of
  share      Print an invite link for a pool
  logout     Discard local credentials
  daemon     Install/uninstall the auto-start service (systemd/launchd/schtasks)
  update     Self-update from the dashboard release manifest
  version    Print version
  help       This message

Add --json to most commands for machine-readable output.
Add --server=URL to override the dashboard endpoint (default: read from creds).
`

// credsFile is where we persist the device-code result.  Mode 0600 so other
// local users can't read the token even if they share the box.
func credsPath() string {
	if c := os.Getenv("SURD_CREDS"); c != "" {
		return c
	}
	if runtime.GOOS == "windows" {
		if d := os.Getenv("APPDATA"); d != "" {
			return filepath.Join(d, "surd", "credentials.json")
		}
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".config", "surd", "credentials.json")
}

type credentials struct {
	Server   string `json:"server"`
	AgentID  string `json:"agent_id"`
	AgentKey string `json:"agent_key"`
}

func loadCreds() (*credentials, error) {
	p := credsPath()
	b, err := os.ReadFile(p)
	if err != nil {
		return nil, err
	}
	var c credentials
	if err := json.Unmarshal(b, &c); err != nil {
		return nil, fmt.Errorf("creds %s: %w", p, err)
	}
	return &c, nil
}

func saveCreds(c *credentials) error {
	p := credsPath()
	if err := os.MkdirAll(filepath.Dir(p), 0o700); err != nil {
		return err
	}
	b, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}
	// Write to a temp file then rename so we never leave a half-written
	// creds file behind if the disk fills up mid-write.
	tmp := p + ".tmp"
	if err := os.WriteFile(tmp, b, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, p)
}

func main() {
	if len(os.Args) < 2 {
		fmt.Print(usage)
		os.Exit(2)
	}
	cmd := os.Args[1]
	args := os.Args[2:]
	var err error
	switch cmd {
	case "login":
		err = cmdLogin(args)
	case "connect":
		err = cmdConnect(args)
	case "status":
		err = cmdStatus(args)
	case "stop":
		err = cmdStop(args)
	case "pools":
		err = cmdPools(args)
	case "share":
		err = cmdShare(args)
	case "logout":
		err = cmdLogout(args)
	case "update":
		err = cmdUpdate(args)
	case "daemon":
		err = cmdDaemon(args)
	case "version", "-v", "--version":
		fmt.Println("surd", version)
	case "help", "-h", "--help":
		fmt.Print(usage)
	default:
		fmt.Fprintf(os.Stderr, "surd: unknown command %q\n\n%s", cmd, usage)
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "surd:", err)
		os.Exit(1)
	}
}

// ── login ────────────────────────────────────────────────────────────────
// Implements the device-code flow against /api/device/code + /api/device/token.
// The user pastes a short code into the dashboard (or clicks the
// pre-filled verification_url_complete) and we then poll until approved.
func cmdLogin(args []string) error {
	fs := flag.NewFlagSet("login", flag.ExitOnError)
	server := fs.String("server", defaultServer(), "dashboard URL (https://…)")
	jsonOut := fs.Bool("json", false, "machine-readable output")
	openBrowser := fs.Bool("open", true, "try to open the verification URL in a browser")
	fs.Parse(args)
	if *server == "" {
		return errors.New("--server is required on first login (set SURD_SERVER or pass --server)")
	}

	hostname, _ := os.Hostname()
	body, _ := json.Marshal(map[string]any{
		"hostname":   hostname,
		"n_gpus":     0,
		"vram_bytes": 0,
	})
	resp, err := http.Post(strings.TrimRight(*server, "/")+"/api/device/code",
		"application/json", bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("mint device code: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		raw, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("device code mint: %d %s", resp.StatusCode, raw)
	}
	var mint struct {
		DeviceCode               string `json:"device_code"`
		UserCode                 string `json:"user_code"`
		VerificationURL          string `json:"verification_url"`
		VerificationURLComplete  string `json:"verification_url_complete"`
		ExpiresIn                int    `json:"expires_in"`
		Interval                 int    `json:"interval"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&mint); err != nil {
		return err
	}

	if *jsonOut {
		_ = json.NewEncoder(os.Stdout).Encode(mint)
	} else {
		fmt.Println()
		fmt.Println("  ▸ open this URL in your browser to approve this rig:")
		fmt.Println("       ", mint.VerificationURLComplete)
		fmt.Println()
		fmt.Println("    user code:  ", mint.UserCode)
		fmt.Println("    expires in: ", mint.ExpiresIn, "seconds")
		fmt.Println()
	}
	if *openBrowser {
		_ = tryOpenBrowser(mint.VerificationURLComplete)
	}

	// Poll until approved or expired.  Interval clamped at 2s minimum
	// so we don't hammer the server; the server suggests its own min.
	interval := mint.Interval
	if interval < 2 {
		interval = 2
	}
	deadline := time.Now().Add(time.Duration(mint.ExpiresIn) * time.Second)
	for time.Now().Before(deadline) {
		time.Sleep(time.Duration(interval) * time.Second)
		tok, err := pollDeviceToken(*server, mint.DeviceCode)
		if err != nil {
			if errors.Is(err, errPending) {
				if !*jsonOut {
					fmt.Print(".")
				}
				continue
			}
			return err
		}
		creds := &credentials{
			Server:   *server,
			AgentID:  tok.AgentID,
			AgentKey: tok.AgentKey,
		}
		// tok.Server is a WS URL, surd works at the HTTP layer so we
		// stash the dashboard URL we used for device code instead.
		if err := saveCreds(creds); err != nil {
			return err
		}
		if *jsonOut {
			_ = json.NewEncoder(os.Stdout).Encode(map[string]any{
				"ok":       true,
				"agent_id": tok.AgentID,
				"server":   creds.Server,
			})
		} else {
			fmt.Println()
			fmt.Println("  ✓ paired as", tok.AgentID)
			fmt.Println("  ✓ credentials saved to", credsPath())
			fmt.Println()
			fmt.Println("    next:  surd connect")
		}
		return nil
	}
	return errors.New("login timed out — run `surd login` again")
}

var errPending = errors.New("authorization pending")

func pollDeviceToken(server, deviceCode string) (*deviceTokenOut, error) {
	body, _ := json.Marshal(map[string]string{"device_code": deviceCode})
	resp, err := http.Post(strings.TrimRight(server, "/")+"/api/device/token",
		"application/json", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode == 428 {
		return nil, errPending
	}
	if resp.StatusCode == 410 {
		return nil, errors.New("device code expired")
	}
	if resp.StatusCode != 200 {
		raw, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("token poll: %d %s", resp.StatusCode, raw)
	}
	var out deviceTokenOut
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	return &out, nil
}

type deviceTokenOut struct {
	AgentID  string `json:"agent_id"`
	AgentKey string `json:"agent_key"`
	Server   string `json:"server"`
}

// ── connect ───────────────────────────────────────────────────────────────
// Reads credentials, fetches bootstrap (pool + WS endpoint + TURN creds),
// and execs `dist-node` with the resulting flags.  Foreground; Ctrl-C
// forwards SIGTERM to dist-node for a clean unregister.
func cmdConnect(args []string) error {
	fs := flag.NewFlagSet("connect", flag.ExitOnError)
	jsonOut := fs.Bool("json", false, "machine-readable status")
	binary := fs.String("dist-node", "dist-node", "path to the dist-node binary")
	maxVRAM := fs.Int("max-vram", 0, "limit VRAM use to N MiB (0 = all)")
	idleOnly := fs.Bool("idle-only", false, "only accept jobs when the system is idle")
	inviteToken := fs.String("invite", "", "pool invite token (skip pool selection)")
	dryRun := fs.Bool("dry-run", false, "print the dist-node command and exit")
	fs.Parse(args)

	creds, err := loadCreds()
	if err != nil {
		return errors.New("no credentials — run `surd login` first")
	}

	// Optional invite handling: redeem before bootstrap so the user lands
	// in the right pool immediately.
	if *inviteToken != "" {
		if err := redeemInvite(creds, *inviteToken); err != nil {
			fmt.Fprintln(os.Stderr, "surd: warning — invite redeem failed:", err)
		}
	}

	boot, err := fetchBootstrap(creds)
	if err != nil {
		return err
	}

	// Build dist-node argv.  Keep this minimal — dist-node has its own
	// config file for the long tail; surd just hands over identity + server.
	argv := []string{
		"--server", boot.WSEndpoint,
		"--agent-id", creds.AgentID,
		"--agent-key", creds.AgentKey,
	}
	if boot.PoolSlug != "" {
		argv = append(argv, "--pool", boot.PoolSlug)
	}
	if *maxVRAM > 0 {
		argv = append(argv, "--max-vram-mib", fmt.Sprint(*maxVRAM))
	}
	if *idleOnly {
		argv = append(argv, "--idle-only")
	}

	if *dryRun || *jsonOut {
		out := map[string]any{
			"binary": *binary,
			"argv":   argv,
			"pool":   boot.PoolSlug,
			"ws":     boot.WSEndpoint,
		}
		if *jsonOut {
			_ = json.NewEncoder(os.Stdout).Encode(out)
		} else {
			fmt.Println(*binary, strings.Join(argv, " "))
		}
		if *dryRun {
			return nil
		}
	} else {
		fmt.Println("  ▸ connecting", creds.AgentID, "to pool", boot.PoolSlug)
		fmt.Println("    via", boot.WSEndpoint)
		fmt.Println("    (Ctrl-C to disconnect cleanly)")
		fmt.Println()
	}

	// Exec dist-node, forwarding signals.
	ctx, cancel := signal.NotifyContext(context.Background(),
		syscall.SIGINT, syscall.SIGTERM)
	defer cancel()
	cmd := exec.CommandContext(ctx, *binary, argv...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Stdin = os.Stdin
	// Record our PID so `surd stop` and the widget can find us.  Best-effort
	// — failure here doesn't block the connect.
	pidPath := controlPIDPath()
	if pidPath != "" {
		_ = os.MkdirAll(filepath.Dir(pidPath), 0o700)
		_ = os.WriteFile(pidPath, []byte(fmt.Sprint(os.Getpid())), 0o600)
		defer os.Remove(pidPath)
	}
	if err := cmd.Run(); err != nil {
		// If the user Ctrl-C'd, treat as clean exit.
		if ctx.Err() != nil {
			return nil
		}
		// Common case: dist-node not on PATH.
		if errors.Is(err, exec.ErrNotFound) {
			return fmt.Errorf("dist-node binary not found on PATH (set --dist-node=…)")
		}
		return fmt.Errorf("dist-node exited: %w", err)
	}
	return nil
}

type bootstrap struct {
	PoolSlug   string `json:"pool_slug"`
	WSEndpoint string `json:"ws_endpoint"`
	BaseURL    string `json:"base_url"`
}

func fetchBootstrap(c *credentials) (*bootstrap, error) {
	// Use the agent's auto-mint endpoint to list pools the rig can use.
	req, _ := http.NewRequest("GET",
		strings.TrimRight(c.Server, "/")+"/api/agent/pools", nil)
	req.Header.Set("X-Agent-Key", c.AgentKey)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("bootstrap: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		raw, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("bootstrap: %d %s", resp.StatusCode, raw)
	}
	var d struct {
		Pools []struct {
			ID      int64  `json:"id"`
			Name    string `json:"name"`
			Slug    string `json:"slug"`
			BaseURL string `json:"base_url"`
		} `json:"pools"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&d); err != nil {
		return nil, err
	}
	b := &bootstrap{
		WSEndpoint: httpToWS(strings.TrimRight(c.Server, "/")) + "/ws/agent",
	}
	if len(d.Pools) > 0 {
		b.PoolSlug = d.Pools[0].Slug
		b.BaseURL = d.Pools[0].BaseURL
	}
	return b, nil
}

func httpToWS(u string) string {
	if strings.HasPrefix(u, "https://") {
		return "wss://" + strings.TrimPrefix(u, "https://")
	}
	if strings.HasPrefix(u, "http://") {
		return "ws://" + strings.TrimPrefix(u, "http://")
	}
	return u
}

func redeemInvite(c *credentials, token string) error {
	body, _ := json.Marshal(map[string]string{"invite": token})
	// Note: /api/pools/join requires a *user* session, not an agent key.
	// On a fresh CLI install we don't have one, so this is best-effort —
	// the user can also redeem via the web UI before running `surd connect`.
	resp, err := http.Post(strings.TrimRight(c.Server, "/")+"/api/pools/join",
		"application/json", bytes.NewReader(body))
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		raw, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("%d %s", resp.StatusCode, raw)
	}
	return nil
}

// ── status / stop / pools / share / logout / update ────────────────────────

func cmdStatus(args []string) error {
	fs := flag.NewFlagSet("status", flag.ExitOnError)
	jsonOut := fs.Bool("json", false, "machine-readable output")
	remote := fs.Bool("remote", false, "fetch live state from /api/widget/state")
	fs.Parse(args)
	creds, err := loadCreds()
	if err != nil {
		if *jsonOut {
			_ = json.NewEncoder(os.Stdout).Encode(map[string]any{"logged_in": false})
			return nil
		}
		fmt.Println("not logged in — run `surd login`")
		return nil
	}

	// Local state: is surd connect running?
	running := false
	if pid, ok := readControlPID(); ok && processAlive(pid) {
		running = true
	}

	if *remote {
		req, _ := http.NewRequest("GET",
			strings.TrimRight(creds.Server, "/")+"/api/widget/state", nil)
		req.Header.Set("X-Agent-Key", creds.AgentKey)
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return err
		}
		defer resp.Body.Close()
		raw, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != 200 {
			return fmt.Errorf("%d %s", resp.StatusCode, raw)
		}
		if *jsonOut {
			os.Stdout.Write(raw)
			return nil
		}
		var d struct {
			Online    int     `json:"online"`
			Total     int     `json:"total"`
			TokensPS  float64 `json:"tokens_sec"`
			Inflight  int     `json:"inflight"`
			Tokens24h int64   `json:"tokens_24h"`
		}
		if err := json.Unmarshal(raw, &d); err != nil {
			return err
		}
		fmt.Println("server     :", creds.Server)
		fmt.Println("agent_id   :", creds.AgentID)
		fmt.Println("local conn :", running)
		fmt.Println("rigs       :", d.Online, "/", d.Total, "online")
		fmt.Println("throughput :", d.TokensPS, "tok/s,", d.Inflight, "inflight")
		fmt.Println("24h tokens :", d.Tokens24h)
		return nil
	}

	if *jsonOut {
		_ = json.NewEncoder(os.Stdout).Encode(map[string]any{
			"logged_in": true,
			"server":    creds.Server,
			"agent_id":  creds.AgentID,
			"running":   running,
		})
		return nil
	}
	fmt.Println("server  :", creds.Server)
	fmt.Println("agent_id:", creds.AgentID)
	fmt.Println("running :", running)
	return nil
}

// readControlPID returns the PID of a running `surd connect`, if any.
func readControlPID() (int, bool) {
	raw, err := os.ReadFile(controlPIDPath())
	if err != nil {
		return 0, false
	}
	pid := 0
	if _, err := fmt.Sscanf(strings.TrimSpace(string(raw)), "%d", &pid); err != nil || pid <= 0 {
		return 0, false
	}
	return pid, true
}

// processAlive returns true if signal 0 to pid succeeds (POSIX) — i.e.
// the kernel knows about the process and we're allowed to signal it.
// On Windows we fall back to FindProcess (which always succeeds), so
// callers should also clean up stale pid files on connect-shutdown.
func processAlive(pid int) bool {
	proc, err := os.FindProcess(pid)
	if err != nil {
		return false
	}
	if runtime.GOOS == "windows" {
		return true
	}
	return proc.Signal(syscall.Signal(0)) == nil
}

// controlPIDPath is where `surd connect` parks its PID for `surd stop`
// (and the desktop widget) to read.  Sibling of credentials.json on each
// platform.  Empty string means we can't write one (no HOME), in which
// case stop falls back to "Ctrl-C the foreground process".
func controlPIDPath() string {
	p := credsPath()
	if p == "" {
		return ""
	}
	return filepath.Join(filepath.Dir(p), "connect.pid")
}

func cmdStop(args []string) error {
	fs := flag.NewFlagSet("stop", flag.ExitOnError)
	jsonOut := fs.Bool("json", false, "machine-readable output")
	fs.Parse(args)

	pidPath := controlPIDPath()
	report := func(state string, extra map[string]any) {
		if !*jsonOut {
			return
		}
		o := map[string]any{"state": state}
		for k, v := range extra {
			o[k] = v
		}
		_ = json.NewEncoder(os.Stdout).Encode(o)
	}

	raw, err := os.ReadFile(pidPath)
	if err != nil {
		report("not_running", nil)
		if !*jsonOut {
			fmt.Println("no running surd connect found.")
		}
		return nil
	}
	pid := 0
	_, _ = fmt.Sscanf(strings.TrimSpace(string(raw)), "%d", &pid)
	if pid <= 0 {
		_ = os.Remove(pidPath)
		report("stale_pid_file", nil)
		return nil
	}
	proc, err := os.FindProcess(pid)
	if err != nil {
		report("not_running", nil)
		return nil
	}
	// SIGTERM gives dist-node time to unregister cleanly.  On Windows
	// os.FindProcess+Signal isn't supported for graceful termination, so
	// we fall back to Kill (which surfaces as an abrupt disconnect — the
	// server's reaper handles it).
	if runtime.GOOS == "windows" {
		err = proc.Kill()
	} else {
		err = proc.Signal(syscall.SIGTERM)
	}
	if err != nil {
		report("error", map[string]any{"error": err.Error()})
		if !*jsonOut {
			fmt.Fprintln(os.Stderr, "surd: signal pid", pid, ":", err)
		}
		return err
	}
	report("stopped", map[string]any{"pid": pid})
	if !*jsonOut {
		fmt.Println("  ✓ sent SIGTERM to surd connect (pid", pid, ")")
	}
	return nil
}

func cmdPools(args []string) error {
	fs := flag.NewFlagSet("pools", flag.ExitOnError)
	jsonOut := fs.Bool("json", false, "machine-readable output")
	fs.Parse(args)
	creds, err := loadCreds()
	if err != nil {
		return errors.New("not logged in — run `surd login`")
	}
	req, _ := http.NewRequest("GET",
		strings.TrimRight(creds.Server, "/")+"/api/agent/pools", nil)
	req.Header.Set("X-Agent-Key", creds.AgentKey)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		return fmt.Errorf("%d %s", resp.StatusCode, raw)
	}
	if *jsonOut {
		os.Stdout.Write(raw)
		return nil
	}
	var d struct {
		Pools []struct {
			ID      int64  `json:"id"`
			Name    string `json:"name"`
			Slug    string `json:"slug"`
			BaseURL string `json:"base_url"`
		} `json:"pools"`
	}
	if err := json.Unmarshal(raw, &d); err != nil {
		return err
	}
	if len(d.Pools) == 0 {
		fmt.Println("(no pools — visit", creds.Server, "to create one)")
		return nil
	}
	for _, p := range d.Pools {
		fmt.Printf("  %-24s  %s\n", p.Name+" ("+p.Slug+")", p.BaseURL)
	}
	return nil
}

// share prints a pool-invite link.  Useful for sysadmins sharing access
// quickly via Slack/SMS without making the recipient log into the web UI.
func cmdShare(args []string) error {
	fs := flag.NewFlagSet("share", flag.ExitOnError)
	pool := fs.String("pool", "", "pool id or slug")
	jsonOut := fs.Bool("json", false, "machine-readable output")
	fs.Parse(args)
	if *pool == "" {
		return errors.New("--pool is required")
	}
	creds, err := loadCreds()
	if err != nil {
		return errors.New("not logged in — run `surd login`")
	}
	// We don't currently have an agent-key authenticated invite endpoint
	// (only the session-cookie one), so we point the user at the dashboard.
	url := strings.TrimRight(creds.Server, "/") + "/pools/" + *pool + "?share=1"
	if *jsonOut {
		_ = json.NewEncoder(os.Stdout).Encode(map[string]string{"url": url})
		return nil
	}
	fmt.Println("share:", url)
	fmt.Println("(open in browser and copy the invite link printed there)")
	return nil
}

func cmdLogout(args []string) error {
	if err := os.Remove(credsPath()); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	fmt.Println("logged out — credentials removed.")
	return nil
}

func cmdUpdate(args []string) error {
	creds, _ := loadCreds()
	server := defaultServer()
	if creds != nil {
		server = creds.Server
	}
	if server == "" {
		return errors.New("no server configured — set SURD_SERVER")
	}
	resp, err := http.Get(strings.TrimRight(server, "/") + "/api/install_targets")
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		return fmt.Errorf("update check: %d %s", resp.StatusCode, raw)
	}
	fmt.Println("available targets:")
	fmt.Println(string(raw))
	fmt.Println()
	fmt.Println("(auto-update will land in a later release;")
	fmt.Println(" for now re-run the install snippet from", server, ")")
	return nil
}

// ── daemon ───────────────────────────────────────────────────────────────
// `surd daemon install|uninstall|status` registers `surd connect` as a
// user-level service that auto-starts at login.  Per-platform backend:
//
//   linux   — systemd user unit at ~/.config/systemd/user/surd.service
//   darwin  — launchd LaunchAgent at ~/Library/LaunchAgents/com.surd.connect.plist
//   windows — scheduled task surd-connect via schtasks
//
// We deliberately avoid system-level installation (root/Administrator).
// A user can always promote with their own tooling if they need it.

const (
	daemonServiceID  = "com.surd.connect"
	daemonLabelShort = "surd-connect"
)

func cmdDaemon(args []string) error {
	if len(args) == 0 {
		fmt.Print(daemonUsage)
		return nil
	}
	sub, rest := args[0], args[1:]
	switch sub {
	case "install":
		return daemonInstall(rest)
	case "uninstall", "remove":
		return daemonUninstall(rest)
	case "status":
		return daemonStatus(rest)
	case "help", "-h", "--help":
		fmt.Print(daemonUsage)
		return nil
	default:
		fmt.Fprintln(os.Stderr, "surd daemon: unknown subcommand", sub)
		fmt.Fprint(os.Stderr, daemonUsage)
		os.Exit(2)
	}
	return nil
}

const daemonUsage = `surd daemon — auto-start surd connect at login

USAGE
  surd daemon install    register and start the service
  surd daemon uninstall  stop and unregister
  surd daemon status     show whether the service is running
`

// surdBinaryPath resolves to the absolute path of the running binary so
// the service definition survives PATH changes.
func surdBinaryPath() (string, error) {
	exe, err := os.Executable()
	if err != nil {
		return "", err
	}
	abs, err := filepath.Abs(exe)
	if err != nil {
		return exe, nil
	}
	return abs, nil
}

func daemonInstall(_ []string) error {
	exe, err := surdBinaryPath()
	if err != nil {
		return err
	}
	if _, err := loadCreds(); err != nil {
		return errors.New("no credentials — run `surd login` first")
	}
	switch runtime.GOOS {
	case "linux":
		return systemdInstall(exe)
	case "darwin":
		return launchdInstall(exe)
	case "windows":
		return windowsTaskInstall(exe)
	default:
		return fmt.Errorf("daemon install: unsupported OS %s", runtime.GOOS)
	}
}

func daemonUninstall(_ []string) error {
	switch runtime.GOOS {
	case "linux":
		return systemdUninstall()
	case "darwin":
		return launchdUninstall()
	case "windows":
		return windowsTaskUninstall()
	default:
		return fmt.Errorf("daemon uninstall: unsupported OS %s", runtime.GOOS)
	}
}

func daemonStatus(_ []string) error {
	switch runtime.GOOS {
	case "linux":
		return runStream("systemctl", "--user", "status", daemonLabelShort+".service")
	case "darwin":
		return runStream("launchctl", "print", "gui/"+fmt.Sprint(os.Getuid())+"/"+daemonServiceID)
	case "windows":
		return runStream("schtasks", "/Query", "/TN", daemonLabelShort, "/V", "/FO", "LIST")
	default:
		return fmt.Errorf("daemon status: unsupported OS %s", runtime.GOOS)
	}
}

func runStream(name string, args ...string) error {
	cmd := exec.Command(name, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

// ── linux / systemd ──

func systemdUnitPath() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".config", "systemd", "user", daemonLabelShort+".service")
}

func systemdInstall(exe string) error {
	unit := fmt.Sprintf(`[Unit]
Description=surd — distributed inference rig
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=%s connect
Restart=on-failure
RestartSec=5
# Don't drown the journal on flapping bootstrap failures.
StartLimitIntervalSec=120
StartLimitBurst=10

[Install]
WantedBy=default.target
`, exe)

	p := systemdUnitPath()
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(p, []byte(unit), 0o644); err != nil {
		return err
	}
	fmt.Println("  ▸ wrote", p)
	if err := runStream("systemctl", "--user", "daemon-reload"); err != nil {
		return err
	}
	if err := runStream("systemctl", "--user", "enable", "--now", daemonLabelShort+".service"); err != nil {
		return err
	}
	fmt.Println("  ✓ surd is now running as a user systemd service")
	fmt.Println("    logs:  journalctl --user -u", daemonLabelShort+".service", "-f")
	return nil
}

func systemdUninstall() error {
	_ = runStream("systemctl", "--user", "disable", "--now", daemonLabelShort+".service")
	if err := os.Remove(systemdUnitPath()); err != nil && !os.IsNotExist(err) {
		return err
	}
	_ = runStream("systemctl", "--user", "daemon-reload")
	fmt.Println("  ✓ systemd unit removed")
	return nil
}

// ── darwin / launchd ──

func launchdPlistPath() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, "Library", "LaunchAgents", daemonServiceID+".plist")
}

func launchdInstall(exe string) error {
	home, _ := os.UserHomeDir()
	logDir := filepath.Join(home, "Library", "Logs", "surd")
	_ = os.MkdirAll(logDir, 0o755)
	plist := fmt.Sprintf(`<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>%s</string>
  <key>ProgramArguments</key>
  <array>
    <string>%s</string>
    <string>connect</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key>
  <dict><key>SuccessfulExit</key><false/></dict>
  <key>StandardOutPath</key><string>%s/surd.out.log</string>
  <key>StandardErrorPath</key><string>%s/surd.err.log</string>
</dict>
</plist>
`, daemonServiceID, exe, logDir, logDir)

	p := launchdPlistPath()
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(p, []byte(plist), 0o644); err != nil {
		return err
	}
	fmt.Println("  ▸ wrote", p)
	uid := fmt.Sprint(os.Getuid())
	// `bootstrap` is the modern equivalent of `load -w`; on older macOS we
	// fall back if it isn't recognised.
	if err := runStream("launchctl", "bootstrap", "gui/"+uid, p); err != nil {
		if err2 := runStream("launchctl", "load", "-w", p); err2 != nil {
			return fmt.Errorf("launchctl bootstrap/load failed: %v", err)
		}
	}
	_ = runStream("launchctl", "enable", "gui/"+uid+"/"+daemonServiceID)
	fmt.Println("  ✓ surd LaunchAgent installed and started")
	fmt.Println("    logs:", logDir)
	return nil
}

func launchdUninstall() error {
	p := launchdPlistPath()
	uid := fmt.Sprint(os.Getuid())
	_ = runStream("launchctl", "bootout", "gui/"+uid+"/"+daemonServiceID)
	_ = runStream("launchctl", "unload", p)
	if err := os.Remove(p); err != nil && !os.IsNotExist(err) {
		return err
	}
	fmt.Println("  ✓ LaunchAgent removed")
	return nil
}

// ── windows / schtasks ──

func windowsTaskInstall(exe string) error {
	// Run at login of the current user, restart on failure.  We use ONLOGON
	// rather than a Windows Service to keep it user-scope (no Admin needed).
	args := []string{
		"/Create", "/F",
		"/SC", "ONLOGON",
		"/TN", daemonLabelShort,
		"/TR", fmt.Sprintf(`"%s" connect`, exe),
		"/RL", "LIMITED",
	}
	if err := runStream("schtasks", args...); err != nil {
		return err
	}
	if err := runStream("schtasks", "/Run", "/TN", daemonLabelShort); err != nil {
		// Not fatal — task is installed, just couldn't start now.
		fmt.Fprintln(os.Stderr, "surd: warning — schtasks /Run failed:", err)
	}
	fmt.Println("  ✓ scheduled task", daemonLabelShort, "installed")
	return nil
}

func windowsTaskUninstall() error {
	if err := runStream("schtasks", "/Delete", "/F", "/TN", daemonLabelShort); err != nil {
		return err
	}
	fmt.Println("  ✓ scheduled task removed")
	return nil
}

// ── helpers ──────────────────────────────────────────────────────────────

func defaultServer() string {
	if s := os.Getenv("SURD_SERVER"); s != "" {
		return s
	}
	// If we've logged in before, fall through to the saved value.
	if c, err := loadCreds(); err == nil {
		return c.Server
	}
	return ""
}

// tryOpenBrowser is best-effort — we never error out the login flow if
// the browser launcher fails (containers, headless, etc).
func tryOpenBrowser(url string) error {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "darwin":
		cmd = exec.Command("open", url)
	case "windows":
		cmd = exec.Command("rundll32", "url.dll,FileProtocolHandler", url)
	default:
		cmd = exec.Command("xdg-open", url)
	}
	return cmd.Start()
}
