// dist-turn — a bundled TURN server with two operating modes.
//
// SIDECAR MODE (default when --auth-secret is given):
//   Started by dist-node on relay-capable rigs.  Takes the per-rig
//   secret over the command line and runs a dumb TURN forwarder.
//
//     dist-turn --auth-secret=<hex> [--listen=…] [--realm=…] [--external-ip=…]
//
// STANDALONE MODE (when --server is given):
//   Self-contained TURN-only node — no GPU, no model, no inference.
//   Pairs with dist-server over WebSocket like any other agent, but
//   advertises relay_only=true.  The server returns a per-rig
//   turn_secret in the welcome frame, which the node then uses to
//   authenticate TURN traffic.  Heartbeats coturn_port + public_ip
//   so the planner picks this rig up automatically.  Persists
//   agent.id / agent.key under --state-dir so restarts resume
//   without a fresh pair token.
//
//     dist-turn --server=https://pool.example.com --token=<pair-token>
//
// Provision flow for a dedicated relay box:
//   1. Operator generates a one-liner / pair token from the dashboard.
//   2. SSH to the relay, run `dist-turn --server=… --token=…`.
//   3. Open UDP/3478 (or --listen=) in the host firewall.
//   4. The relay shows up in the swarm dashboard as a TURN-only rig
//      and is automatically used for symmetric-NAT pairs.
//
// Both modes use pion/turn (Go, MIT) so the same binary builds for
// Linux, macOS, and Windows without an external coturn dependency.
// TURN traffic is UDP-forwarded; the A↔C DTLS handshake happens
// end-to-end through the allocation and the relay never sees plaintext.
//
// Auth follows the coturn REST scheme (use-auth-secret):
//   username   = "<unix-expiry>:<audience>"
//   credential = base64(HMAC-SHA1(<auth-secret>, username))
//
// Logs go to stderr; exits 0 on SIGINT/SIGTERM.

package main

import (
	"context"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha1"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/coder/websocket"
	stun "github.com/pion/stun/v2"
	"github.com/pion/turn/v3"
)

func main() {
	listen := flag.String("listen", "0.0.0.0:3478",
		"UDP listen address (host:port) for the TURN server")
	realm := flag.String("realm", "dist", "TURN realm advertised to clients")
	secret := flag.String("auth-secret", "",
		"[sidecar mode] HMAC secret matching dist-server's mintRigTURNCreds")
	extIP := flag.String("external-ip", "",
		"Public IPv4 to advertise in allocations (auto-discovered in standalone mode when omitted)")

	// Standalone-mode flags.
	server := flag.String("server", "",
		"[standalone mode] dist-server URL to register with (e.g. https://pool.example.com)")
	token := flag.String("token", "",
		"[standalone mode] pair token from the dashboard (first run only)")
	stateDir := flag.String("state-dir", defaultStateDir(),
		"[standalone mode] directory for agent.id / agent.key persistence")
	hostnameOverride := flag.String("hostname", "",
		"[standalone mode] hostname advertised in heartbeat (default: os.Hostname)")

	flag.Parse()

	// Dispatch on mode.
	if *server != "" {
		if *secret != "" {
			fmt.Fprintln(os.Stderr,
				"dist-turn: --auth-secret and --server are mutually exclusive")
			os.Exit(2)
		}
		runStandalone(*server, *token, *stateDir, *listen, *realm,
			*extIP, *hostnameOverride)
		return
	}
	if *secret == "" {
		fmt.Fprintln(os.Stderr,
			"dist-turn: pass either --auth-secret (sidecar) or --server (standalone)")
		flag.Usage()
		os.Exit(2)
	}
	runSidecar(*listen, *realm, *secret, *extIP)
}

// ─── Sidecar mode ──────────────────────────────────────────────────────────

func runSidecar(listenAddr, realm, secret, extIP string) {
	srv, conns, err := startTurn(listenAddr, realm, secret, extIP)
	if err != nil {
		fmt.Fprintf(os.Stderr, "dist-turn: %v\n", err)
		os.Exit(1)
	}
	logListeners(conns, realm, extIP)
	waitForExit()
	if err := srv.Close(); err != nil {
		fmt.Fprintf(os.Stderr, "dist-turn: close: %v\n", err)
	}
}

// ─── Standalone mode ───────────────────────────────────────────────────────

func runStandalone(serverURL, token, stateDir, listenAddr, realmFlag, extIP,
	hostnameOverride string) {

	if err := os.MkdirAll(stateDir, 0o700); err != nil {
		fmt.Fprintf(os.Stderr, "dist-turn: cannot create state-dir %s: %v\n",
			stateDir, err)
		os.Exit(1)
	}
	agentID, _ := readState(stateDir, "agent.id")
	agentKey, _ := readState(stateDir, "agent.key")
	if agentID == "" {
		// Mint a fresh agent_id locally — server reuses ours as-is.
		var buf [16]byte
		_, _ = rand.Read(buf[:])
		agentID = "relay-" + hex.EncodeToString(buf[:])
		_ = writeState(stateDir, "agent.id", agentID)
	}

	hostname := hostnameOverride
	if hostname == "" {
		hostname, _ = os.Hostname()
		if hostname == "" {
			hostname = "dist-turn"
		}
	}

	// Auto-discover public IP via STUN if the operator didn't override.
	if extIP == "" {
		if ip := discoverPublicIP(3 * time.Second); ip != "" {
			extIP = ip
			fmt.Fprintf(os.Stderr,
				"dist-turn: discovered public_ip=%s via STUN\n", extIP)
		} else {
			fmt.Fprintln(os.Stderr,
				"dist-turn: warning — could not auto-discover public IP; "+
					"clients may see allocations on a non-routable address")
		}
	}

	// Reconnect loop.  Each iteration runs one pair-or-resume cycle plus a
	// heartbeat loop until the WS drops.  We keep the TURN listener up
	// across reconnects when the secret hasn't changed.
	var (
		turnSrv     *turn.Server
		turnSecret  string
		turnRealm   = realmFlag
		turnPort    int
		turnMu      sync.Mutex
		stop        = make(chan struct{})
		sig         = make(chan os.Signal, 1)
		ctx, cancel = context.WithCancel(context.Background())
	)
	defer cancel()
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	go func() { <-sig; close(stop); cancel() }()

	backoff := time.Second
	for {
		select {
		case <-stop:
			turnMu.Lock()
			if turnSrv != nil {
				_ = turnSrv.Close()
			}
			turnMu.Unlock()
			return
		default:
		}

		err := runOnePairCycle(ctx, pairConfig{
			ServerURL:    serverURL,
			Token:        token,
			StateDir:     stateDir,
			AgentID:      agentID,
			AgentKey:     agentKey,
			Hostname:     hostname,
			ListenAddr:   listenAddr,
			DefaultRealm: turnRealm,
			ExtIP:        extIP,
		}, &turnMu, &turnSrv, &turnSecret, &turnRealm, &turnPort)

		if err != nil {
			fmt.Fprintf(os.Stderr, "dist-turn: cycle failed: %v\n", err)
			// Reload state in case the cycle persisted a fresh agent_key.
			agentKey, _ = readState(stateDir, "agent.key")
			select {
			case <-stop:
				return
			case <-time.After(backoff):
			}
			if backoff < 30*time.Second {
				backoff = backoff*2 + time.Second/2
			}
			continue
		}
		backoff = time.Second
		// runOnePairCycle returned nil → graceful WS close on our side
		// (rare); loop and re-pair.
		agentKey, _ = readState(stateDir, "agent.key")
	}
}

// ─── One pair-or-resume cycle ──────────────────────────────────────────────

type pairConfig struct {
	ServerURL, Token, StateDir, AgentID, AgentKey, Hostname string
	ListenAddr, DefaultRealm, ExtIP                         string
}

func runOnePairCycle(ctx context.Context, cfg pairConfig,
	turnMu *sync.Mutex,
	turnSrv **turn.Server,
	turnSecret *string,
	turnRealm *string,
	turnPort *int) error {

	wsURL, err := httpToWS(cfg.ServerURL)
	if err != nil {
		return fmt.Errorf("server URL: %w", err)
	}
	wsURL = strings.TrimRight(wsURL, "/") + "/api/agent"

	dialCtx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()
	conn, _, err := websocket.Dial(dialCtx, wsURL, nil)
	if err != nil {
		return fmt.Errorf("ws dial: %w", err)
	}
	defer conn.CloseNow()

	// 1 MiB cap is plenty for the welcome / heartbeat acks.
	conn.SetReadLimit(1 << 20)

	hello := buildHello(cfg)
	if err := wsWriteJSON(ctx, conn, hello); err != nil {
		return fmt.Errorf("send hello: %w", err)
	}

	// Read welcome.  The server may issue a challenge for signed agents,
	// but a freshly-minted relay node has no keypair so it'll be welcomed
	// directly (legacy/key-only auth path on the server).
	welcome, err := wsReadJSON(ctx, conn, 5*time.Second)
	if err != nil {
		return fmt.Errorf("read welcome: %w", err)
	}
	if kind, _ := welcome["kind"].(string); kind != "welcome" {
		return fmt.Errorf("unexpected first frame: kind=%v body=%v",
			kind, welcome)
	}
	if k, _ := welcome["agent_key"].(string); k != "" && cfg.AgentKey == "" {
		_ = writeState(cfg.StateDir, "agent.key", k)
		_ = writeState(cfg.StateDir, "agent.server", cfg.ServerURL)
	}
	newSecret, _ := welcome["turn_secret"].(string)
	newRealm, _ := welcome["turn_realm"].(string)
	if newRealm == "" {
		newRealm = cfg.DefaultRealm
	}
	if newSecret == "" {
		return errors.New("server did not provide turn_secret (DIST_TURN_SECRET unset on server?)")
	}

	// Start or restart the TURN server if the credentials changed.
	turnMu.Lock()
	needRestart := *turnSrv == nil || *turnSecret != newSecret || *turnRealm != newRealm
	if needRestart {
		if *turnSrv != nil {
			_ = (*turnSrv).Close()
			*turnSrv = nil
		}
		srv, conns, err := startTurn(cfg.ListenAddr, newRealm, newSecret, cfg.ExtIP)
		if err != nil {
			turnMu.Unlock()
			return fmt.Errorf("start turn: %w", err)
		}
		logListeners(conns, newRealm, cfg.ExtIP)
		*turnSrv = srv
		*turnSecret = newSecret
		*turnRealm = newRealm
		_, portStr, _ := net.SplitHostPort(cfg.ListenAddr)
		*turnPort, _ = strconv.Atoi(portStr)
	}
	port := *turnPort
	turnMu.Unlock()

	fmt.Fprintf(os.Stderr,
		"dist-turn: registered as agent_id=%s, advertising coturn_port=%d\n",
		cfg.AgentID, port)

	// Heartbeat + read loop.  We use a write mutex so we can fire status
	// frames concurrently with read-loop drainage.
	var writeMu sync.Mutex
	send := func(v any) error {
		writeMu.Lock()
		defer writeMu.Unlock()
		return wsWriteJSON(ctx, conn, v)
	}

	readErr := make(chan error, 1)
	go func() {
		// Drain any frames the server sends.  The relay node doesn't act
		// on inference, comfy, or pp_route messages — we just keep the
		// socket healthy so the server doesn't garbage-collect us.
		for {
			_, _, err := conn.Read(ctx)
			if err != nil {
				readErr <- err
				return
			}
		}
	}()

	tick := time.NewTicker(5 * time.Second)
	defer tick.Stop()
	// Send the first status immediately so the planner sees us within ms,
	// not the full 5s tick.
	if err := send(buildStatus(cfg, port)); err != nil {
		return fmt.Errorf("send first status: %w", err)
	}
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case err := <-readErr:
			return fmt.Errorf("ws read: %w", err)
		case <-tick.C:
			if err := send(buildStatus(cfg, port)); err != nil {
				return fmt.Errorf("send status: %w", err)
			}
		}
	}
}

func buildHello(cfg pairConfig) map[string]any {
	h := map[string]any{
		"agent_id":   cfg.AgentID,
		"hostname":   cfg.Hostname,
		"n_gpus":     0,
		"vram_bytes": 0,
	}
	if cfg.AgentKey != "" {
		h["kind"] = "resume"
		h["agent_key"] = cfg.AgentKey
	} else {
		h["kind"] = "hello"
		h["token"] = cfg.Token
	}
	return h
}

func buildStatus(cfg pairConfig, coturnPort int) map[string]any {
	return map[string]any{
		"kind":          "status",
		"n_gpus":        0,
		"tokens_sec":    0,
		"uptime_sec":    int64(time.Since(processStart).Seconds()),
		"inflight":      0,
		"nat_type":      "open", // a registered relay box is, by definition, reachable
		"relay_capable": true,
		"coturn_port":   coturnPort,
		"public_ip":     cfg.ExtIP,
		"roles":         []string{"turn_relay"},
	}
}

var processStart = time.Now()

// ─── TURN listener helper (shared by both modes) ──────────────────────────

func startTurn(listenAddr, realm, secret, extIP string,
) (*turn.Server, []net.PacketConn, error) {

	host, portStr, err := net.SplitHostPort(listenAddr)
	if err != nil {
		return nil, nil, fmt.Errorf("bad listen %q: %v", listenAddr, err)
	}
	port, err := strconv.Atoi(portStr)
	if err != nil || port <= 0 || port > 65535 {
		return nil, nil, fmt.Errorf("bad port in %q", listenAddr)
	}

	var packetConns []net.PacketConn
	wildcard := host == "" || host == "0.0.0.0" || host == "::"
	if wildcard {
		if pc, err := net.ListenPacket("udp4",
			net.JoinHostPort("0.0.0.0", portStr)); err == nil {
			packetConns = append(packetConns, pc)
		} else {
			fmt.Fprintf(os.Stderr, "dist-turn: udp4 bind skipped: %v\n", err)
		}
		if pc, err := net.ListenPacket("udp6",
			net.JoinHostPort("::", portStr)); err == nil {
			packetConns = append(packetConns, pc)
		} else {
			fmt.Fprintf(os.Stderr, "dist-turn: udp6 bind skipped: %v\n", err)
		}
	} else {
		pc, err := net.ListenPacket("udp", net.JoinHostPort(host, portStr))
		if err != nil {
			return nil, nil, fmt.Errorf("bind %s: %v", listenAddr, err)
		}
		packetConns = append(packetConns, pc)
	}
	if len(packetConns) == 0 {
		return nil, nil, fmt.Errorf("no UDP listener could be opened on port %d",
			port)
	}

	var relayGen turn.RelayAddressGenerator
	if extIP != "" {
		ip := net.ParseIP(extIP)
		if ip == nil {
			for _, pc := range packetConns {
				_ = pc.Close()
			}
			return nil, nil, fmt.Errorf("bad --external-ip %q", extIP)
		}
		relayGen = &turn.RelayAddressGeneratorStatic{
			RelayAddress: ip,
			Address:      "0.0.0.0",
		}
	} else {
		relayGen = &turn.RelayAddressGeneratorNone{Address: "0.0.0.0"}
	}

	authHandler := func(username, _ string, _ net.Addr) ([]byte, bool) {
		colon := strings.IndexByte(username, ':')
		if colon <= 0 {
			return nil, false
		}
		expStr := username[:colon]
		exp, err := strconv.ParseInt(expStr, 10, 64)
		if err != nil {
			return nil, false
		}
		if time.Now().Unix() > exp {
			return nil, false
		}
		mac := hmac.New(sha1.New, []byte(secret))
		mac.Write([]byte(username))
		cred := base64.StdEncoding.EncodeToString(mac.Sum(nil))
		return turn.GenerateAuthKey(username, realm, cred), true
	}

	pcCfgs := make([]turn.PacketConnConfig, 0, len(packetConns))
	for _, pc := range packetConns {
		pcCfgs = append(pcCfgs, turn.PacketConnConfig{
			PacketConn:            pc,
			RelayAddressGenerator: relayGen,
		})
	}
	srv, err := turn.NewServer(turn.ServerConfig{
		Realm:             realm,
		AuthHandler:       authHandler,
		PacketConnConfigs: pcCfgs,
	})
	if err != nil {
		for _, pc := range packetConns {
			_ = pc.Close()
		}
		return nil, nil, fmt.Errorf("server start: %v", err)
	}
	return srv, packetConns, nil
}

func logListeners(conns []net.PacketConn, realm, extIP string) {
	for _, pc := range conns {
		fmt.Fprintf(os.Stderr, "dist-turn: listening on %s (%s) realm=%s\n",
			pc.LocalAddr(), pc.LocalAddr().Network(), realm)
	}
	if extIP != "" {
		fmt.Fprintf(os.Stderr, "dist-turn: external-ip=%s\n", extIP)
	}
}

// ─── helpers ──────────────────────────────────────────────────────────────

func defaultStateDir() string {
	if d, err := os.UserConfigDir(); err == nil && d != "" {
		return filepath.Join(d, "dist-turn")
	}
	if runtime.GOOS == "windows" {
		return filepath.Join(os.Getenv("APPDATA"), "dist-turn")
	}
	return filepath.Join(os.Getenv("HOME"), ".dist-turn")
}

func readState(dir, name string) (string, error) {
	b, err := os.ReadFile(filepath.Join(dir, name))
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(b)), nil
}

func writeState(dir, name, value string) error {
	return os.WriteFile(filepath.Join(dir, name), []byte(value), 0o600)
}

func httpToWS(u string) (string, error) {
	parsed, err := url.Parse(u)
	if err != nil {
		return "", err
	}
	switch parsed.Scheme {
	case "http":
		parsed.Scheme = "ws"
	case "https":
		parsed.Scheme = "wss"
	case "ws", "wss":
		// already
	default:
		return "", fmt.Errorf("unsupported scheme %q", parsed.Scheme)
	}
	return parsed.String(), nil
}

// Cheap STUN client: ask Google's STUN for our srflx address.  Returns ""
// if we couldn't get a useful answer in the budget.  We try IPv4 first
// because dist-server only handles IPv4 TURN URLs today.
func discoverPublicIP(budget time.Duration) string {
	servers := []string{
		"stun.l.google.com:19302",
		"stun1.l.google.com:19302",
	}
	deadline := time.Now().Add(budget)
	for _, s := range servers {
		left := time.Until(deadline)
		if left <= 0 {
			break
		}
		if ip := stunQuery(s, left); ip != "" {
			return ip
		}
	}
	return ""
}

func stunQuery(server string, budget time.Duration) string {
	conn, err := net.DialTimeout("udp4", server, budget)
	if err != nil {
		return ""
	}
	defer conn.Close()
	_ = conn.SetDeadline(time.Now().Add(budget))

	msg, err := stun.Build(stun.TransactionID, stun.BindingRequest)
	if err != nil {
		return ""
	}
	if _, err := conn.Write(msg.Raw); err != nil {
		return ""
	}
	buf := make([]byte, 1500)
	n, err := conn.Read(buf)
	if err != nil {
		return ""
	}
	resp := &stun.Message{Raw: append([]byte(nil), buf[:n]...)}
	if err := resp.Decode(); err != nil {
		return ""
	}
	var xor stun.XORMappedAddress
	if err := xor.GetFrom(resp); err == nil && xor.IP != nil {
		if v4 := xor.IP.To4(); v4 != nil {
			return v4.String()
		}
		return xor.IP.String()
	}
	var mapped stun.MappedAddress
	if err := mapped.GetFrom(resp); err == nil && mapped.IP != nil {
		if v4 := mapped.IP.To4(); v4 != nil {
			return v4.String()
		}
		return mapped.IP.String()
	}
	return ""
}

// ─── tiny WS JSON helpers ─────────────────────────────────────────────────

func wsWriteJSON(ctx context.Context, c *websocket.Conn, v any) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	wctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	return c.Write(wctx, websocket.MessageText, b)
}

func wsReadJSON(ctx context.Context, c *websocket.Conn,
	timeout time.Duration) (map[string]any, error) {

	rctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	_, data, err := c.Read(rctx)
	if err != nil {
		return nil, err
	}
	var out map[string]any
	if err := json.Unmarshal(data, &out); err != nil {
		return nil, fmt.Errorf("unmarshal: %w (raw=%q)", err, truncate(data, 200))
	}
	return out, nil
}

func truncate(b []byte, n int) string {
	if len(b) <= n {
		return string(b)
	}
	return string(b[:n]) + "…"
}

func waitForExit() {
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	<-sig
}

// Silence unused-import warnings on build configurations where some helpers
// are stripped.  All currently used.
var _ = io.EOF
var _ = http.StatusOK
