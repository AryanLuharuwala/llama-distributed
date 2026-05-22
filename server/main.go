// llama-distributed control-plane server.
//
// Responsibilities (Milestone 1):
//   - Serve the single-page UI (./ui.html, embedded).
//   - GitHub OAuth login flow → session cookie.
//   - Mint short-lived pairing tokens that the browser embeds in a
//     distpool:// deep link.  The native agent, when launched with that
//     link, connects back over WebSocket and presents the token.
//   - Two WebSocket endpoints:
//       /ws/browser — the logged-in UI subscribes to its own rigs' events
//       /ws/agent   — the native agent authenticates via pairing token,
//                     streams status and receives commands
//   - SQLite store for users, rigs, and audit events.
package main

import (
	"context"
	"database/sql"
	"embed"
	"errors"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"runtime"
	"strings"
	"syscall"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

//go:embed ui.html
var uiFS embed.FS

type config struct {
	addr          string
	dbPath        string
	publicURL     string // e.g. http://localhost:8080 — used in deep links
	apexHost      string // e.g. surds.co.in — dashboard here, <slug>.apex for pools
	devMode       bool   // enables a "dev login" endpoint (no GitHub needed)
	githubClient  string
	githubSecret  string
	sessionSecret string
	modelsDir     string // root dir where model shards live
	splitterBin   string // path to llama-split-gguf binary
	releasesDir   string // on-disk cache for GitHub release tarballs
}

func loadConfig() config {
	c := config{
		addr:          envOr("DIST_ADDR", ":8080"),
		dbPath:        envOr("DIST_DB", "distpool.sqlite"),
		publicURL:     envOr("DIST_PUBLIC_URL", "http://localhost:8080"),
		apexHost:      envOr("DIST_APEX_HOST", "surds.co.in"),
		githubClient:  os.Getenv("DIST_GITHUB_CLIENT"),
		githubSecret:  os.Getenv("DIST_GITHUB_SECRET"),
		sessionSecret: envOr("DIST_SESSION_SECRET", "dev-secret-change-me"),
		modelsDir:     envOr("DIST_MODELS_DIR", "./models-store"),
		splitterBin:   envOr("DIST_SPLITTER", "/home/boom/startup/Project/llama.cpp/build/bin/llama-split-gguf"),
		releasesDir:   envOr("DIST_RELEASES_DIR", "./releases-cache"),
	}
	flag.StringVar(&c.addr, "addr", c.addr, "listen address")
	flag.StringVar(&c.dbPath, "db", c.dbPath, "SQLite path")
	flag.StringVar(&c.publicURL, "public-url", c.publicURL, "public URL for deep links")
	flag.StringVar(&c.apexHost, "apex", c.apexHost, "apex domain — dashboard here, <slug>.apex for pool endpoints")
	flag.StringVar(&c.modelsDir, "models-dir", c.modelsDir, "dir to store model shards")
	flag.StringVar(&c.splitterBin, "splitter", c.splitterBin, "path to llama-split-gguf")
	flag.StringVar(&c.releasesDir, "releases-dir", c.releasesDir, "on-disk cache dir for release tarballs")
	// DIST_DEV_MODE=1/true forces /auth/dev on even when GitHub OAuth is
	// configured — useful for first-boot smoke tests on a deployment where
	// real OAuth isn't wired up yet.
	devDefault := c.githubClient == ""
	if v := strings.ToLower(os.Getenv("DIST_DEV_MODE")); v == "1" || v == "true" || v == "yes" {
		devDefault = true
	}
	flag.BoolVar(&c.devMode, "dev", devDefault, "dev mode: enable /auth/dev endpoint")
	flag.Parse()
	return c
}

func envOr(k, d string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return d
}

func main() {
	cfg := loadConfig()

	// SQLite WAL is broken over CIFS (Azure Files mounts /data via CIFS),
	// so the journal mode must be overridable. Default stays WAL for local
	// disk; set DIST_SQLITE_JOURNAL_MODE=DELETE on CIFS-backed hosts.
	journalMode := envOr("DIST_SQLITE_JOURNAL_MODE", "WAL")
	db, err := sql.Open("sqlite3", cfg.dbPath+"?_journal_mode="+journalMode+"&_foreign_keys=on")
	if err != nil {
		log.Fatalf("open db: %v", err)
	}
	defer db.Close()
	if err := migrate(db); err != nil {
		log.Fatalf("migrate: %v", err)
	}

	srv := newServer(cfg, db)
	if err := srv.backfillSlugs(); err != nil {
		log.Fatalf("backfill slugs: %v", err)
	}
	httpSrv := &http.Server{
		Addr:              cfg.addr,
		Handler:           srv.router(),
		ReadHeaderTimeout: 10 * time.Second,
	}

	ctx, cancel := signal.NotifyContext(context.Background(),
		syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	go srv.reapLoop(ctx)
	go refreshTargets(srv)

	go func() {
		log.Printf("listening on %s (public %s, dev=%v)",
			cfg.addr, cfg.publicURL, cfg.devMode)
		if err := httpSrv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("listen: %v", err)
		}
	}()

	// Firewall pre-flight: once the listener is up, try reaching /healthz
	// over every non-loopback interface IP.  If any fail, print a concrete
	// hint so the user can open the port before rigs fail to pair.
	go firewallPreflight(cfg.addr)

	<-ctx.Done()
	shutdown, c2 := context.WithTimeout(context.Background(), 5*time.Second)
	defer c2()
	_ = httpSrv.Shutdown(shutdown)
	fmt.Println("bye")
}

// firewallProbeResult is the shape returned by firewallProbe() and the
// dashboard's GET /api/firewall_check handler.
type firewallProbeResult struct {
	Port      string   `json:"port"`
	Reachable []string `json:"reachable"`
	Failed    []string `json:"failed"`
	Hint      string   `json:"hint,omitempty"`
}

// firewallProbe dials /healthz on every non-loopback interface IP.  Shared by
// the boot-time pre-flight and the on-demand dashboard button.
func firewallProbe(addr string) firewallProbeResult {
	res := firewallProbeResult{Port: ""}
	_, port, err := net.SplitHostPort(addr)
	if err != nil || port == "" {
		return res
	}
	res.Port = port
	ifaces, err := net.InterfaceAddrs()
	if err != nil {
		return res
	}

	for _, a := range ifaces {
		ipnet, ok := a.(*net.IPNet)
		if !ok || ipnet.IP.IsLoopback() || ipnet.IP.IsLinkLocalUnicast() {
			continue
		}
		ip := ipnet.IP
		if ip4 := ip.To4(); ip4 != nil {
			ip = ip4
		} else {
			continue
		}
		target := net.JoinHostPort(ip.String(), port)
		c, err := net.DialTimeout("tcp", target, 750*time.Millisecond)
		if err != nil {
			res.Failed = append(res.Failed, target)
			continue
		}
		_ = c.Close()
		res.Reachable = append(res.Reachable, target)
	}

	if len(res.Failed) > 0 {
		switch runtime.GOOS {
		case "linux":
			res.Hint = fmt.Sprintf(
				"sudo firewall-cmd --add-port=%s/tcp --permanent && sudo firewall-cmd --reload   "+
					"(or: sudo ufw allow %s/tcp)", port, port)
		case "darwin":
			res.Hint = "macOS: System Settings → Network → Firewall → Options → allow dist-server"
		case "windows":
			res.Hint = fmt.Sprintf(
				"New-NetFirewallRule -DisplayName 'distpool' -Direction Inbound -LocalPort %s -Protocol TCP -Action Allow",
				port)
		}
	}
	return res
}

// firewallPreflight runs the probe once at boot and logs the result.  The
// on-demand button from the dashboard uses firewallProbe directly.
func firewallPreflight(addr string) {
	time.Sleep(500 * time.Millisecond)
	res := firewallProbe(addr)
	if len(res.Reachable) > 0 {
		log.Printf("firewall pre-flight: reachable on %s", strings.Join(res.Reachable, ", "))
	}
	if len(res.Failed) == 0 {
		return
	}
	log.Printf("firewall pre-flight: NOT reachable on %s — other laptops won't be able to pair",
		strings.Join(res.Failed, ", "))
	if res.Hint != "" {
		log.Printf("firewall pre-flight: open the port with: %s", res.Hint)
	}
}
