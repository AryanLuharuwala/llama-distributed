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
	"net/http"
	"os"
	"os/signal"
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
	}
	flag.StringVar(&c.addr, "addr", c.addr, "listen address")
	flag.StringVar(&c.dbPath, "db", c.dbPath, "SQLite path")
	flag.StringVar(&c.publicURL, "public-url", c.publicURL, "public URL for deep links")
	flag.StringVar(&c.apexHost, "apex", c.apexHost, "apex domain — dashboard here, <slug>.apex for pool endpoints")
	flag.StringVar(&c.modelsDir, "models-dir", c.modelsDir, "dir to store model shards")
	flag.StringVar(&c.splitterBin, "splitter", c.splitterBin, "path to llama-split-gguf")
	flag.BoolVar(&c.devMode, "dev", c.githubClient == "", "dev mode: enable /auth/dev endpoint")
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

	db, err := sql.Open("sqlite3", cfg.dbPath+"?_journal_mode=WAL&_foreign_keys=on")
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

	go func() {
		log.Printf("listening on %s (public %s, dev=%v)",
			cfg.addr, cfg.publicURL, cfg.devMode)
		if err := httpSrv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("listen: %v", err)
		}
	}()

	<-ctx.Done()
	shutdown, c2 := context.WithTimeout(context.Background(), 5*time.Second)
	defer c2()
	_ = httpSrv.Shutdown(shutdown)
	fmt.Println("bye")
}
