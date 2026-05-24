// llama-distributed control-plane server.
//
// Responsibilities (Milestone 1):
//   - Serve the editorial sign-in / nexus / console / device pages.
//   - GitHub + Google OAuth login → session cookie.
//   - Device-code pairing so `dist-node login` can self-bind to a user.
//   - Two WebSocket endpoints:
//       /ws/browser — the logged-in UI subscribes to its own rigs' events
//       /ws/agent   — the native agent authenticates via pairing token,
//                     streams status and receives commands
//   - SQLite store for users, rigs, and audit events.
package main

import (
	"context"
	"database/sql"
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

type config struct {
	addr          string
	dbPath        string
	// dbDriver selects the SQL backend.  Default "sqlite3" (file-backed,
	// dbPath is the sqlite file).  Set DIST_DB_DRIVER=postgres and
	// DIST_DB_DSN=postgres://user:pw@host:5432/dbname?sslmode=require to
	// run on a real Postgres cluster.  When driver is postgres, dbDSN is
	// used as-is and dbPath is ignored.
	dbDriver      string
	dbDSN         string
	publicURL     string // e.g. http://localhost:8080 — used in deep links
	apexHost      string // e.g. surds.co.in — dashboard here, <slug>.apex for pools
	devMode       bool   // enables a "dev login" endpoint (no GitHub needed)
	githubClient  string
	githubSecret  string
	googleClient  string
	googleSecret  string
	sessionSecret string
	modelsDir     string // root dir where model shards live
	splitterBin   string // path to llama-split-gguf binary
	releasesDir   string // on-disk cache for GitHub release tarballs
	comfyOutDir   string // dir where ComfyUI renders are stored
	comfyEnabled  bool   // whether the ComfyUI integration is enabled on the agent install
	converterPy   string // path to llama.cpp's convert_hf_to_gguf.py
	pythonBin     string // python interpreter used to run the converter
	convertQuant  string // default convert output type (e.g. q8_0, f16, q4_k_m)

	// TURN relay (for WebRTC ACTV peers behind symmetric NATs).
	//
	// Two credential modes — pick one:
	//   • Static:    set turnURL + turnStaticUser + turnStaticPass.  Every
	//                pp_route ships the same long-lived credential pair.
	//                Easy, but the credential is reusable until rotated.
	//   • Ephemeral: set turnURL + turnSecret (+ optional realm/TTL).  The
	//                server mints a fresh `<exp>:<agent>` / HMAC-SHA1
	//                credential pair per pp_route, valid for turnTTL.
	//                Matches coturn's `use-auth-secret` /
	//                `static-auth-secret` REST scheme — no static password
	//                ever leaves the server.
	//
	// When both modes are configured ephemeral wins.  Leave all of them
	// blank to advertise STUN-only and rely on WS fallback for symmetric
	// NAT pairs.
	turnURL        string        // e.g. "turn:turn.example.com:3478"
	turnRealm      string        // optional, mostly cosmetic for clients
	turnSecret     string        // coturn shared secret (ephemeral mode)
	turnTTL        time.Duration // ephemeral credential lifetime
	turnStaticUser string        // long-lived username (static mode)
	turnStaticPass string        // long-lived password (static mode)

	// Trusted-proxy CIDRs.  When non-empty, X-Forwarded-For is honored
	// only for hops within these networks (see trusted_proxy.go).  Set
	// via DIST_TRUSTED_PROXIES="10.0.0.0/8,127.0.0.1/32".  Empty means
	// XFF is ignored and clientIP/remoteIPForRateLimit fall back to
	// r.RemoteAddr — the safe default when the server is exposed
	// directly to the internet.
	trustedProxiesRaw string
	trustedProxies    *trustedProxySet

	// Redis URL for the cross-instance rate-limit backend (P9).  Set
	// DIST_REDIS_URL=redis://host:6379/0 to share token-bucket state
	// across replicas behind envoy.  Empty ⇒ in-process limiter only
	// (single-replica deploy).  The dialed client lives on
	// cfg.rateBackend so the server can construct ipRateLimiterSet
	// without a second connect at request time.
	redisURL    string
	rateBackend rateBackend

	// OIDC federation (P5).  Set DIST_OIDC_ISSUER to enable; the server
	// will discover endpoints + JWKS via /.well-known/openid-configuration
	// at boot, then expose /auth/oidc/start + /auth/oidc/callback for the
	// browser code flow and accept JWT bearer tokens carrying matching
	// claims on /api endpoints.  All four vars are required when issuer
	// is set; the server fails closed (logs and skips OIDC) otherwise.
	//
	//   DIST_OIDC_ISSUER          https://dex.example.com
	//   DIST_OIDC_CLIENT_ID       distpool-server
	//   DIST_OIDC_CLIENT_SECRET   <shared with OP>
	//   DIST_OIDC_BEARER_AUDIENCE optional; the aud claim the bearer
	//                             middleware enforces.  Defaults to
	//                             DIST_OIDC_CLIENT_ID.
	//
	// At runtime the live provider sits on cfg.oidc; nil when disabled.
	oidcIssuer         string
	oidcClientID       string
	oidcClientSecret   string
	oidcBearerAudience string
	oidcLabel          string // human-readable label rendered on /auth (defaults to "SSO")
	oidc               *oidcProvider

	// SPIFFE / SPIRE workload identity (P6).  Set spiffeTrustDomain to
	// enable.  Two SVID flavours are accepted:
	//
	//   1. JWT-SVID over the /ws/agent hello — rig fetches it from
	//      its local SPIRE workload API socket and ships it as
	//      `spiffe_token`.  Verified against DIST_SPIFFE_JWKS_URL.
	//   2. X.509-SVID via envoy XFCC — envoy terminates mTLS, validates
	//      the client cert against the SPIRE trust bundle, then
	//      forwards the verified cert metadata in
	//      `x-forwarded-client-cert`.  Honoured only when the immediate
	//      TCP peer is in DIST_TRUSTED_PROXIES.
	//
	// At runtime the live provider sits on cfg.spiffe; nil when
	// disabled.  Legacy agent_key auth is untouched in either case.
	spiffeTrustDomain string
	spiffeJWKSURL     string
	spiffeAudience    string
	spiffe            *spiffeProvider
}

func loadConfig() config {
	c := config{
		addr:          envOr("DIST_ADDR", ":8080"),
		dbPath:        envOr("DIST_DB", "distpool.sqlite"),
		dbDriver:      envOr("DIST_DB_DRIVER", "sqlite3"),
		dbDSN:         os.Getenv("DIST_DB_DSN"),
		publicURL:     envOr("DIST_PUBLIC_URL", "http://localhost:8080"),
		apexHost:      envOr("DIST_APEX_HOST", "surds.co.in"),
		githubClient:  os.Getenv("DIST_GITHUB_CLIENT"),
		githubSecret:  os.Getenv("DIST_GITHUB_SECRET"),
		googleClient:  os.Getenv("DIST_GOOGLE_CLIENT"),
		googleSecret:  os.Getenv("DIST_GOOGLE_SECRET"),
		sessionSecret: envOr("DIST_SESSION_SECRET", "dev-secret-change-me"),
		modelsDir:     envOr("DIST_MODELS_DIR", "./models-store"),
		splitterBin:   envOr("DIST_SPLITTER", "/home/boom/startup/Project/llama.cpp/build/bin/llama-split-gguf"),
		releasesDir:   envOr("DIST_RELEASES_DIR", "./releases-cache"),
		comfyOutDir:   envOr("DIST_COMFY_OUT_DIR", "./comfy-out"),
		converterPy:   envOr("DIST_CONVERTER", "./third_party/llama.cpp/convert_hf_to_gguf.py"),
		pythonBin:     envOr("DIST_PYTHON", "python3"),
		convertQuant:  envOr("DIST_CONVERT_QUANT", "q8_0"),

		turnURL:        os.Getenv("DIST_TURN_URL"),
		turnRealm:      os.Getenv("DIST_TURN_REALM"),
		turnSecret:     os.Getenv("DIST_TURN_SECRET"),
		turnStaticUser: os.Getenv("DIST_TURN_USER"),
		turnStaticPass: os.Getenv("DIST_TURN_PASS"),
		turnTTL:        parseDurationEnv("DIST_TURN_TTL", time.Hour),

		trustedProxiesRaw: os.Getenv("DIST_TRUSTED_PROXIES"),
		redisURL:          os.Getenv("DIST_REDIS_URL"),

		oidcIssuer:         os.Getenv("DIST_OIDC_ISSUER"),
		oidcClientID:       os.Getenv("DIST_OIDC_CLIENT_ID"),
		oidcClientSecret:   os.Getenv("DIST_OIDC_CLIENT_SECRET"),
		oidcBearerAudience: os.Getenv("DIST_OIDC_BEARER_AUDIENCE"),
		oidcLabel:          envOr("DIST_OIDC_LABEL", "SSO"),

		spiffeTrustDomain: os.Getenv("DIST_SPIFFE_TRUST_DOMAIN"),
		spiffeJWKSURL:     os.Getenv("DIST_SPIFFE_JWKS_URL"),
		spiffeAudience:    os.Getenv("DIST_SPIFFE_AUDIENCE"),
	}
	tp, bad := parseTrustedProxies(c.trustedProxiesRaw)
	c.trustedProxies = tp
	if len(bad) > 0 {
		log.Printf("warn: DIST_TRUSTED_PROXIES contained %d invalid entries; ignored: %v",
			len(bad), bad)
	}
	if v := strings.ToLower(os.Getenv("DIST_WITH_COMFYUI")); v == "1" || v == "true" || v == "yes" {
		c.comfyEnabled = true
	}
	flag.StringVar(&c.addr, "addr", c.addr, "listen address")
	flag.StringVar(&c.dbPath, "db", c.dbPath, "SQLite path")
	flag.StringVar(&c.publicURL, "public-url", c.publicURL, "public URL for deep links")
	flag.StringVar(&c.apexHost, "apex", c.apexHost, "apex domain — dashboard here, <slug>.apex for pool endpoints")
	flag.StringVar(&c.modelsDir, "models-dir", c.modelsDir, "dir to store model shards")
	flag.StringVar(&c.splitterBin, "splitter", c.splitterBin, "path to llama-split-gguf")
	flag.StringVar(&c.releasesDir, "releases-dir", c.releasesDir, "on-disk cache dir for release tarballs")
	// Dev mode (POST /auth/dev — open account creation) must be explicitly
	// opted into via DIST_DEV_MODE=1 or --dev.  Previously this defaulted to
	// ON whenever DIST_GITHUB_CLIENT was unset, which meant any half-configured
	// deployment silently exposed an unauthenticated account-creation endpoint.
	// Fail closed instead.
	devDefault := false
	if v := strings.ToLower(os.Getenv("DIST_DEV_MODE")); v == "1" || v == "true" || v == "yes" {
		devDefault = true
	}
	flag.BoolVar(&c.devMode, "dev", devDefault, "dev mode: enable /auth/dev endpoint (DIST_DEV_MODE=1 to opt in)")
	flag.Parse()
	return c
}

func envOr(k, d string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return d
}

// parseIntEnv reads a base-10 integer from env and falls back to `d`
// when the variable is unset or unparseable.
func parseIntEnv(k string, d int) int {
	v := os.Getenv(k)
	if v == "" {
		return d
	}
	var n int
	_, err := fmt.Sscanf(v, "%d", &n)
	if err != nil {
		return d
	}
	return n
}

// buildDSN constructs the driver-specific DSN.  SQLite gets pragma
// flags glued onto the file path; Postgres uses DIST_DB_DSN verbatim.
func buildDSN(cfg config) string {
	switch cfg.dbDriver {
	case "postgres", "pgx":
		if cfg.dbDSN == "" {
			log.Fatalf("DIST_DB_DRIVER=%s requires DIST_DB_DSN", cfg.dbDriver)
		}
		return cfg.dbDSN
	default:
		// SQLite WAL is broken over CIFS (Azure Files mounts /data via
		// CIFS), so the journal mode must be overridable.  Default WAL
		// for local disk; set DIST_SQLITE_JOURNAL_MODE=DELETE on
		// CIFS-backed hosts.  On CIFS also set DIST_SQLITE_NOLOCK=1
		// (POSIX advisory locks are unreliable on CIFS).
		journalMode := envOr("DIST_SQLITE_JOURNAL_MODE", "WAL")
		dsn := cfg.dbPath + "?_journal_mode=" + journalMode +
			"&_foreign_keys=on&_busy_timeout=30000"
		if v := strings.ToLower(os.Getenv("DIST_SQLITE_NOLOCK")); v == "1" || v == "true" || v == "yes" {
			dsn += "&nolock=1"
		}
		return dsn
	}
}

// parseDurationEnv reads a Go duration string (e.g. "1h", "45m") from env
// and falls back to `d` when the variable is unset or unparseable.
func parseDurationEnv(k string, d time.Duration) time.Duration {
	v := os.Getenv(k)
	if v == "" {
		return d
	}
	if t, err := time.ParseDuration(v); err == nil {
		return t
	}
	return d
}

// cleanStaleSQLiteSidecars removes -journal / -wal / -shm files when the
// main DB is zero-length.  On Azure Files (CIFS), a writer that crashes
// mid-migration leaves these in place and the next start sees
// "database is locked" forever because SQLite cannot complete recovery
// against an empty main file.  Safe at boot — we only touch sidecars
// when the DB itself has no useful state to recover.
func cleanStaleSQLiteSidecars(dbPath string) {
	if dbPath == "" {
		return
	}
	st, err := os.Stat(dbPath)
	if err != nil || st.Size() > 0 {
		return
	}
	for _, suf := range []string{"-journal", "-wal", "-shm"} {
		p := dbPath + suf
		if _, err := os.Stat(p); err == nil {
			if rmErr := os.Remove(p); rmErr == nil {
				log.Printf("sqlite: cleared stale sidecar %s", p)
			}
		}
	}
	// Drop the zero-byte main file too so SQLite creates fresh.
	if err := os.Remove(dbPath); err == nil {
		log.Printf("sqlite: cleared zero-length %s", dbPath)
	}
}

// runMigrationsWithRetry executes the eight migration steps in order,
// retrying transient SQLITE_BUSY ("database is locked") errors with
// exponential backoff.  The 30 s busy_timeout baked into the DSN handles
// the common case; this outer loop covers the rarer case where a
// previous-replica crash leaves the CIFS file lease pinned for longer
// than the busy_timeout window.
func runMigrationsWithRetry(db *sql.DB, d sqlDialect) error {
	steps := []struct {
		name string
		fn   func(*sql.DB, sqlDialect) error
	}{
		{"core", migrate},
		{"hf", migrateHF},
		{"comfy", migrateComfy},
		{"reputation", migrateReputation},
		{"mcp", migrateMCP},
		{"rag", migrateRAG},
		{"conv_memory", migrateConvMemory},
		{"quarantine", migrateQuarantine},
		{"prefix_affinity", migratePrefixAffinity},
		{"spec_caps", migrateSpecCaps},
	}
	const maxAttempts = 8
	backoff := 2 * time.Second
	for _, s := range steps {
		var lastErr error
		for attempt := 1; attempt <= maxAttempts; attempt++ {
			err := s.fn(db, d)
			if err == nil {
				lastErr = nil
				break
			}
			lastErr = err
			msg := strings.ToLower(err.Error())
			if !strings.Contains(msg, "locked") && !strings.Contains(msg, "busy") {
				return fmt.Errorf("%s: %w", s.name, err)
			}
			log.Printf("migrate %s: attempt %d/%d hit lock (%v); backing off %s",
				s.name, attempt, maxAttempts, err, backoff)
			time.Sleep(backoff)
			if backoff < 30*time.Second {
				backoff *= 2
			}
		}
		if lastErr != nil {
			return fmt.Errorf("%s: %w", s.name, lastErr)
		}
		backoff = 2 * time.Second
	}
	return nil
}

func main() {
	cfg := loadConfig()

	// OpenTelemetry SDK init (P2).  No-op when OTEL_EXPORTER_OTLP_ENDPOINT
	// is unset — global providers stay no-op and every otel.Meter /
	// otel.Tracer call is free.  When set, traces + metrics export to
	// the configured collector at 5s/15s flush cadence.
	otelShutdown, err := setupTelemetry(context.Background())
	if err != nil {
		log.Fatalf("otel setup: %v", err)
	}
	defer func() {
		shutdownCtx, c := context.WithTimeout(context.Background(), 5*time.Second)
		defer c()
		_ = otelShutdown(shutdownCtx)
	}()

	dialect, err := dialectFor(cfg.dbDriver)
	if err != nil {
		log.Fatalf("db driver: %v", err)
	}

	// Stale journal files from a previous crash on a CIFS-backed mount
	// (Azure Files) can leave SQLite returning SQLITE_BUSY indefinitely.
	// When the journal/wal/shm sidecar files exist but the main DB is
	// zero-length, recovery is impossible — clear them so the first
	// connection can re-initialize cleanly.  Only runs for sqlite paths.
	if cfg.dbDriver == "" || cfg.dbDriver == "sqlite3" {
		cleanStaleSQLiteSidecars(cfg.dbPath)
	}

	dsn := buildDSN(cfg)
	db, err := sql.Open(dialect.Name(), dsn)
	if err != nil {
		log.Fatalf("open db: %v", err)
	}
	defer db.Close()
	// Tune the pool.  SQLite tolerates many opens but only one writer at
	// a time; Postgres benefits from a real pool size matched to the API
	// concurrency cap.  DIST_DB_MAX_OPEN_CONNS overrides.
	if n := parseIntEnv("DIST_DB_MAX_OPEN_CONNS", 0); n > 0 {
		db.SetMaxOpenConns(n)
	}
	if n := parseIntEnv("DIST_DB_MAX_IDLE_CONNS", 0); n > 0 {
		db.SetMaxIdleConns(n)
	}
	if err := runMigrationsWithRetry(db, dialect); err != nil {
		log.Fatalf("migrate: %v", err)
	}
	// Versioned migrations (P3) run after the legacy baseline.  Anything
	// new — ALTER TABLE, CREATE INDEX, new tables — lands in
	// server/migrations/<ts>_<name>.sql and is applied here exactly once
	// per DB.  Atlas-compatible naming so operators can generate diffs
	// with the atlas CLI and drop files straight into the directory.
	if err := applyVersionedMigrations(context.Background(), db, dialect); err != nil {
		log.Fatalf("versioned migrate: %v", err)
	}

	// Cross-instance rate-limit backend (P9).  Without DIST_REDIS_URL the
	// limiter is per-process — fine for single-replica deploys.  With it,
	// every replica behind envoy shares one budget per (bucket, IP), so
	// an attacker can't multiply their allowance by the replica count.
	// Failure to dial is fatal here: an operator who set DIST_REDIS_URL
	// wants the protection, and silently fail-opening to per-process
	// limits would be a worse surprise than a crashloop they can fix.
	if cfg.redisURL != "" {
		be, err := newRedisRateBackend(context.Background(), cfg.redisURL)
		if err != nil {
			log.Fatalf("redis ratelimit backend: %v", err)
		}
		cfg.rateBackend = be
		log.Printf("ratelimit: using redis backend at %s", cfg.redisURL)
	}

	// OIDC federation (P5).  Discovery is best-effort: if the OP is down
	// at boot we log and continue — the server still serves browser
	// sessions via GitHub/Google OAuth and api-key bearers.  The
	// alternative (fail-closed) would chain a transient identity-provider
	// outage into a full dist-server outage.
	if cfg.oidcIssuer != "" {
		op, err := newOIDCProvider(context.Background(), &cfg)
		if err != nil {
			log.Printf("oidc: discovery failed at %s: %v — /auth/oidc disabled", cfg.oidcIssuer, err)
		} else {
			cfg.oidc = op
			log.Printf("oidc: federation enabled (issuer=%s, audience=%s)",
				op.issuer, op.audience)
		}
	}

	// SPIFFE workload identity (P6).  Same best-effort posture as OIDC:
	// a SPIRE outage at boot logs and continues — legacy agent_key
	// auth keeps the fleet online.  The background refresher will
	// pick the bundle up on its next tick.
	if cfg.spiffeTrustDomain != "" {
		sp, err := newSPIFFEProvider(context.Background(), &cfg)
		if err != nil {
			log.Printf("spiffe: setup failed for %s: %v — SVID auth disabled", cfg.spiffeTrustDomain, err)
		} else {
			cfg.spiffe = sp
			sp.startRefreshLoop(context.Background())
			log.Printf("spiffe: workload identity enabled (trust_domain=%s, audience=%s)",
				sp.td, sp.audience)
		}
	}

	srv := newServer(cfg, db)
	srv.dialect = dialect

	// P11: enable BigQuery analytics sink when DIST_BQ_* is configured.
	// Absent any env, the nopAnalyticsSink default leaves behaviour
	// identical to a non-cloud deploy.
	if bq := os.Getenv("DIST_BQ_PROJECT"); bq != "" {
		bqCfg := bigqueryAnalyticsConfig{
			ProjectID: bq,
			DatasetID: os.Getenv("DIST_BQ_DATASET"),
			TableID:   os.Getenv("DIST_BQ_TABLE"),
		}
		if bqCfg.TableID == "" {
			bqCfg.TableID = "inference_log"
		}
		sink, err := newBigQueryAnalyticsSink(context.Background(), bqCfg)
		if err != nil {
			log.Printf("bigquery sink: disabled (%v)", err)
		} else {
			srv.analytics = sink
			log.Printf("bigquery sink: enabled (%s.%s.%s)",
				bqCfg.ProjectID, bqCfg.DatasetID, bqCfg.TableID)
		}
	}

	// P11: opt-in Bigtable hot-counter store.  Falls back to the
	// SQLite store when env is unset.  The current bigtableCounterStore
	// is a counted shim (see counter_store.go header) — flipping the
	// real Bigtable SDK on is a separate, build-tagged commit.
	if bt := os.Getenv("DIST_BT_PROJECT"); bt != "" {
		btCfg := bigtableCounterConfig{
			ProjectID:      bt,
			InstanceID:     os.Getenv("DIST_BT_INSTANCE"),
			TableID:        os.Getenv("DIST_BT_TABLE"),
			MirrorToSQLite: true,
		}
		fb := newSQLiteCounterStore(srv)
		store, err := newBigtableCounterStore(btCfg, fb)
		if err != nil {
			log.Printf("bigtable counters: disabled (%v)", err)
		} else {
			srv.counters = store
			log.Printf("bigtable counters: enabled (project=%s instance=%s table=%s)",
				btCfg.ProjectID, btCfg.InstanceID, btCfg.TableID)
		}
	}

	if err := srv.backfillSlugs(); err != nil {
		log.Fatalf("backfill slugs: %v", err)
	}
	if err := srv.backfillAgentKeyHashes(); err != nil {
		log.Fatalf("backfill agent_key_hashes: %v", err)
	}
	if err := srv.drift.hydrateQuarantine(db); err != nil {
		log.Printf("hydrate quarantine: %v", err)
	}
	httpSrv := &http.Server{
		Addr:              cfg.addr,
		Handler:           wrapWithOTel(srv.router()),
		ReadHeaderTimeout: 10 * time.Second,
		// Slowloris defense: cap the time a client can take to send the
		// request body, and how long an idle keep-alive connection survives.
		// WebSocket upgrades hijack the conn before these fire, so they
		// don't break long-lived agent/browser connections.
		ReadTimeout: 30 * time.Second,
		IdleTimeout: 120 * time.Second,
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

	// Phase 1 — flip /readyz to 503 so load balancers stop sending new
	// traffic.  Sleep briefly so the LB has time to notice (most poll at
	// 2–5s intervals).  Keep the listener open: in-flight requests are
	// still served.
	srv.shuttingDown.Store(true)
	log.Printf("shutdown: signalling /readyz=503, waiting 3s for LB drain")
	time.Sleep(3 * time.Second)

	// Phase 2 — fail every in-flight relay assignment so the right
	// reputation rows get debited even if the agent doesn't get a clean
	// release in.  Without this, a server restart looks identical to a
	// session that "leaked into the void" — both end without a release,
	// but only one is actually the relay's fault.  Crediting the failure
	// to ourselves (rather than the rig) is the honest call: the rig was
	// still alive.  We rely on operators redeploying behind a rolling
	// upgrade where the next process picks up the slack.
	if srv.relays != nil {
		var drained int
		srv.relays.mu.Lock()
		keys := make([]relayKey, 0, len(srv.relays.byKey))
		for k := range srv.relays.byKey {
			keys = append(keys, k)
		}
		srv.relays.mu.Unlock()
		for _, k := range keys {
			if a := srv.relays.remove(k.sessionID, k.agentID); a != nil {
				// Don't bump failure for our own shutdown — the rig didn't
				// do anything wrong.  Just log so operators have a record.
				log.Printf("shutdown: dropping relay assignment %s/%s (server stop)",
					a.AgentID, a.SessionID)
				drained++
			}
		}
		if drained > 0 {
			log.Printf("shutdown: drained %d active relay assignments", drained)
		}
	}

	// Phase 3 — graceful HTTP shutdown.  10s is enough for typical HTTP
	// request completions; WS connections will be force-closed by the
	// context cancellation in their per-conn goroutines.
	shutdown, c2 := context.WithTimeout(context.Background(), 10*time.Second)
	defer c2()
	_ = httpSrv.Shutdown(shutdown)
	// Final flush of pending last_seen rows.
	srv.flushLastSeen()
	// P11: drain the analytics sink with a tight deadline so a slow
	// BigQuery never holds the process up past ~5s.
	if srv.analytics != nil {
		drainCtx, drainCancel := context.WithTimeout(context.Background(), 5*time.Second)
		_ = srv.analytics.Close(drainCtx)
		drainCancel()
	}
	if srv.counters != nil {
		_ = srv.counters.Close()
	}
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
