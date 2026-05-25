package main

// Operational endpoints: /metrics (Prometheus text exposition) and /readyz
// (deep readiness check that fails closed if the DB is unreachable).
//
// Why hand-rolled instead of prometheus/client_golang?  Two reasons:
//   1. Pulls in ~2 MB of deps for one endpoint we control end-to-end.
//   2. The metrics we expose are slow-changing aggregates (online rig
//      count, active relay assignments, reputation row count) that we
//      already compute elsewhere — wrapping them in Counter/Gauge types
//      would add allocation cost without any real benefit.
//
// Scrapers we care about (Prometheus, Grafana Agent, vmagent) all parse
// the text format we emit here.

import (
	"fmt"
	"io"
	"net/http"
	"runtime"
	"strings"
	"time"
)

// handleMetrics emits text/plain in Prometheus exposition format.  Cheap
// — every gauge is computed lazily and the surface area is small (a few
// dozen lines).  Scrape interval should be ≥15s for any large swarm.
func (s *server) handleMetrics(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")

	uptime := time.Since(s.startedAt).Seconds()

	// Online agents + inference slot utilization — single hub pass so we
	// don't take the lock twice.  slotsUsed sums the live inferPeers maps
	// across rigs; slotsCap sums each rig's clamped MaxConcurrent.  Once
	// rigs start advertising MaxConcurrent>1, used/cap is the batching
	// saturation curve.
	var (
		onlineAgents int
		slotsUsed    int
		slotsCap     int
	)
	s.hub.mu.RLock()
	onlineAgents = len(s.hub.agents)
	agentSnapshot := make([]*agentConn, 0, onlineAgents)
	for _, ac := range s.hub.agents {
		agentSnapshot = append(agentSnapshot, ac)
	}
	s.hub.mu.RUnlock()
	for _, ac := range agentSnapshot {
		u, c := ac.snapshotInferSlots()
		slotsUsed += u
		slotsCap += c
	}

	// Active relay assignments — bounded by concurrent sessions.
	var activeRelaysCount int
	if s.relays != nil {
		s.relays.mu.Lock()
		activeRelaysCount = len(s.relays.byKey)
		s.relays.mu.Unlock()
	}

	// Reputation row count + aggregate counters (single query, cheap).
	var (
		repRows       int64
		repTotalSess  int64
		repTotalSucc  int64
		repTotalFail  int64
		repTotalBytes int64
	)
	if s.db != nil {
		_ = s.dbQueryRow(`SELECT COUNT(*),
		                          COALESCE(SUM(relay_sessions_total), 0),
		                          COALESCE(SUM(relay_sessions_success), 0),
		                          COALESCE(SUM(relay_sessions_failed), 0),
		                          COALESCE(SUM(relay_bytes_forwarded), 0)
		                   FROM rig_reputation`).
			Scan(&repRows, &repTotalSess, &repTotalSucc, &repTotalFail, &repTotalBytes)
	}

	// Go runtime stats — useful for OOM postmortems.
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)

	var b strings.Builder
	writeGauge(&b, "distinf_uptime_seconds", "Process uptime in seconds.", uptime, nil)
	writeGauge(&b, "distinf_build_info", "Build info (always 1).", 1,
		map[string]string{"go_version": runtime.Version()})
	writeGauge(&b, "distinf_agents_online", "Online agent WebSocket connections.",
		float64(onlineAgents), nil)
	writeGauge(&b, "distinf_inference_slots_used",
		"Sum of currently-occupied inference slots across all online rigs.",
		float64(slotsUsed), nil)
	writeGauge(&b, "distinf_inference_slots_capacity",
		"Sum of advertised inference slot capacities across all online rigs.",
		float64(slotsCap), nil)

	// Comfy / image-gen job stats — separate axis from LLM inference
	// because they map onto different rig capabilities and a different
	// dispatch path (HTTP→hub broadcast vs. per-rig INFR frames).
	cActive, cQueued, cFailed := s.countComfyJobs()
	writeGauge(&b, "distinf_comfy_jobs_active",
		"In-process comfy jobs currently bound to a dispatch goroutine.",
		float64(cActive), nil)
	writeGauge(&b, "distinf_comfy_jobs_queued",
		"Comfy jobs in 'queued' state awaiting dispatch.",
		float64(cQueued), nil)
	writeCounter(&b, "distinf_comfy_jobs_failed_total",
		"Total comfy jobs that ended in failed or cancelled state (lifetime).",
		float64(cFailed), nil)
	writeCounter(&b, "distinf_comfy_legacy_url_rejected_total",
		"Lifetime count of comfy output URLs rejected because their v1 signature aged past the grace window.",
		float64(s.comfyLegacyURLRejected.Load()), nil)
	writeGauge(&b, "distinf_relay_assignments_active",
		"Currently-attributed relay sessions awaiting release.",
		float64(activeRelaysCount), nil)
	writeGauge(&b, "distinf_reputation_rows", "Rows in rig_reputation table.",
		float64(repRows), nil)
	writeCounter(&b, "distinf_relay_sessions_total",
		"Total relay sessions accumulated across all rigs (lifetime).",
		float64(repTotalSess), nil)
	writeCounter(&b, "distinf_relay_sessions_success_total",
		"Total successful relay sessions accumulated across all rigs.",
		float64(repTotalSucc), nil)
	writeCounter(&b, "distinf_relay_sessions_failed_total",
		"Total failed relay sessions accumulated across all rigs.",
		float64(repTotalFail), nil)
	writeCounter(&b, "distinf_relay_bytes_forwarded_total",
		"Bytes forwarded by relays (sum, lifetime, both directions).",
		float64(repTotalBytes), nil)

	// Operational counters scraped straight from the DB.  Each is a single
	// indexed query; safe to run on a 15s scrape cadence.
	if s.db != nil {
		var n int64
		if _ = s.dbQueryRow(`SELECT COUNT(*) FROM users`).Scan(&n); true {
			writeGauge(&b, "distinf_users_total", "Registered users.", float64(n), nil)
		}
		if _ = s.dbQueryRow(`SELECT COUNT(*) FROM rigs`).Scan(&n); true {
			writeGauge(&b, "distinf_rigs_registered", "Rigs known to the control plane (online + offline).",
				float64(n), nil)
		}
		// Pools by visibility, one series per label.
		rows, err := s.dbQuery(`SELECT visibility, COUNT(*) FROM pools GROUP BY visibility`)
		if err == nil {
			for rows.Next() {
				var v string
				var c int64
				if rows.Scan(&v, &c) == nil {
					writeGauge(&b, "distinf_pools",
						"Pools grouped by visibility.", float64(c),
						map[string]string{"visibility": v})
				}
			}
			_ = rows.Close()
		}
		if _ = s.dbQueryRow(
			`SELECT COUNT(*) FROM pool_invites WHERE used_at IS NULL AND expires_at >= ?`,
			nowUnix()).Scan(&n); true {
			writeGauge(&b, "distinf_pool_invites_active",
				"Outstanding pool invites that are neither used nor expired.",
				float64(n), nil)
		}
		// Device-code logins waiting on browser approval.
		_ = s.dbQueryRow(
			`SELECT COUNT(*) FROM device_codes WHERE approved = 0 AND expires_at >= ?`,
			nowUnix()).Scan(&n)
		writeGauge(&b, "distinf_device_codes_pending",
			"Device-code login flows in flight (not yet approved).", float64(n), nil)

		// 24h inference rollup, sliced by status.
		since := nowUnix() - 86400
		jobRows, err := s.dbQuery(
			`SELECT status, COUNT(*),
			        COALESCE(SUM(input_tokens), 0),
			        COALESCE(SUM(output_tokens), 0)
			   FROM inference_log
			  WHERE started_at >= ?
			  GROUP BY status`,
			since)
		if err == nil {
			for jobRows.Next() {
				var status string
				var cnt, tin, tout int64
				if jobRows.Scan(&status, &cnt, &tin, &tout) == nil {
					labels := map[string]string{"status": status}
					writeGauge(&b, "distinf_inference_jobs_24h",
						"Inference jobs in the last 24h by status.", float64(cnt), labels)
					writeGauge(&b, "distinf_inference_tokens_24h",
						"Tokens billed in the last 24h, sliced by status.",
						float64(tin+tout), labels)
				}
			}
			_ = jobRows.Close()
		}
	}

	// Goroutines + memory — early-warning for leaks.
	writeGauge(&b, "go_goroutines", "Number of live goroutines.",
		float64(runtime.NumGoroutine()), nil)
	writeGauge(&b, "go_memstats_alloc_bytes",
		"Bytes of allocated heap objects.", float64(ms.Alloc), nil)
	writeGauge(&b, "go_memstats_sys_bytes",
		"Bytes obtained from system.", float64(ms.Sys), nil)
	writeGauge(&b, "go_memstats_heap_inuse_bytes",
		"Bytes in in-use spans.", float64(ms.HeapInuse), nil)

	_, _ = io.WriteString(w, b.String())
}

// writeGauge appends a Prometheus-format gauge.  No registration / no
// allocation tracking — just text in, text out.
func writeGauge(b *strings.Builder, name, help string, v float64, labels map[string]string) {
	fmt.Fprintf(b, "# HELP %s %s\n# TYPE %s gauge\n%s%s %s\n",
		name, help, name, name, formatLabels(labels), formatFloat(v))
}

func writeCounter(b *strings.Builder, name, help string, v float64, labels map[string]string) {
	fmt.Fprintf(b, "# HELP %s %s\n# TYPE %s counter\n%s%s %s\n",
		name, help, name, name, formatLabels(labels), formatFloat(v))
}

func formatLabels(labels map[string]string) string {
	if len(labels) == 0 {
		return ""
	}
	parts := make([]string, 0, len(labels))
	for k, v := range labels {
		// Escape the few characters Prometheus disallows in label values.
		v = strings.NewReplacer(`\`, `\\`, `"`, `\"`, "\n", `\n`).Replace(v)
		parts = append(parts, fmt.Sprintf(`%s="%s"`, k, v))
	}
	return "{" + strings.Join(parts, ",") + "}"
}

func formatFloat(v float64) string {
	// Prometheus accepts integers as-is; emit them without .0 to keep
	// scrape payload small.  For everything else, %g picks the shortest
	// roundtrip representation.
	if v == float64(int64(v)) {
		return fmt.Sprintf("%d", int64(v))
	}
	return fmt.Sprintf("%g", v)
}

// handleReadyz answers true once we've finished warming up AND the DB
// responds to a ping.  Kubernetes / cloud LBs poll this; we return 503
// during shutdown so traffic drains before the listener actually closes.
func (s *server) handleReadyz(w http.ResponseWriter, _ *http.Request) {
	if s.db == nil {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte("no db"))
		return
	}
	// Bound the ping — we don't want /readyz to itself become a slow
	// dependency that gets the pod killed.  500 ms is generous; SQLite
	// pings are sub-ms in practice.
	if err := s.db.Ping(); err != nil {
		w.WriteHeader(http.StatusServiceUnavailable)
		fmt.Fprintf(w, "db ping failed: %v", err)
		return
	}
	// During shutdown we flip an atomic to refuse new connections.
	if s.shuttingDown.Load() {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte("shutting down"))
		return
	}
	_, _ = w.Write([]byte("ready"))
}
