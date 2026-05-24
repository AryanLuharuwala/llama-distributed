package main

// P11: Pluggable analytics sink for the inference event stream.
//
// The control-plane has always kept a short retention `inference_log`
// row per request: who called, which pool/agent served it, in/out
// tokens, status, latency.  That table answers "did this request
// happen?" and feeds the user-facing recent-requests endpoint, but it
// is *not* a long-term analytics store — it lives in the same SQLite
// (or Postgres) the request path uses, gets purged after a few weeks,
// and offers no way to OLAP over months of traffic.
//
// Production deploys want both:
//
//   1. Keep the local `inference_log` row for receipts + recent-traffic
//      UIs (synchronous, must not be dropped).
//   2. *Also* stream the same row, plus more context, to a warehouse
//      where analysts can answer "what's the p95 TTFT by model over
//      the last 30 days, broken down by region".  BigQuery is the
//      canonical Google-stack choice — streaming insert lands rows
//      within seconds, and partitioned tables by `started_at` keep
//      query cost bounded.
//
// This file defines the `analyticsSink` interface and a couple of
// implementations:
//
//   - `nopAnalyticsSink` — default, used when BQ is unconfigured.
//   - `bigqueryAnalyticsSink` — async fan-out to BigQuery via the
//     streaming-insert REST API.  Auth via google.FindDefaultCredentials
//     (workload identity, service-account JSON, or metadata server).
//
// Important invariants:
//
//   - The sink is *purely additive*: every code path that previously
//     wrote `inference_log` still does so, regardless of sink config.
//     A BQ outage cannot break inference.
//
//   - LogInference must never block the request path.  The BQ impl
//     buffers into a bounded channel; if the channel is full (the
//     sink is unhealthy or rows are arriving faster than BQ can
//     accept) we drop the row and increment a metric — better than
//     stalling /v1/chat/completions on a warehouse hiccup.
//
//   - The Close path drains the buffer with a bounded deadline so
//     shutting down the server doesn't lose more than ~one flush
//     window of rows.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
)

// inferenceEvent is the canonical shape of one inference row as it
// flows into the analytics sink.  Fields are JSON-tagged for the
// BigQuery streaming-insert payload, which expects a flat JSON object
// per row.
//
// New fields can be added safely: BigQuery streaming inserts ignore
// columns not in the destination table (when `ignoreUnknownValues` is
// true on insertAll — we set it).  Renames need a schema migration on
// the BQ side; flag them in the comment so deploy-side knows.
type inferenceEvent struct {
	// Local DB id; stable per row.  Lets us de-dupe if the sink
	// retries an insert that already landed.
	ID int64 `json:"id"`

	UserID      int64  `json:"user_id"`
	PoolID      int64  `json:"pool_id"`
	AgentUserID int64  `json:"agent_user_id"`
	AgentID     string `json:"agent_id"`

	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`

	StartedAt   int64 `json:"started_at"`
	FinishedAt  int64 `json:"finished_at,omitempty"`
	LatencyMs   int64 `json:"latency_ms,omitempty"`

	Status string `json:"status"`

	// Optional context.  These will be empty for legacy callsites
	// that don't pass them; the BQ table tolerates NULLs.
	ModelName string `json:"model_name,omitempty"`
	Region    string `json:"region,omitempty"`
}

// analyticsSink is the minimum surface the inference path needs.
// Implementations must be safe for concurrent callers.
type analyticsSink interface {
	// LogInference fires-and-forgets the event.  Must not block the
	// caller; if the sink is overloaded, drop and metric.
	LogInference(ev *inferenceEvent)

	// Close flushes any buffered rows with a bounded deadline.
	Close(ctx context.Context) error
}

// ─── nop sink ─────────────────────────────────────────────────────────

type nopAnalyticsSink struct{}

func (nopAnalyticsSink) LogInference(*inferenceEvent)         {}
func (nopAnalyticsSink) Close(context.Context) error          { return nil }

// ─── BigQuery streaming-insert sink ───────────────────────────────────

// bigqueryAnalyticsSink streams rows to one BigQuery table via the
// `insertAll` REST endpoint.  We don't use the cloud.google.com/go/bigquery
// SDK because:
//
//   - It pulls 50+ transitive deps; this binary is already big.
//   - The streaming insert REST surface is tiny (one POST per batch).
//   - We want full control over batching, retries, and drop semantics.
//
// Auth: google.FindDefaultCredentials picks the right path for the
// deployment — workload identity (GKE), metadata server (Cloud Run),
// or GOOGLE_APPLICATION_CREDENTIALS file.  Scopes we need:
// https://www.googleapis.com/auth/bigquery.insertdata.
type bigqueryAnalyticsSink struct {
	projectID string
	datasetID string
	tableID   string

	endpoint  string // overridable for tests
	tokenSrc  oauth2.TokenSource
	client    *http.Client

	buf chan *inferenceEvent

	// flushSize triggers an early flush when the buffer reaches this
	// many rows; otherwise we wait for flushInterval.
	flushSize     int
	flushInterval time.Duration

	// Lifecycle.
	wg     sync.WaitGroup
	doneCh chan struct{}
	closed atomic.Bool

	// Drop counter — exposed via metrics for ops.
	dropped atomic.Int64
}

// bigqueryAnalyticsConfig captures the env-driven configuration; an
// unset projectID/datasetID/tableID disables the sink (the boot path
// substitutes a nopAnalyticsSink).
type bigqueryAnalyticsConfig struct {
	ProjectID string
	DatasetID string
	TableID   string
	// BufferSize bounds the in-memory queue.  When full, LogInference
	// drops the row.  Default: 10_000 — enough to ride out a 30s BQ
	// blip at 300 rps without dropping, but small enough that an
	// hours-long outage doesn't OOM the server.
	BufferSize int
	// FlushSize / FlushInterval: an insertAll fires when either is
	// reached.  Defaults: 500 rows or 5 seconds — BigQuery's stated
	// streaming throughput sweet spot.
	FlushSize     int
	FlushInterval time.Duration
	// EndpointBase lets tests point the sink at a httptest.Server.
	// Defaults to https://bigquery.googleapis.com if empty.
	EndpointBase string
}

func newBigQueryAnalyticsSink(ctx context.Context, cfg bigqueryAnalyticsConfig) (*bigqueryAnalyticsSink, error) {
	if cfg.ProjectID == "" || cfg.DatasetID == "" || cfg.TableID == "" {
		return nil, fmt.Errorf("bigquery sink: project/dataset/table required")
	}
	if cfg.BufferSize <= 0 {
		cfg.BufferSize = 10_000
	}
	if cfg.FlushSize <= 0 {
		cfg.FlushSize = 500
	}
	if cfg.FlushInterval <= 0 {
		cfg.FlushInterval = 5 * time.Second
	}
	endpointBase := cfg.EndpointBase
	if endpointBase == "" {
		endpointBase = "https://bigquery.googleapis.com"
	}

	creds, err := google.FindDefaultCredentials(ctx,
		"https://www.googleapis.com/auth/bigquery.insertdata")
	if err != nil {
		return nil, fmt.Errorf("bigquery sink: find creds: %w", err)
	}
	endpoint := fmt.Sprintf("%s/bigquery/v2/projects/%s/datasets/%s/tables/%s/insertAll",
		endpointBase, cfg.ProjectID, cfg.DatasetID, cfg.TableID)

	s := &bigqueryAnalyticsSink{
		projectID:     cfg.ProjectID,
		datasetID:     cfg.DatasetID,
		tableID:       cfg.TableID,
		endpoint:      endpoint,
		tokenSrc:      creds.TokenSource,
		client:        &http.Client{Timeout: 30 * time.Second},
		buf:           make(chan *inferenceEvent, cfg.BufferSize),
		flushSize:     cfg.FlushSize,
		flushInterval: cfg.FlushInterval,
		doneCh:        make(chan struct{}),
	}
	s.wg.Add(1)
	go s.run()
	return s, nil
}

func (s *bigqueryAnalyticsSink) LogInference(ev *inferenceEvent) {
	if s.closed.Load() {
		return
	}
	// Non-blocking send: under load we'd rather drop a row than
	// stall the request path.
	select {
	case s.buf <- ev:
	default:
		s.dropped.Add(1)
	}
}

func (s *bigqueryAnalyticsSink) Dropped() int64 { return s.dropped.Load() }

func (s *bigqueryAnalyticsSink) Close(ctx context.Context) error {
	if !s.closed.CompareAndSwap(false, true) {
		return nil
	}
	close(s.doneCh)
	done := make(chan struct{})
	go func() { s.wg.Wait(); close(done) }()
	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (s *bigqueryAnalyticsSink) run() {
	defer s.wg.Done()
	tick := time.NewTicker(s.flushInterval)
	defer tick.Stop()
	batch := make([]*inferenceEvent, 0, s.flushSize)
	flush := func() {
		if len(batch) == 0 {
			return
		}
		if err := s.insertAll(batch); err != nil {
			log.Printf("bigquery sink: insertAll: %v (rows=%d dropped++)", err, len(batch))
			s.dropped.Add(int64(len(batch)))
		}
		batch = batch[:0]
	}
	for {
		select {
		case <-s.doneCh:
			// Drain whatever's left in the buffer before exiting.  The
			// Close() ctx deadline bounds this loop indirectly via the
			// HTTP client timeout.
			for {
				select {
				case ev := <-s.buf:
					batch = append(batch, ev)
					if len(batch) >= s.flushSize {
						flush()
					}
				default:
					flush()
					return
				}
			}
		case ev := <-s.buf:
			batch = append(batch, ev)
			if len(batch) >= s.flushSize {
				flush()
			}
		case <-tick.C:
			flush()
		}
	}
}

// bqInsertAllRequest is the wire shape BigQuery expects for
// streaming insertAll.  Each row needs a stable insertId for
// best-effort de-duplication; we use the local DB row id formatted
// as decimal — BQ keeps an in-memory dedupe window of ~1 minute.
type bqInsertAllRequest struct {
	IgnoreUnknownValues bool             `json:"ignoreUnknownValues"`
	SkipInvalidRows     bool             `json:"skipInvalidRows"`
	Rows                []bqInsertAllRow `json:"rows"`
}

type bqInsertAllRow struct {
	InsertID string          `json:"insertId"`
	JSON     *inferenceEvent `json:"json"`
}

// bqInsertAllResponse — BigQuery returns 200 OK even when individual
// rows fail; the failing rows are listed in insertErrors.  We log them
// but don't retry (re-inserting a row that failed validation will fail
// again).
type bqInsertAllResponse struct {
	InsertErrors []struct {
		Index  int `json:"index"`
		Errors []struct {
			Reason  string `json:"reason"`
			Message string `json:"message"`
		} `json:"errors"`
	} `json:"insertErrors"`
}

func (s *bigqueryAnalyticsSink) insertAll(batch []*inferenceEvent) error {
	rows := make([]bqInsertAllRow, len(batch))
	for i, ev := range batch {
		rows[i] = bqInsertAllRow{
			InsertID: fmt.Sprintf("%d", ev.ID),
			JSON:     ev,
		}
	}
	body, err := json.Marshal(bqInsertAllRequest{
		IgnoreUnknownValues: true,
		SkipInvalidRows:     true,
		Rows:                rows,
	})
	if err != nil {
		return fmt.Errorf("marshal: %w", err)
	}

	tok, err := s.tokenSrc.Token()
	if err != nil {
		return fmt.Errorf("token: %w", err)
	}

	req, err := http.NewRequest(http.MethodPost, s.endpoint, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("new request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+tok.AccessToken)
	req.Header.Set("Content-Type", "application/json")

	resp, err := s.client.Do(req)
	if err != nil {
		return fmt.Errorf("do: %w", err)
	}
	defer resp.Body.Close()
	respBody, _ := io.ReadAll(resp.Body)

	if resp.StatusCode/100 != 2 {
		return fmt.Errorf("bq insertAll http %d: %s", resp.StatusCode, string(respBody))
	}

	var out bqInsertAllResponse
	if err := json.Unmarshal(respBody, &out); err == nil && len(out.InsertErrors) > 0 {
		// BQ returned 200 but with per-row errors — log so an operator
		// can spot a schema drift, but treat the batch as accepted.
		log.Printf("bigquery sink: %d per-row errors out of %d", len(out.InsertErrors), len(batch))
	}
	return nil
}
