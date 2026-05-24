package main

// Tests for the BigQuery analytics sink.  We don't talk to BigQuery
// directly here — instead we point the sink at an httptest server
// that mimics the insertAll endpoint shape, so we exercise:
//
//   - request shape (Authorization header, body schema)
//   - batching (flushSize and flushInterval both trigger flushes)
//   - drop-on-overflow when the buffer fills faster than the sink drains
//   - Close drains pending rows within the deadline

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"golang.org/x/oauth2"
)

// fakeTokenSource skips the real google.FindDefaultCredentials so the
// sink test runs without GOOGLE_APPLICATION_CREDENTIALS.
type fakeTokenSource struct{}

func (fakeTokenSource) Token() (*oauth2.Token, error) {
	return &oauth2.Token{AccessToken: "test-token", Expiry: time.Now().Add(time.Hour)}, nil
}

// newTestBigQuerySink mirrors newBigQueryAnalyticsSink but injects a
// fake token source so the test doesn't touch real credentials.
func newTestBigQuerySink(t *testing.T, endpoint string, bufSize, flushSize int, flushInterval time.Duration) *bigqueryAnalyticsSink {
	t.Helper()
	s := &bigqueryAnalyticsSink{
		projectID:     "p",
		datasetID:     "d",
		tableID:       "t",
		endpoint:      endpoint,
		tokenSrc:      fakeTokenSource{},
		client:        &http.Client{Timeout: 5 * time.Second},
		buf:           make(chan *inferenceEvent, bufSize),
		flushSize:     flushSize,
		flushInterval: flushInterval,
		doneCh:        make(chan struct{}),
	}
	s.wg.Add(1)
	go s.run()
	return s
}

func TestBigQuerySink_RequestShapeAndAuth(t *testing.T) {
	var got bqInsertAllRequest
	var gotAuth string
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &got)
		calls.Add(1)
		w.WriteHeader(200)
		_, _ = w.Write([]byte(`{}`))
	}))
	defer srv.Close()

	sink := newTestBigQuerySink(t, srv.URL, 100, 1, time.Hour) // flushSize=1 → flushes per row
	defer func() {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		_ = sink.Close(ctx)
	}()

	sink.LogInference(&inferenceEvent{
		ID: 42, UserID: 7, AgentID: "rig-1",
		InputTokens: 100, OutputTokens: 200,
		StartedAt: 1716000000, Status: "ok",
	})

	// Wait for the flush goroutine to pick it up.
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) && calls.Load() == 0 {
		time.Sleep(20 * time.Millisecond)
	}
	if calls.Load() == 0 {
		t.Fatal("sink never POSTed to endpoint")
	}
	if gotAuth != "Bearer test-token" {
		t.Errorf("Authorization=%q, want Bearer test-token", gotAuth)
	}
	if len(got.Rows) != 1 {
		t.Fatalf("want 1 row, got %d", len(got.Rows))
	}
	if got.Rows[0].InsertID != "42" {
		t.Errorf("insertId=%q, want 42", got.Rows[0].InsertID)
	}
	if got.Rows[0].JSON.UserID != 7 {
		t.Errorf("user_id=%d, want 7", got.Rows[0].JSON.UserID)
	}
	if !got.IgnoreUnknownValues {
		t.Error("want IgnoreUnknownValues=true so schema additions are forward-compatible")
	}
}

func TestBigQuerySink_BatchesByFlushSize(t *testing.T) {
	var rowsTotal atomic.Int32
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req bqInsertAllRequest
		_ = json.Unmarshal(body, &req)
		rowsTotal.Add(int32(len(req.Rows)))
		calls.Add(1)
		w.WriteHeader(200)
		_, _ = w.Write([]byte(`{}`))
	}))
	defer srv.Close()

	sink := newTestBigQuerySink(t, srv.URL, 100, 5, time.Hour)
	defer func() {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		_ = sink.Close(ctx)
	}()

	for i := int64(0); i < 12; i++ {
		sink.LogInference(&inferenceEvent{ID: i})
	}
	// Force the drain by closing.
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := sink.Close(ctx); err != nil {
		t.Fatalf("Close: %v", err)
	}

	if rowsTotal.Load() != 12 {
		t.Errorf("rows delivered=%d, want 12", rowsTotal.Load())
	}
	// 12 rows / flushSize 5 = at least 3 calls (2 full batches + final drain).
	if calls.Load() < 3 {
		t.Errorf("calls=%d, want >=3 (batches should split)", calls.Load())
	}
}

func TestBigQuerySink_DropOnOverflow(t *testing.T) {
	// Slow handler so the sink can't drain fast enough.
	block := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-block
		w.WriteHeader(200)
		_, _ = w.Write([]byte(`{}`))
	}))
	defer srv.Close()
	defer close(block)

	sink := newTestBigQuerySink(t, srv.URL, 4, 100, time.Hour) // small buffer
	defer func() {
		// Unblock the handler before Close so the draining goroutine
		// can finish.
		ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
		defer cancel()
		_ = sink.Close(ctx)
	}()

	// Pump many more rows than the buffer holds; the slow handler
	// keeps the run goroutine busy on the first batch, so subsequent
	// LogInference calls fill and overflow the channel.
	for i := int64(0); i < 200; i++ {
		sink.LogInference(&inferenceEvent{ID: i})
	}
	if sink.Dropped() == 0 {
		t.Error("expected drops when buffer overflows under a stalled sink")
	}
}

func TestBigQuerySink_Insertions_ParallelSafe(t *testing.T) {
	var n atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req bqInsertAllRequest
		_ = json.Unmarshal(body, &req)
		n.Add(int32(len(req.Rows)))
		w.WriteHeader(200)
		_, _ = w.Write([]byte(`{}`))
	}))
	defer srv.Close()

	sink := newTestBigQuerySink(t, srv.URL, 1000, 50, 100*time.Millisecond)
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(base int64) {
			defer wg.Done()
			for i := int64(0); i < 100; i++ {
				sink.LogInference(&inferenceEvent{ID: base*1000 + i})
			}
		}(int64(g))
	}
	wg.Wait()

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	if err := sink.Close(ctx); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if got := n.Load(); got != 800 {
		t.Errorf("rows delivered=%d, want 800 (zero drops under healthy sink)", got)
	}
}

// TestBigQuerySink_Endpoint5xx tests that a 5xx response drops the batch and
// increments the drop counter (we don't retry; the next batch carries on).
func TestBigQuerySink_Endpoint5xx(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "internal", 500)
	}))
	defer srv.Close()

	sink := newTestBigQuerySink(t, srv.URL, 10, 1, time.Hour)
	for i := int64(0); i < 3; i++ {
		sink.LogInference(&inferenceEvent{ID: i})
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := sink.Close(ctx); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if sink.Dropped() == 0 {
		t.Error("5xx batch should be counted as dropped")
	}
}

// ensure the JSON shape matches the BQ insertAll wire contract.
func TestInferenceEvent_JSONShape(t *testing.T) {
	ev := &inferenceEvent{
		ID: 1, UserID: 2, PoolID: 3, AgentUserID: 4, AgentID: "a",
		InputTokens: 5, OutputTokens: 6,
		StartedAt: 7, FinishedAt: 8, LatencyMs: 9,
		Status: "ok", ModelName: "llama", Region: "us-central1",
	}
	b, err := json.Marshal(ev)
	if err != nil {
		t.Fatal(err)
	}
	// Must NOT include latency_ms when zero / status when empty? We do
	// emit them explicitly because BQ takes nulls/zeros fine; this test
	// just ensures the field names are stable.
	for _, want := range []string{
		`"id":1`, `"user_id":2`, `"pool_id":3`, `"agent_user_id":4`,
		`"agent_id":"a"`, `"input_tokens":5`, `"output_tokens":6`,
		`"started_at":7`, `"finished_at":8`, `"latency_ms":9`,
		`"status":"ok"`, `"model_name":"llama"`, `"region":"us-central1"`,
	} {
		if !bytes.Contains(b, []byte(want)) {
			t.Errorf("missing field %q in marshalled event: %s", want, b)
		}
	}
}
