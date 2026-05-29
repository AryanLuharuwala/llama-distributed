package main

// Custom OpenTelemetry instruments.  Distinct from metrics.go, which
// emits Prometheus exposition for /metrics — this file is the OTLP
// push-side surface that flows to the configured OTel collector.
//
// Each instrument is lazy-initialized via sync.Once: when OTel is
// disabled the global noop MeterProvider gives us free-cost
// implementations, and even the Once'd init becomes ~one atomic load
// per process.  Hot-path callers pay nothing they didn't already pay.
//
// What's instrumented here:
//   - dist.ratelimit.allow / .deny / .error
//       Per-bucket counters so an operator can graph "device-poll
//       denies per minute by bucket" or "redis backend errors".
//       Tagged with bucket=<name>, backend=local|redis.
//   - dist.ws.active
//       UpDownCounter for live WebSocket connections.  Incremented on
//       accept, decremented on close.  /ws/agent + /ws/browser are
//       distinguished by the kind label.
//   - dist.inference.requests
//       Counter of settled inferences, tagged by model + status
//       (ok|error|timeout).  Driven from tokenization.settle.

import (
	"context"
	"log"
	"sync"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric"
)

type otelInstruments struct {
	rateAllow metric.Int64Counter
	rateDeny  metric.Int64Counter
	rateErr   metric.Int64Counter
	wsActive  metric.Int64UpDownCounter
	infReq    metric.Int64Counter
}

var (
	otelInstOnce sync.Once
	otelInst     *otelInstruments
)

// otelM returns the lazy-initialized instrument bag.  Safe to call
// from any goroutine; the underlying instruments are concurrent-safe.
// Returns a bag with no-op instruments if the global MeterProvider has
// not been configured (matches OTel's noop semantics — no error).
func otelM() *otelInstruments {
	otelInstOnce.Do(func() {
		m := otel.Meter("github.com/gpunet/server")
		bag := &otelInstruments{}

		var err error
		bag.rateAllow, err = m.Int64Counter("dist.ratelimit.allow",
			metric.WithDescription("Allowed token-bucket checks per (bucket, backend)"),
		)
		if err != nil {
			log.Printf("otel metric init ratelimit.allow: %v", err)
		}
		bag.rateDeny, err = m.Int64Counter("dist.ratelimit.deny",
			metric.WithDescription("Denied token-bucket checks per (bucket, backend)"),
		)
		if err != nil {
			log.Printf("otel metric init ratelimit.deny: %v", err)
		}
		bag.rateErr, err = m.Int64Counter("dist.ratelimit.error",
			metric.WithDescription("Rate-limit backend errors (fail-open events)"),
		)
		if err != nil {
			log.Printf("otel metric init ratelimit.error: %v", err)
		}
		bag.wsActive, err = m.Int64UpDownCounter("dist.ws.active",
			metric.WithDescription("Live WebSocket connections per kind"),
		)
		if err != nil {
			log.Printf("otel metric init ws.active: %v", err)
		}
		bag.infReq, err = m.Int64Counter("dist.inference.requests",
			metric.WithDescription("Settled inference requests per (model, status)"),
		)
		if err != nil {
			log.Printf("otel metric init inference.requests: %v", err)
		}

		otelInst = bag
	})
	return otelInst
}

// recordRate is the single entry point used by ipRateBucket.allow so
// instrumentation code lives in one place.  bucket is the bucket name
// (dev-approve, dev-poll, …); backend is "local" or "redis"; allowed
// indicates the decision.
func recordRate(ctx context.Context, bucket, backend string, allowed bool) {
	m := otelM()
	attrs := metric.WithAttributes(
		attribute.String("bucket", bucket),
		attribute.String("backend", backend),
	)
	if allowed {
		if m.rateAllow != nil {
			m.rateAllow.Add(ctx, 1, attrs)
		}
	} else {
		if m.rateDeny != nil {
			m.rateDeny.Add(ctx, 1, attrs)
		}
	}
}

func recordRateError(ctx context.Context, bucket, backend string) {
	m := otelM()
	if m.rateErr == nil {
		return
	}
	m.rateErr.Add(ctx, 1, metric.WithAttributes(
		attribute.String("bucket", bucket),
		attribute.String("backend", backend),
	))
}

// wsConnDelta adjusts the live-connection gauge.  Callers pair +1 on
// successful upgrade with -1 on close (defer is the idiomatic shape).
func wsConnDelta(ctx context.Context, kind string, delta int64) {
	m := otelM()
	if m.wsActive == nil {
		return
	}
	m.wsActive.Add(ctx, delta, metric.WithAttributes(
		attribute.String("kind", kind),
	))
}

// recordInference is a stable wrapper around the inference counter so
// settle paths in tokenization.go and elsewhere don't have to import
// otel/attribute directly.
func recordInference(ctx context.Context, model, status string) {
	m := otelM()
	if m.infReq == nil {
		return
	}
	m.infReq.Add(ctx, 1, metric.WithAttributes(
		attribute.String("model", model),
		attribute.String("status", status),
	))
}
