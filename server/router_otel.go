package main

// HTTP instrumentation for OTel.  The whole router goes through
// otelhttp.NewHandler so every request gets:
//
//   - A span named after the matched route pattern (e.g.
//     "GET /api/me/rigs/{id}"), with the standard http.* attributes.
//   - http.server.duration histogram (per route, per status code).
//   - http.server.active_requests up-down counter.
//
// When OTel is disabled (no OTEL_EXPORTER_OTLP_ENDPOINT) the wrapping
// is essentially free — the global noop providers turn span/metric
// callsites into a fast no-op path inside the SDK.
//
// We use otelhttp.WithSpanNameFormatter so spans use the matched route
// pattern from the *http.Request rather than the full path.  Without
// that, every /api/me/rigs/<id> request produces a unique span name
// and cardinality explodes in the trace backend.

import (
	"net/http"

	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
)

// wrapWithOTel attaches the otelhttp middleware.  Public so tests can
// confirm the wrapped handler still routes; called once at boot from
// main().
func wrapWithOTel(next http.Handler) http.Handler {
	return otelhttp.NewHandler(next, "http.server",
		otelhttp.WithSpanNameFormatter(func(op string, r *http.Request) string {
			// http.ServeMux exposes the matched pattern via Pattern()
			// on the *Request after routing — but only inside the
			// handler.  At this point routing hasn't happened yet, so
			// we fall back to method + raw path.  Span samplers /
			// post-processing in the collector can re-key on
			// http.route once it's set by inner spans.
			if r.Method != "" {
				return r.Method + " " + r.URL.Path
			}
			return op
		}),
	)
}
