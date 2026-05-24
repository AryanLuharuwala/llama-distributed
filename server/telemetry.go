package main

// OpenTelemetry SDK wiring.
//
// Activated by setting OTEL_EXPORTER_OTLP_ENDPOINT (e.g.
// http://otel-collector:4318).  When unset, setupTelemetry returns a
// no-op shutdown and registers no providers — every metric/trace call
// becomes free.  This keeps the single-container deploy (no collector
// sidecar required) identical to before.
//
// Three signals:
//   - Traces:  HTTP requests via otelhttp.NewHandler in router_wrap.go.
//              Custom spans can be opened via otel.Tracer("dist-server")
//              wherever a flow crosses a meaningful boundary.
//   - Metrics: a few hand-rolled counters/gauges exposed via the global
//              meter (see metrics.go) — rate-limit allow/deny, active WS
//              conns, inference requests.  Plus otelhttp's
//              http.server.duration + http.server.active_requests.
//   - Logs:    OTel Go logs SDK is still maturing; we keep log.Printf.
//              Operators bridge via the collector's filelog receiver if
//              they want structured ingestion.
//
// Resource attributes are mandatory for useful dashboards: service.name,
// service.version, service.instance.id (hostname or pod IP), and the
// deployment.environment hint.  Set DIST_OTEL_ENV=prod|staging|dev to
// distinguish revisions in the same backend.

import (
	"context"
	"errors"
	"fmt"
	"os"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/otlp/otlpmetric/otlpmetrichttp"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.26.0"
)

// telemetryShutdown is a single function the caller defers; it tears
// down trace + metric providers, flushing any buffered exports.  A
// nil-safe call (the no-op return when OTel is disabled is just a func
// that returns nil immediately).
type telemetryShutdown func(context.Context) error

// setupTelemetry installs global trace + metric providers when an OTLP
// endpoint is configured.  Returns a shutdown closer the caller must
// invoke during graceful shutdown so the last batch is flushed.
//
// Reads OTEL_EXPORTER_OTLP_ENDPOINT (the SDK's standard env var, kept
// intentionally — operators already know it).  Also honors
// DIST_OTEL_SERVICE (default "dist-server"), DIST_OTEL_ENV (default
// "prod"), and DIST_OTEL_VERSION (default "unknown" — set in your build
// via -ldflags "-X main.BuildVersion=...").
func setupTelemetry(ctx context.Context) (telemetryShutdown, error) {
	endpoint := os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
	if endpoint == "" {
		// Disabled.  Global providers stay no-op; metrics + tracer
		// instances created from otel.Meter / otel.Tracer become safe
		// no-ops.  No goroutines, no exporters, no allocations.
		return func(context.Context) error { return nil }, nil
	}

	res, err := buildOtelResource(ctx)
	if err != nil {
		return nil, fmt.Errorf("otel resource: %w", err)
	}

	// otlptracehttp + otlpmetrichttp both auto-pick up
	// OTEL_EXPORTER_OTLP_ENDPOINT and OTEL_EXPORTER_OTLP_HEADERS from
	// the env (insecure by default for http://, TLS for https://).
	// Passing WithEndpointURL explicitly so the choice is visible at the
	// callsite rather than buried in env-var magic.
	traceExp, err := otlptracehttp.New(ctx,
		otlptracehttp.WithEndpointURL(endpoint),
	)
	if err != nil {
		return nil, fmt.Errorf("otel trace exporter: %w", err)
	}
	tp := sdktrace.NewTracerProvider(
		sdktrace.WithResource(res),
		sdktrace.WithBatcher(traceExp,
			sdktrace.WithBatchTimeout(5*time.Second),
			sdktrace.WithMaxExportBatchSize(512),
		),
		// Sample 10% in prod; full sample in dev.  Operator can override
		// via OTEL_TRACES_SAMPLER / OTEL_TRACES_SAMPLER_ARG if they want.
		sdktrace.WithSampler(sdktrace.ParentBased(
			sdktrace.TraceIDRatioBased(envFloat("DIST_OTEL_SAMPLE", 0.1)),
		)),
	)
	otel.SetTracerProvider(tp)

	metricExp, err := otlpmetrichttp.New(ctx,
		otlpmetrichttp.WithEndpointURL(endpoint),
	)
	if err != nil {
		_ = tp.Shutdown(ctx)
		return nil, fmt.Errorf("otel metric exporter: %w", err)
	}
	mp := sdkmetric.NewMeterProvider(
		sdkmetric.WithResource(res),
		sdkmetric.WithReader(sdkmetric.NewPeriodicReader(metricExp,
			sdkmetric.WithInterval(15*time.Second),
		)),
	)
	otel.SetMeterProvider(mp)

	return func(ctx context.Context) error {
		// 5s ceiling — past that we'd hold up SIGTERM-driven shutdowns
		// that already have their own ten-second budget upstream.
		ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
		defer cancel()
		return errors.Join(tp.Shutdown(ctx), mp.Shutdown(ctx))
	}, nil
}

func buildOtelResource(ctx context.Context) (*resource.Resource, error) {
	host, _ := os.Hostname()
	if host == "" {
		host = "unknown"
	}
	svc := envOr("DIST_OTEL_SERVICE", "dist-server")
	ver := envOr("DIST_OTEL_VERSION", "unknown")
	env := envOr("DIST_OTEL_ENV", "prod")

	return resource.New(ctx,
		resource.WithFromEnv(),
		resource.WithTelemetrySDK(),
		resource.WithProcess(),
		resource.WithAttributes(
			semconv.ServiceName(svc),
			semconv.ServiceVersion(ver),
			semconv.ServiceInstanceID(host),
			semconv.DeploymentEnvironment(env),
		),
	)
}

// envFloat reads a base-10 float from env, falling back to d on unset
// or parse error.  Local helper so telemetry.go doesn't reach into
// strconv just for this.
func envFloat(k string, d float64) float64 {
	v := os.Getenv(k)
	if v == "" {
		return d
	}
	var f float64
	if _, err := fmt.Sscanf(v, "%f", &f); err != nil {
		return d
	}
	return f
}
