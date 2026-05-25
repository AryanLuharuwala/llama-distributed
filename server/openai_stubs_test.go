package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// TestOpenAIStubsShape verifies the 501 envelopes match the SDKs.
// OpenAI shape: {error: {message, type, param, code}}.
// Anthropic shape: {type:"error", error:{type, message}}.
func TestOpenAIStubsShape(t *testing.T) {
	mux := http.NewServeMux()
	s := &server{}
	s.registerOpenAIStubs(mux)

	t.Run("oai_embeddings_envelope", func(t *testing.T) {
		req := httptest.NewRequest("POST", "/v1/embeddings", strings.NewReader(`{"model":"x","input":"y"}`))
		rec := httptest.NewRecorder()
		mux.ServeHTTP(rec, req)
		if rec.Code != http.StatusNotImplemented {
			t.Fatalf("status = %d, want 501", rec.Code)
		}
		var body map[string]any
		if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
			t.Fatalf("bad json: %v: %s", err, rec.Body.String())
		}
		errObj, ok := body["error"].(map[string]any)
		if !ok {
			t.Fatalf("missing error object: %v", body)
		}
		// SDK requires "type" present even if null param/code.
		if _, ok := errObj["type"]; !ok {
			t.Errorf("error.type missing")
		}
		if _, ok := errObj["message"]; !ok {
			t.Errorf("error.message missing")
		}
		// param/code must be present (even as null) for openai-python
		// pydantic validator to accept the shape on newer versions.
		if _, ok := errObj["param"]; !ok {
			t.Errorf("error.param key missing")
		}
		if _, ok := errObj["code"]; !ok {
			t.Errorf("error.code key missing")
		}
	})

	t.Run("anthropic_messages_envelope", func(t *testing.T) {
		req := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(`{"model":"x","messages":[]}`))
		rec := httptest.NewRecorder()
		mux.ServeHTTP(rec, req)
		if rec.Code != http.StatusNotImplemented {
			t.Fatalf("status = %d, want 501", rec.Code)
		}
		var body map[string]any
		if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
			t.Fatalf("bad json: %v", err)
		}
		if body["type"] != "error" {
			t.Errorf("top-level type = %v, want 'error'", body["type"])
		}
		errObj, ok := body["error"].(map[string]any)
		if !ok {
			t.Fatalf("missing inner error object")
		}
		if errObj["type"] == nil || errObj["message"] == nil {
			t.Errorf("anthropic error missing type/message: %v", errObj)
		}
	})

	// Per-pool slug variants must respond identically — same SDKs hit
	// either depending on whether the user is using the apex base URL
	// or the per-pool subdomain.
	t.Run("slug_variant_routes_too", func(t *testing.T) {
		req := httptest.NewRequest("POST", "/v1/mypool/embeddings", strings.NewReader(`{}`))
		rec := httptest.NewRecorder()
		mux.ServeHTTP(rec, req)
		if rec.Code != http.StatusNotImplemented {
			t.Fatalf("slug-form status = %d, want 501; body=%s", rec.Code, rec.Body.String())
		}
	})

	// /v1/models/{id} is NOT stubbed because it would conflict at
	// startup with the existing real /v1/{slug}/models route — both
	// are 3-segment patterns that match /v1/models/foo with neither
	// being more specific.  Document the resulting 404 so a future
	// reader doesn't add it back without thinking about the conflict.
	t.Run("model_retrieve_404_not_stubbed", func(t *testing.T) {
		req := httptest.NewRequest("GET", "/v1/models/llama-3-8b", nil)
		rec := httptest.NewRecorder()
		mux.ServeHTTP(rec, req)
		if rec.Code == http.StatusNotImplemented {
			t.Errorf("got 501 — see comment: this path conflicts with /v1/{slug}/models so we deliberately don't stub it")
		}
	})

	// Sanity: an unrelated path under /v1 that we have NOT stubbed
	// should 404 from the bare mux (no NotFoundHandler set), proving
	// our stubs cover specific routes rather than wildcarding /v1/*.
	t.Run("unstubbed_path_falls_through", func(t *testing.T) {
		req := httptest.NewRequest("GET", "/v1/wholly/made-up", nil)
		rec := httptest.NewRecorder()
		mux.ServeHTTP(rec, req)
		if rec.Code == http.StatusNotImplemented {
			t.Errorf("status = 501 — would mean we accidentally wildcarded /v1/*")
		}
	})
}
