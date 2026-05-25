package main

// OpenAI / Anthropic compatibility stubs.
//
// We implement /v1/models, /v1/chat/completions, /v1/images/generations
// for real (see openai.go).  For the rest of the surface area the OpenAI
// SDKs probe, we serve a structured 501 in OpenAI's error shape so:
//
//   1. SDK consumers get a predictable shape (error.type=not_supported)
//      instead of an HTML 404, which crashes openai-python's response
//      validator.
//   2. The /v1/{slug}/* variants work the same way under per-pool
//      subdomains and the dev path-form base URLs.
//   3. /v1/messages (Anthropic) returns Anthropic's own error envelope,
//      not OpenAI's, so the Anthropic SDK doesn't blow up either.
//
// If/when one of these endpoints gains real backing, drop the stub line
// here and add a real HandleFunc in server.go — the stub registration
// will be shadowed by the real one if registered first.  We keep stub
// registrations in their own function so the audit trail is obvious.

import (
	"encoding/json"
	"io"
	"net/http"
)

// oaiErr writes an OpenAI-shaped error envelope.  Matches the
// {"error": {"message": "...", "type": "...", "param": null, "code": null}}
// shape that openai-python / openai-node validate against.
func oaiErr(w http.ResponseWriter, status int, kind, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{
			"message": msg,
			"type":    kind,
			"param":   nil,
			"code":    nil,
		},
	})
}

// anthropicErr writes an Anthropic-shaped error envelope.  Matches the
// {"type":"error", "error":{"type":"...","message":"..."}} shape that
// the Anthropic SDK validates against.
func anthropicErr(w http.ResponseWriter, status int, kind, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"type": "error",
		"error": map[string]any{
			"type":    kind,
			"message": msg,
		},
	})
}

// handleOAINotImplemented is the default stub for OpenAI endpoints we
// don't (yet) implement.  Drains the request body up to 1 MiB so we
// don't leave the connection hanging mid-upload, then 501s.
func (s *server) handleOAINotImplemented(feature string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, io.LimitReader(r.Body, 1<<20))
		oaiErr(w, http.StatusNotImplemented, "not_supported",
			feature+" is not supported by this provider — see /openapi.json for the supported surface")
	}
}

// handleAnthropicMessages stubs Anthropic's /v1/messages.  Direction of
// travel: we may wire this through the same dispatcher as
// handleOAIChat once message-shape translation is settled.  Until then,
// a 501 in Anthropic's envelope keeps the Anthropic SDK happy.
func (s *server) handleAnthropicMessages(w http.ResponseWriter, r *http.Request) {
	_, _ = io.Copy(io.Discard, io.LimitReader(r.Body, 1<<20))
	anthropicErr(w, http.StatusNotImplemented, "not_supported_error",
		"the Anthropic Messages API is not yet implemented on this provider — use /v1/chat/completions (OpenAI-compatible)")
}

// registerOpenAIStubs wires the stub routes.  Called from server.go's
// router setup, after the real /v1/chat/completions etc. registrations
// so the real handlers win on shared prefixes.
//
// Go 1.22 ServeMux refuses to register two patterns that would both
// match the same path without one being strictly more specific — e.g.
// `GET /v1/files/{id}` conflicts with `GET /v1/{slug}/files` because
// the path "/v1/files/files" matches both with no winner.
//
// So we only stub the *collection* endpoints (paths with no inner
// {id} segment).  SDK calls to a per-resource probe (files.retrieve,
// batches.retrieve, responses.retrieve, models.retrieve) will 404 from
// the bare mux — which is still a "not implemented" signal, just not a
// pretty one.  The SDKs we care about (openai-python/-node, Anthropic)
// will surface the 404 as an APIError without crashing the call site.
//
// When any of these endpoints gain a real backend, drop the stub from
// this list AND register the matching {id} apex/slug patterns in
// server.go directly (the conflict goes away because the real route is
// strictly more specific than the collection slug-form).
func (s *server) registerOpenAIStubs(mux *http.ServeMux) {
	stubs := []struct{ method, path, feature string }{
		// Embeddings & completions surface.
		{"POST", "/v1/embeddings", "embeddings"},
		{"POST", "/v1/completions", "legacy completions"},
		{"POST", "/v1/moderations", "moderations"},

		// Audio.
		{"POST", "/v1/audio/transcriptions", "audio.transcriptions"},
		{"POST", "/v1/audio/translations", "audio.translations"},
		{"POST", "/v1/audio/speech", "audio.speech"},

		// Files / Batches / Responses — collection endpoints only.
		{"POST", "/v1/files", "files.upload"},
		{"GET", "/v1/files", "files.list"},
		{"POST", "/v1/batches", "batches.create"},
		{"GET", "/v1/batches", "batches.list"},
		{"POST", "/v1/responses", "responses.create"},

		// Images — edits / variations unsupported (we run real
		// generations via ComfyUI).
		{"POST", "/v1/images/edits", "images.edits"},
		{"POST", "/v1/images/variations", "images.variations"},
	}
	for _, st := range stubs {
		h := s.handleOAINotImplemented(st.feature)
		mux.HandleFunc(st.method+" "+st.path, h)
		mux.HandleFunc(st.method+" /v1/{slug}"+st.path[len("/v1"):], h)
	}

	// Anthropic.  Two routes: apex + per-pool slug.
	mux.HandleFunc("POST /v1/messages", s.handleAnthropicMessages)
	mux.HandleFunc("POST /v1/{slug}/messages", s.handleAnthropicMessages)
}
