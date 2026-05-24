package main

// Vercel-style per-pool subdomains + OpenAI-compatible endpoints.
//
// Every pool gets a URL-safe slug.  When the server is configured with
// DIST_APEX_HOST=surds.co.in (default), requests to
//
//   <slug>.surds.co.in/v1/chat/completions
//   <slug>.surds.co.in/v1/models
//
// are routed to the matching pool.  Authentication is via
// `Authorization: Bearer <api-key>` (issued from the dashboard).  The
// endpoints speak the OpenAI chat-completion shape so any OpenAI SDK can
// point its base_url at the pool URL and Just Work.
//
// Local dev: apex is typically "localhost:8080", which can't have
// subdomains, so the router falls back to reading `?pool=<slug>` or the
// `X-Pool-Slug` header when the Host matches the apex verbatim.

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"
	"unicode"
)

// ─── Slug generation ───────────────────────────────────────────────────────

// slugify converts a display name to a URL-safe slug: lowercase, hyphens
// between runs of alphanumerics, trimmed, max 40 chars.
func slugify(s string) string {
	var b strings.Builder
	lastDash := true
	for _, r := range strings.ToLower(s) {
		switch {
		case unicode.IsLetter(r) || unicode.IsDigit(r):
			b.WriteRune(r)
			lastDash = false
		case !lastDash:
			b.WriteByte('-')
			lastDash = true
		}
	}
	out := strings.Trim(b.String(), "-")
	if len(out) > 40 {
		out = strings.TrimRight(out[:40], "-")
	}
	return out
}

// userHandle returns the stable per-user slug component: github_login if
// present, otherwise a slug of display_name, otherwise "u<id>".
func userHandle(u *user) string {
	if u.GithubLogin != "" {
		return slugify(u.GithubLogin)
	}
	if h := slugify(u.DisplayName); h != "" {
		return h
	}
	return "u" + strconv.FormatInt(u.ID, 10)
}

// pickPoolSlug picks a slug for a new (or rename-target) pool, ensuring
// uniqueness by suffixing -2, -3, … when needed.  ownerHandle is the
// pool owner's user handle.
func (s *server) pickPoolSlug(ownerHandle, poolName string, skipPoolID int64) (string, error) {
	base := slugify(ownerHandle + "-" + poolName)
	if base == "" {
		base = "pool"
	}
	for n := 0; n < 50; n++ {
		candidate := base
		if n > 0 {
			candidate = fmt.Sprintf("%s-%d", base, n+1)
		}
		var existing int64
		err := s.dbQueryRow(
			`SELECT id FROM pools WHERE slug = ? AND id != ?`,
			candidate, skipPoolID,
		).Scan(&existing)
		if errors.Is(err, sql.ErrNoRows) {
			return candidate, nil
		}
		if err != nil {
			return "", err
		}
	}
	return "", errors.New("could not find a free slug after 50 attempts")
}

// backfillSlugs runs once at boot to give any pre-migration pool a slug.
func (s *server) backfillSlugs() error {
	rows, err := s.dbQuery(
		`SELECT p.id, p.name, COALESCE(u.github_login, ''), COALESCE(u.display_name, ''), u.id
		 FROM pools p JOIN users u ON u.id = p.owner_id
		 WHERE p.slug IS NULL OR p.slug = ''`,
	)
	if err != nil {
		return err
	}
	type row struct {
		id      int64
		name    string
		u       user
	}
	var todo []row
	for rows.Next() {
		var r row
		if err := rows.Scan(&r.id, &r.name, &r.u.GithubLogin, &r.u.DisplayName, &r.u.ID); err == nil {
			todo = append(todo, r)
		}
	}
	rows.Close()
	for _, r := range todo {
		slug, err := s.pickPoolSlug(userHandle(&r.u), r.name, r.id)
		if err != nil {
			return err
		}
		if _, err := s.dbExec(`UPDATE pools SET slug = ? WHERE id = ?`, slug, r.id); err != nil {
			return err
		}
	}
	return nil
}

// ─── Host-based pool resolution ────────────────────────────────────────────

// stripPort returns host without the ":port" suffix, if any.
func stripPort(host string) string {
	if i := strings.LastIndex(host, ":"); i >= 0 {
		return host[:i]
	}
	return host
}

// resolvePoolFromHost returns the pool_id for a request, or 0 if no slug
// can be resolved.  Resolution order:
//   1. Subdomain:  <slug>.<apex>/v1/…               (prod)
//   2. Path param: /v1/<slug>/…                     (works behind any host)
//   3. Header:     X-Pool-Slug: <slug>              (dev / curl)
//   4. Query:      ?pool=<slug>                     (dev / curl)
func (s *server) resolvePoolFromHost(r *http.Request) (int64, string) {
	host := stripPort(r.Host)
	apex := stripPort(s.cfg.apexHost)

	var slug string
	if apex != "" && host != apex && strings.HasSuffix(host, "."+apex) {
		slug = strings.TrimSuffix(host, "."+apex)
	}
	if slug == "" {
		slug = r.PathValue("slug") // populated by /v1/{slug}/… mux routes
	}
	if slug == "" {
		slug = r.Header.Get("X-Pool-Slug")
	}
	if slug == "" {
		slug = r.URL.Query().Get("pool")
	}
	if slug == "" {
		return 0, ""
	}

	var pid int64
	err := s.dbQueryRow(`SELECT id FROM pools WHERE slug = ?`, slug).Scan(&pid)
	if err != nil {
		return 0, slug
	}
	return pid, slug
}

// ─── API key auth ──────────────────────────────────────────────────────────

const apiKeyPrefix = "sk-dist-"

// mintAPIKey inserts a fresh key for a user and returns (plaintext, id).
// The plaintext is shown to the user exactly once.
func (s *server) mintAPIKey(userID int64, label string) (string, int64, error) {
	raw := make([]byte, 24)
	if _, err := rand.Read(raw); err != nil {
		return "", 0, err
	}
	plain := apiKeyPrefix + hex.EncodeToString(raw)
	sum := sha256.Sum256([]byte(plain))
	hash := hex.EncodeToString(sum[:])
	prefix := plain[:12]
	res, err := s.dbExec(
		`INSERT INTO api_keys (user_id, label, prefix, hash, created_at) VALUES (?, ?, ?, ?, ?)`,
		userID, label, prefix, hash, nowUnix(),
	)
	if err != nil {
		return "", 0, err
	}
	id, _ := res.LastInsertId()
	return plain, id, nil
}

// userFromAPIKey looks up the owning user of a Bearer-token-presented key.
func (s *server) userFromAPIKey(key string) (*user, bool) {
	if !strings.HasPrefix(key, apiKeyPrefix) {
		return nil, false
	}
	sum := sha256.Sum256([]byte(key))
	hash := hex.EncodeToString(sum[:])
	var uid int64
	err := s.dbQueryRow(`SELECT user_id FROM api_keys WHERE hash = ?`, hash).Scan(&uid)
	if err != nil {
		return nil, false
	}
	_, _ = s.dbExec(`UPDATE api_keys SET last_used_at = ? WHERE hash = ?`, nowUnix(), hash)

	u := &user{}
	var gh sql.NullString
	err = s.dbQueryRow(
		`SELECT id, github_login, display_name FROM users WHERE id = ?`, uid,
	).Scan(&u.ID, &gh, &u.DisplayName)
	if err != nil {
		return nil, false
	}
	u.GithubLogin = gh.String
	return u, true
}

// bearerFromRequest extracts the Bearer token from Authorization, or "".
func bearerFromRequest(r *http.Request) string {
	h := r.Header.Get("Authorization")
	const p = "Bearer "
	if !strings.HasPrefix(h, p) {
		return ""
	}
	return strings.TrimSpace(h[len(p):])
}

// ─── API key CRUD (dashboard) ──────────────────────────────────────────────

// POST /api/api_keys  { label }  → { key, id, prefix }
func (s *server) handleMintAPIKey(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	var body struct {
		Label string `json:"label"`
	}
	_ = json.NewDecoder(io.LimitReader(r.Body, 4<<10)).Decode(&body)
	plain, id, err := s.mintAPIKey(u.ID, body.Label)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	writeJSON(w, 200, map[string]any{
		"id":     id,
		"key":    plain,        // shown once
		"prefix": plain[:12],
		"label":  body.Label,
	})
}

// GET /api/api_keys
func (s *server) handleListAPIKeys(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	rows, err := s.dbQuery(
		`SELECT id, label, prefix, created_at, COALESCE(last_used_at, 0)
		 FROM api_keys WHERE user_id = ? ORDER BY created_at DESC`, u.ID,
	)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	defer rows.Close()
	type keyOut struct {
		ID         int64  `json:"id"`
		Label      string `json:"label"`
		Prefix     string `json:"prefix"`
		CreatedAt  int64  `json:"created_at"`
		LastUsedAt int64  `json:"last_used_at"`
	}
	var out []keyOut
	for rows.Next() {
		var k keyOut
		if err := rows.Scan(&k.ID, &k.Label, &k.Prefix, &k.CreatedAt, &k.LastUsedAt); err == nil {
			out = append(out, k)
		}
	}
	writeJSON(w, 200, map[string]any{"keys": out})
}

// DELETE /api/api_keys/{id}
func (s *server) handleRevokeAPIKey(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	id, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil {
		writeErr(w, 400, "bad id")
		return
	}
	res, err := s.dbExec(`DELETE FROM api_keys WHERE id = ? AND user_id = ?`, id, u.ID)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	n, _ := res.RowsAffected()
	if n == 0 {
		writeErr(w, 404, "not found")
		return
	}
	writeJSON(w, 200, map[string]any{"revoked": true})
}

// ─── OpenAI-compatible endpoints ───────────────────────────────────────────

// GET /v1/models  (on pool subdomain)
func (s *server) handleOAIModels(w http.ResponseWriter, r *http.Request) {
	poolID, slug := s.resolvePoolFromHost(r)
	if poolID == 0 {
		writeErr(w, 404, "pool not found: "+slug)
		return
	}
	// Auth.  We still scope to keys owned by a pool member so key leakage
	// outside the pool doesn't reveal model names.
	u := s.authOAI(w, r, poolID)
	if u == nil {
		return
	}
	// The pool's bound model (if any) plus any model currently loaded by
	// an online rig in the pool (best-effort; "distpool-default" if none).
	type modelOut struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		OwnedBy string `json:"owned_by"`
	}
	var out []modelOut
	var mid sql.NullInt64
	_ = s.dbQueryRow(`SELECT model_id FROM pools WHERE id = ?`, poolID).Scan(&mid)
	if mid.Valid {
		var name string
		var created int64
		_ = s.dbQueryRow(
			`SELECT name, created_at FROM models WHERE id = ?`, mid.Int64,
		).Scan(&name, &created)
		if name != "" {
			out = append(out, modelOut{
				ID: name, Object: "model", Created: created, OwnedBy: slug,
			})
		}
	}
	if len(out) == 0 {
		out = append(out, modelOut{
			ID: "distpool-default", Object: "model",
			Created: nowUnix(), OwnedBy: slug,
		})
	}
	writeJSON(w, 200, map[string]any{"object": "list", "data": out})
}

// authOAI validates a Bearer API key and returns the caller, or writes
// the error and returns nil.  The caller must be a member of poolID
// (unless the pool is public).
func (s *server) authOAI(w http.ResponseWriter, r *http.Request, poolID int64) *user {
	tok := bearerFromRequest(r)
	if tok == "" {
		writeErr(w, 401, "missing Bearer token")
		return nil
	}
	u, ok := s.userFromAPIKey(tok)
	if !ok {
		writeErr(w, 401, "invalid api key")
		return nil
	}
	vis, _, ok := s.poolVisibility(poolID)
	if !ok {
		writeErr(w, 404, "pool not found")
		return nil
	}
	if _, member := s.userIsMember(poolID, u.ID); !member && vis != "public" {
		writeErr(w, 403, "not a member of pool")
		return nil
	}
	return u
}

// POST /v1/chat/completions
//
// Minimal OpenAI-compat request:
//   { "model": "...",
//     "messages": [{"role":"system"|"user"|"assistant","content":"..."}],
//     "max_tokens": N, "temperature": T, "stream": true|false }
func (s *server) handleOAIChat(w http.ResponseWriter, r *http.Request) {
	poolID, slug := s.resolvePoolFromHost(r)
	if poolID == 0 {
		writeErr(w, 404, "pool not found: "+slug)
		return
	}
	u := s.authOAI(w, r, poolID)
	if u == nil {
		return
	}

	var body struct {
		Model       string          `json:"model"`
		Messages    []oaiMsg        `json:"messages"`
		MaxTokens   *int            `json:"max_tokens"`
		Temperature *float64        `json:"temperature"`
		Stream      bool            `json:"stream"`
		RAG         *chatRAGOpts    `json:"rag"`
		Memory      *chatMemoryOpts `json:"memory"`
		Tools       *chatToolsOpts  `json:"tools"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 256<<10)).Decode(&body); err != nil {
		writeErr(w, 400, "bad json")
		return
	}
	if len(body.Messages) == 0 {
		writeErr(w, 400, "messages required")
		return
	}
	// Capture the last user turn before augmentation so we can summarise
	// the (user → assistant) pair after the response lands.
	origUserTurn := lastUserContent(body.Messages)

	augmented, augErr := s.augmentMessages(r.Context(), u.ID, body.Messages, body.RAG, body.Memory)
	if augErr != nil {
		writeErr(w, 400, augErr.Error())
		return
	}
	body.Messages = augmented

	// Tool manifest is a separate prepend so its scope is obvious to the
	// model: tools are an instruction frame, not retrieved context.
	if !body.Tools.isEmpty() {
		manifest, mErr := s.buildToolsManifest(u.ID, body.Tools)
		if mErr != nil {
			writeErr(w, 400, mErr.Error())
			return
		}
		if manifest != "" {
			body.Messages = append([]oaiMsg{{Role: "system", Content: manifest}}, body.Messages...)
		}
	}
	maxTokens := 256
	if body.MaxTokens != nil && *body.MaxTokens > 0 {
		maxTokens = *body.MaxTokens
	}
	// Pointer-default so the caller can ask for temperature=0 (greedy) without
	// the server silently substituting 0.7.
	temperature := 0.7
	if body.Temperature != nil {
		temperature = *body.Temperature
	}
	prompt := messagesToPrompt(body.Messages)

	// Rate limit.
	ok, policy, snap := s.reserveRequestSlot(u.ID)
	if !ok {
		s.logInference(u.ID, poolID, 0, "", 0, 0, "rate_limit")
		writeJSON(w, 429, map[string]any{
			"error": map[string]any{
				"message": "rate limit exceeded",
				"type":    "rate_limit_error",
				"policy":  policy,
				"usage":   snap,
			},
		})
		return
	}
	slotBilled := false
	defer func() {
		if !slotBilled {
			s.refundRequestSlot(u.ID)
		}
	}()

	// Attach inference peer.  Pick a rig with a free slot AND reserve it
	// in one atomic step so multi-slot rigs actually receive the batched
	// concurrent traffic instead of bouncing the later requests.
	ip := &inferPeer{
		reqID:    nextReqID(),
		incoming: make(chan *inferChunk, 32),
		closed:   make(chan struct{}),
	}
	ac := s.pickAndReserveRig(poolID, ip)
	if ac == nil {
		s.logInference(u.ID, poolID, 0, "", 0, 0, "no_rig")
		writeErr(w, 503, "no online rigs with free slots")
		return
	}
	// Mark this rig busy for the duration of the request so the next picker
	// call sees current load.
	ac.incInflight()
	defer ac.decInflight()
	defer ac.releaseInferSlot(ip)

	logID := s.logInference(u.ID, poolID, ac.userID, ac.agentID, 0, 0, "running")

	// Mint a signed shard URL for the pool's bound model so the rig can
	// pull weights on demand.  Matches the handleInfer payload shape — the
	// agent reports "pool has no model bound" if shard_url is missing.
	//
	// TODO(pp-chat): this is the single-rig fast path and hardcodes
	// stage-0.gguf, so it only works for models registered with n_stages=1.
	// Multi-stage models (e.g. tinyllama in models-store) need this handler
	// to call s.planPipeline(poolID, ...) and fan the request out per stage,
	// mirroring handleInferPP in pp_route.go.  Tracked as task #20.
	var shardURL, shardFile string
	{
		var modelID *int64
		_ = s.dbQueryRow(`SELECT model_id FROM pools WHERE id = ?`, poolID).Scan(&modelID)
		if modelID != nil && *modelID > 0 {
			shardFile = "stage-0.gguf"
			shardURL = s.mintShardURL(*modelID, shardFile, 15*time.Minute)
		}
	}
	payloadJSON, _ := json.Marshal(map[string]any{
		"model":       body.Model,
		"prompt":      prompt,
		"max_tokens":  maxTokens,
		"temperature": temperature,
		"shard_url":   shardURL,
		"shard_file":  shardFile,
	})
	reqFrame := encodeInferRequest(ip.reqID, uint32(estimateTokens(prompt)), payloadJSON)
	if !ac.sendBin(reqFrame) {
		s.finishInference(logID, 0, 0, "failed")
		writeErr(w, 503, "agent buffer full")
		return
	}

	respID := "chatcmpl-" + newRandomToken(12)
	created := nowUnix()
	modelName := body.Model
	if modelName == "" {
		modelName = "distpool-default"
	}

	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Minute)
	defer cancel()

	if body.Stream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("X-Accel-Buffering", "no")
		flusher, _ := w.(http.Flusher)

		// Prime with a role delta (per OpenAI convention).
		writeOAIDelta(w, flusher, respID, created, modelName, "role", "assistant")

		var inTok, outTok uint32
		var bytesStreamed int
		var streamBuf strings.Builder
		for {
			select {
			case c, ok := <-ip.incoming:
				if !ok {
					// Disconnect mid-stream: peer was closed without a
					// terminating chunk arriving first.  Don't fabricate
					// a successful "done" delta — surface the failure.
					writeOAIErr(w, flusher, "agent disconnected")
					inSafe, outSafe := s.settleTokens(ac.agentID, int(inTok), int(outTok), len(prompt), bytesStreamed)
					s.finishInference(logID, inSafe, outSafe, "failed")
					s.recordTokens(u.ID, inSafe, outSafe)
					return
				}
				inTok, outTok = c.tokIn, c.tokOut
				switch c.kind {
				case chunkKindToken:
					bytesStreamed += len(c.payload)
					streamBuf.Write(c.payload)
					writeOAIDelta(w, flusher, respID, created, modelName, "content", string(c.payload))
				case chunkKindDone:
					// If the model emitted tool_call envelopes and auto_execute
					// was set, invoke them now and stream the structured results
					// as a follow-up delta so SSE clients can render them inline.
					if body.Tools != nil && body.Tools.AutoExecute {
						if calls := extractToolCalls(streamBuf.String()); len(calls) > 0 {
							results := s.executeToolCalls(r.Context(), u.ID, body.Tools, calls)
							if blob, mErr := json.Marshal(map[string]any{"tool_calls": results}); mErr == nil {
								writeOAIDelta(w, flusher, respID, created, modelName, "tool_results", string(blob))
							}
						}
					}
					writeOAIDone(w, flusher, respID, created, modelName)
					inSafe, outSafe := s.settleTokens(ac.agentID, int(inTok), int(outTok), len(prompt), bytesStreamed)
					s.finishInference(logID, inSafe, outSafe, "ok")
					s.recordTokens(u.ID, inSafe, outSafe)
					slotBilled = true
					if body.Memory != nil && body.Memory.SummarizeAfter {
						go s.summarizeChatTurnSafely(u.ID, body.Memory.ConversationID, origUserTurn, streamBuf.String())
					}
					return
				case chunkKindError:
					writeOAIErr(w, flusher, string(c.payload))
					inSafe, outSafe := s.settleTokens(ac.agentID, int(inTok), int(outTok), len(prompt), bytesStreamed)
					s.finishInference(logID, inSafe, outSafe, "failed")
					return
				}
			case <-ctx.Done():
				writeOAIErr(w, flusher, "timeout")
				inSafe, outSafe := s.settleTokens(ac.agentID, int(inTok), int(outTok), len(prompt), bytesStreamed)
				s.finishInference(logID, inSafe, outSafe, "failed")
				return
			}
		}
	}

	// Non-streaming: buffer all tokens, return one JSON response.
	var builder strings.Builder
	var inTok, outTok uint32
	var bytesStreamed int
	for {
		select {
		case c, ok := <-ip.incoming:
			if !ok {
				// Same disconnect-without-done semantics as the streaming
				// path: report the failure instead of returning whatever
				// partial bytes accumulated as if they were complete.
				inSafe, outSafe := s.settleTokens(ac.agentID, int(inTok), int(outTok), len(prompt), bytesStreamed)
				s.finishInference(logID, inSafe, outSafe, "failed")
				writeErr(w, 502, "agent disconnected")
				return
			}
			inTok, outTok = c.tokIn, c.tokOut
			switch c.kind {
			case chunkKindToken:
				bytesStreamed += len(c.payload)
				builder.Write(c.payload)
			case chunkKindDone:
				goto DONE
			case chunkKindError:
				inSafe, outSafe := s.settleTokens(ac.agentID, int(inTok), int(outTok), len(prompt), bytesStreamed)
				s.finishInference(logID, inSafe, outSafe, "failed")
				writeErr(w, 502, string(c.payload))
				return
			}
		case <-ctx.Done():
			inSafe, outSafe := s.settleTokens(ac.agentID, int(inTok), int(outTok), len(prompt), bytesStreamed)
			s.finishInference(logID, inSafe, outSafe, "failed")
			writeErr(w, 504, "timeout")
			return
		}
	}
DONE:
	inSafe, outSafe := s.settleTokens(ac.agentID, int(inTok), int(outTok), len(prompt), bytesStreamed)
	s.finishInference(logID, inSafe, outSafe, "ok")
	s.recordTokens(u.ID, inSafe, outSafe)
	slotBilled = true
	inTok, outTok = uint32(inSafe), uint32(outSafe)
	if body.Memory != nil && body.Memory.SummarizeAfter {
		go s.summarizeChatTurnSafely(u.ID, body.Memory.ConversationID, origUserTurn, builder.String())
	}

	// If the model emitted tool_call envelopes, execute them through the
	// MCP broker (when auto_execute=true) and return both the assistant
	// text (with envelopes stripped) and the structured tool_calls array.
	assistantContent := builder.String()
	var toolResults []chatToolResult
	if !body.Tools.isEmpty() {
		calls := extractToolCalls(assistantContent)
		if len(calls) > 0 {
			assistantContent = stripToolCalls(assistantContent)
			if body.Tools.AutoExecute {
				toolResults = s.executeToolCalls(r.Context(), u.ID, body.Tools, calls)
			} else {
				toolResults = make([]chatToolResult, len(calls))
				for i, c := range calls {
					toolResults[i] = chatToolResult{Call: c}
				}
			}
		}
	}

	message := map[string]any{"role": "assistant", "content": assistantContent}
	if len(toolResults) > 0 {
		message["tool_calls"] = toolResults
	}
	finishReason := "stop"
	if len(toolResults) > 0 && !body.Tools.AutoExecute {
		finishReason = "tool_calls"
	}
	writeJSON(w, 200, map[string]any{
		"id":      respID,
		"object":  "chat.completion",
		"created": created,
		"model":   modelName,
		"choices": []map[string]any{{
			"index":         0,
			"message":       message,
			"finish_reason": finishReason,
		}},
		"usage": map[string]any{
			"prompt_tokens":     inTok,
			"completion_tokens": outTok,
			"total_tokens":      inTok + outTok,
		},
	})
}

type oaiMsg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// messagesToPrompt flattens a chat thread into a single prompt string.
// Very simple — a future step is to delegate to the target model's
// tokenizer/chat template on the agent, but this works end-to-end today.
//
// User-controlled content is scrubbed of `[SYSTEM]`/`[USER]`/`[ASSISTANT]`
// markers so a malicious message can't inject a fake role frame into the
// prompt seen by the model.
func messagesToPrompt(msgs []oaiMsg) string {
	var b strings.Builder
	for _, m := range msgs {
		switch m.Role {
		case "system":
			b.WriteString("[SYSTEM]\n")
		case "user":
			b.WriteString("[USER]\n")
		case "assistant":
			b.WriteString("[ASSISTANT]\n")
		default:
			b.WriteString("[")
			b.WriteString(strings.ToUpper(m.Role))
			b.WriteString("]\n")
		}
		b.WriteString(scrubRoleMarkers(m.Content))
		b.WriteString("\n\n")
	}
	b.WriteString("[ASSISTANT]\n")
	return b.String()
}

// scrubRoleMarkers removes literal role-frame strings from user-supplied
// content so they can't inject fake roles into the flattened prompt.
func scrubRoleMarkers(s string) string {
	for _, m := range []string{"[SYSTEM]", "[USER]", "[ASSISTANT]"} {
		s = strings.ReplaceAll(s, m, "")
	}
	return s
}

// ─── SSE helpers for the OpenAI stream format ──────────────────────────────

func writeOAIDelta(w http.ResponseWriter, fl http.Flusher,
	id string, created int64, model, field, val string) {
	delta := map[string]any{field: val}
	chunk := map[string]any{
		"id":      id,
		"object":  "chat.completion.chunk",
		"created": created,
		"model":   model,
		"choices": []map[string]any{{
			"index":         0,
			"delta":         delta,
			"finish_reason": nil,
		}},
	}
	b, _ := json.Marshal(chunk)
	fmt.Fprintf(w, "data: %s\n\n", b)
	if fl != nil {
		fl.Flush()
	}
}

func writeOAIDone(w http.ResponseWriter, fl http.Flusher, id string, created int64, model string) {
	chunk := map[string]any{
		"id":      id,
		"object":  "chat.completion.chunk",
		"created": created,
		"model":   model,
		"choices": []map[string]any{{
			"index":         0,
			"delta":         map[string]any{},
			"finish_reason": "stop",
		}},
	}
	b, _ := json.Marshal(chunk)
	fmt.Fprintf(w, "data: %s\n\n", b)
	fmt.Fprintf(w, "data: [DONE]\n\n")
	if fl != nil {
		fl.Flush()
	}
}

func writeOAIErr(w http.ResponseWriter, fl http.Flusher, msg string) {
	b, _ := json.Marshal(map[string]any{
		"error": map[string]any{"message": msg, "type": "server_error"},
	})
	fmt.Fprintf(w, "data: %s\n\n", b)
	if fl != nil {
		fl.Flush()
	}
}
