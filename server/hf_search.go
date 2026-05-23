package main

// HuggingFace search proxy.
//
// The browser playground (/playground → Discover tab) needs a tag-aware,
// category-aware model search.  Hitting huggingface.co directly from the
// page would force CORS and leak the user's HF token; instead we proxy
// through the server, attach the per-user HF bearer when available, and
// normalise the response so the UI can consume a flat list.

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

type hfSearchRow struct {
	ID           string   `json:"id"`
	Author       string   `json:"author,omitempty"`
	Downloads    int64    `json:"downloads"`
	Likes        int64    `json:"likes"`
	LastModified string   `json:"last_modified,omitempty"`
	PipelineTag  string   `json:"pipeline_tag,omitempty"`
	LibraryName  string   `json:"library,omitempty"`
	Tags         []string `json:"tags,omitempty"`
	Private      bool     `json:"private,omitempty"`
	Gated        bool     `json:"gated,omitempty"`
}

// hfSearchUpstream is the subset of huggingface.co/api/models we use.
// HF returns `gated` as either a bool (false) or a string ("auto"/"manual")
// so we decode it into json.RawMessage and coerce below.
type hfSearchUpstream struct {
	ID           string          `json:"id"`
	Author       string          `json:"author"`
	Downloads    int64           `json:"downloads"`
	Likes        int64           `json:"likes"`
	LastModified string          `json:"lastModified"`
	PipelineTag  string          `json:"pipeline_tag"`
	LibraryName  string          `json:"library_name"`
	Tags         []string        `json:"tags"`
	Private      bool            `json:"private"`
	Gated        json.RawMessage `json:"gated"`
}

// GET /api/hf/search
//
// Query params:
//   q              free-text search; matches repo id / author / tags
//   pipeline_tag   one HF pipeline_tag (text-generation, text-to-image, …)
//   library        one HF library_name (gguf, transformers, diffusers, …)
//   tags           comma-separated tag filter (e.g. gguf,llama,instruct)
//   sort           downloads | likes | lastModified | trendingScore
//   direction      -1 (desc, default) | 1 (asc)
//   limit          1..100, default 25
//
// All filters are optional.  An empty query just returns the top results
// for the given sort.
func (s *server) handleHFSearch(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}

	q := strings.TrimSpace(r.URL.Query().Get("q"))
	if len(q) > 256 {
		writeErr(w, 400, "q too long")
		return
	}
	pipelineTag := strings.TrimSpace(r.URL.Query().Get("pipeline_tag"))
	library := strings.TrimSpace(r.URL.Query().Get("library"))
	tags := strings.TrimSpace(r.URL.Query().Get("tags"))

	sort := strings.TrimSpace(r.URL.Query().Get("sort"))
	switch sort {
	case "", "downloads", "likes", "lastModified", "trendingScore":
		// ok
	default:
		writeErr(w, 400, "invalid sort")
		return
	}
	if sort == "" {
		sort = "trendingScore"
	}
	direction := r.URL.Query().Get("direction")
	if direction != "1" && direction != "-1" {
		direction = "-1"
	}
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	if limit <= 0 {
		limit = 25
	}
	if limit > 100 {
		limit = 100
	}

	// Build the upstream query.
	v := url.Values{}
	if q != "" {
		v.Set("search", q)
	}
	if pipelineTag != "" {
		v.Set("pipeline_tag", pipelineTag)
	}
	if library != "" {
		v.Set("library", library)
	}
	if tags != "" {
		// HF expects multiple ?filter= params for AND-of-tags.
		for _, t := range strings.Split(tags, ",") {
			t = strings.TrimSpace(t)
			if t == "" {
				continue
			}
			v.Add("filter", t)
		}
	}
	v.Set("sort", sort)
	v.Set("direction", direction)
	v.Set("limit", strconv.Itoa(limit))
	v.Set("full", "false")
	// HF returns these fields by default; ask explicitly so library_name +
	// downloads + likes are always present.
	v.Add("config", "false")
	v.Add("expand[]", "downloads")
	v.Add("expand[]", "likes")
	v.Add("expand[]", "lastModified")
	v.Add("expand[]", "pipeline_tag")
	v.Add("expand[]", "library_name")
	v.Add("expand[]", "tags")
	v.Add("expand[]", "private")
	v.Add("expand[]", "gated")

	ctx, cancel := context.WithTimeout(r.Context(), 20*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, "GET",
		hfAPIBase+"/api/models?"+v.Encode(), nil)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	req.Header.Set("Accept", "application/json")
	if tok := s.userHFToken(u.ID); tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}

	resp, err := hfClient.Do(req)
	if err != nil {
		writeErr(w, 502, "hf upstream: "+err.Error())
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<10))
		writeJSON(w, resp.StatusCode, map[string]any{
			"error":         "hf returned " + resp.Status,
			"upstream_body": string(b),
		})
		return
	}

	var rows []hfSearchUpstream
	if err := json.NewDecoder(io.LimitReader(resp.Body, 4<<20)).Decode(&rows); err != nil {
		writeErr(w, 502, "hf decode: "+err.Error())
		return
	}

	out := make([]hfSearchRow, 0, len(rows))
	for _, r := range rows {
		gated := false
		if len(r.Gated) > 0 {
			var b bool
			if json.Unmarshal(r.Gated, &b) == nil {
				gated = b
			} else {
				var s string
				if json.Unmarshal(r.Gated, &s) == nil && s != "" && s != "false" {
					gated = true
				}
			}
		}
		out = append(out, hfSearchRow{
			ID:           r.ID,
			Author:       r.Author,
			Downloads:    r.Downloads,
			Likes:        r.Likes,
			LastModified: r.LastModified,
			PipelineTag:  r.PipelineTag,
			LibraryName:  r.LibraryName,
			Tags:         r.Tags,
			Private:      r.Private,
			Gated:        gated,
		})
	}

	writeJSON(w, 200, map[string]any{
		"query": map[string]any{
			"q": q, "pipeline_tag": pipelineTag, "library": library,
			"tags": tags, "sort": sort, "direction": direction, "limit": limit,
		},
		"results": out,
	})
}
