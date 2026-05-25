package main

// DPP backbone registry — the canonical list of model identifiers the
// dpp_runtime (Python) knows how to split + serve.  Mirrors the keyword
// heuristic in python/dpp_runtime/backbones.py::detect_pipeline_class and
// the _ADAPTERS table beneath it.  Kept Go-side so the UI can populate a
// model picker without a Python round-trip.
//
// We deliberately list a handful of well-known *reference* model IDs per
// family rather than try to enumerate everything on HuggingFace — the
// runtime accepts arbitrary IDs that resolve to a known PIPELINE_CLASS,
// but the UI just needs sane defaults for a dropdown.

import (
	"encoding/json"
	"net/http"
	"strings"
)

// dppBackbone is one row in /api/dpp/backbones.
type dppBackbone struct {
	ID         string `json:"id"`           // HF model identifier (canonical name passed to dpp_runtime --model)
	Family     string `json:"family"`       // sdxl | sd15 | sd3 | flux | pixart | svd | animatediff | cogvideox | hunyuan_video | mochi | ltx | wan
	Kind       string `json:"kind"`         // image | video
	Pipeline   string `json:"pipeline"`     // diffusers pipeline class
	DefaultW   int    `json:"default_w"`
	DefaultH   int    `json:"default_h"`
	DefaultSteps int  `json:"default_steps"`
	DefaultCFG float64 `json:"default_cfg"`
	Note       string `json:"note,omitempty"`
}

// dppBackboneRegistry is the static list.  Lower-case, no spaces — these
// are the canonical IDs the worker resolves through HF.
var dppBackboneRegistry = []dppBackbone{
	// ── Image, Tier-1 ────────────────────────────────────────────────
	{ID: "stabilityai/sdxl-turbo", Family: "sdxl", Kind: "image",
		Pipeline: "StableDiffusionXLPipeline",
		DefaultW: 512, DefaultH: 512, DefaultSteps: 4, DefaultCFG: 0,
		Note: "distilled — 1–4 steps, cfg=0"},
	{ID: "stabilityai/stable-diffusion-xl-base-1.0", Family: "sdxl", Kind: "image",
		Pipeline: "StableDiffusionXLPipeline",
		DefaultW: 1024, DefaultH: 1024, DefaultSteps: 30, DefaultCFG: 7.5},
	{ID: "runwayml/stable-diffusion-v1-5", Family: "sd15", Kind: "image",
		Pipeline: "StableDiffusionPipeline",
		DefaultW: 512, DefaultH: 512, DefaultSteps: 30, DefaultCFG: 7.5},
	{ID: "PixArt-alpha/PixArt-XL-2-512x512", Family: "pixart", Kind: "image",
		Pipeline: "PixArtAlphaPipeline",
		DefaultW: 512, DefaultH: 512, DefaultSteps: 25, DefaultCFG: 4.5,
		Note: "T5 encoder is large (~9GB) — place TE on a rig with ≥16GB free"},
	{ID: "PixArt-alpha/PixArt-Sigma-XL-2-512-MS", Family: "pixart", Kind: "image",
		Pipeline: "PixArtSigmaPipeline",
		DefaultW: 512, DefaultH: 512, DefaultSteps: 25, DefaultCFG: 4.5},
	{ID: "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", Family: "pixart", Kind: "image",
		Pipeline: "PixArtSigmaPipeline",
		DefaultW: 1024, DefaultH: 1024, DefaultSteps: 25, DefaultCFG: 4.5},
	{ID: "stabilityai/stable-diffusion-3-medium-diffusers", Family: "sd3", Kind: "image",
		Pipeline: "StableDiffusion3Pipeline",
		DefaultW: 1024, DefaultH: 1024, DefaultSteps: 28, DefaultCFG: 7.0,
		Note: "T5+CLIP-L+CLIP-G text stack; needs HF auth"},
	{ID: "black-forest-labs/FLUX.1-schnell", Family: "flux", Kind: "image",
		Pipeline: "FluxPipeline",
		DefaultW: 1024, DefaultH: 1024, DefaultSteps: 4, DefaultCFG: 0,
		Note: "distilled — 1–4 steps, guidance via cfg_split off"},
	{ID: "black-forest-labs/FLUX.1-dev", Family: "flux", Kind: "image",
		Pipeline: "FluxPipeline",
		DefaultW: 1024, DefaultH: 1024, DefaultSteps: 28, DefaultCFG: 3.5,
		Note: "full FLUX — 24–32 steps; needs HF auth"},

	// ── Video, Tier-2 ────────────────────────────────────────────────
	{ID: "stabilityai/stable-video-diffusion-img2vid-xt", Family: "svd", Kind: "video",
		Pipeline: "StableVideoDiffusionPipeline",
		DefaultW: 1024, DefaultH: 576, DefaultSteps: 25, DefaultCFG: 1.0,
		Note: "img2vid — needs init_image"},
	{ID: "guoyww/animatediff-motion-adapter-v1-5-2", Family: "animatediff", Kind: "video",
		Pipeline: "AnimateDiffPipeline",
		DefaultW: 512, DefaultH: 512, DefaultSteps: 20, DefaultCFG: 7.5,
		Note: "uses SD1.5 base + motion adapter"},
	{ID: "THUDM/CogVideoX-5b", Family: "cogvideox", Kind: "video",
		Pipeline: "CogVideoXPipeline",
		DefaultW: 720, DefaultH: 480, DefaultSteps: 50, DefaultCFG: 6.0,
		Note: "5B params; 49 frames at 8 fps"},
	{ID: "tencent/HunyuanVideo", Family: "hunyuan_video", Kind: "video",
		Pipeline: "HunyuanVideoPipeline",
		DefaultW: 1280, DefaultH: 720, DefaultSteps: 30, DefaultCFG: 6.0,
		Note: "13B params; very heavy — DPP across ≥2 24GB rigs recommended"},
	{ID: "genmo/mochi-1-preview", Family: "mochi", Kind: "video",
		Pipeline: "MochiPipeline",
		DefaultW: 848, DefaultH: 480, DefaultSteps: 64, DefaultCFG: 4.5,
		Note: "10B params"},
	{ID: "Lightricks/LTX-Video", Family: "ltx", Kind: "video",
		Pipeline: "LTXPipeline",
		DefaultW: 768, DefaultH: 512, DefaultSteps: 40, DefaultCFG: 3.0,
		Note: "2B params; fast"},
	{ID: "Wan-AI/Wan2.1-T2V-14B", Family: "wan", Kind: "video",
		Pipeline: "WanPipeline",
		DefaultW: 1280, DefaultH: 720, DefaultSteps: 50, DefaultCFG: 5.0,
		Note: "14B params; heavy"},
}

// dppFamilyForModel returns the family tag for an arbitrary model
// identifier using the same keyword heuristic as detect_pipeline_class.
// Empty string means "not a known DPP backbone."
//
// This is the single source of truth for "is this model DPP-eligible?"
// auto-routing in handleComfyGenerate consults it.
func dppFamilyForModel(model string) string {
	if model == "" {
		return ""
	}
	// Exact match against the registry wins (preserves the registered family).
	for _, b := range dppBackboneRegistry {
		if b.ID == model {
			return b.Family
		}
	}
	m := strings.ToLower(model)
	switch {
	case strings.Contains(m, "stable-video-diffusion"), strings.HasPrefix(m, "svd"):
		return "svd"
	case strings.Contains(m, "animatediff"), strings.Contains(m, "animate-diff"):
		return "animatediff"
	case strings.Contains(m, "cogvideox"), strings.Contains(m, "cogvideo"):
		return "cogvideox"
	case strings.Contains(m, "hunyuanvideo"), strings.Contains(m, "hunyuan-video"):
		return "hunyuan_video"
	case strings.Contains(m, "mochi"):
		return "mochi"
	case strings.Contains(m, "ltx-video"), strings.Contains(m, "ltxvideo"),
		strings.HasPrefix(m, "ltx/"):
		return "ltx"
	case strings.Contains(m, "wan2"), strings.HasPrefix(m, "wan/"),
		strings.HasPrefix(m, "wan-ai/"):
		return "wan"
	case strings.Contains(m, "flux"):
		return "flux"
	case strings.Contains(m, "stable-diffusion-3"),
		strings.HasPrefix(m, "sd3"), strings.Contains(m, "sd-3"):
		return "sd3"
	case strings.Contains(m, "pixart"):
		return "pixart"
	case (strings.Contains(m, "stable-diffusion") || strings.Contains(m, "sdxl")) &&
		strings.Contains(m, "xl"):
		return "sdxl"
	case strings.Contains(m, "stable-diffusion"), strings.HasPrefix(m, "sd"):
		return "sd15"
	}
	return ""
}

// isDPPEligible — convenience wrapper.
func isDPPEligible(model string) bool {
	return dppFamilyForModel(model) != ""
}

// handleDPPBackbones — GET /api/dpp/backbones.
//
// Returns the static registry plus, for each registered backbone, whether
// any rig in any of the user's pools has the model loaded in its comfy
// cache (best-effort hint — DPP runs through HF cache, not comfy_models,
// so absence here doesn't block the run).
func (s *server) handleDPPBackbones(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "auth required")
		return
	}

	// Pull every comfy model the user has visibility into.  Used to set
	// the `comfy_registered` hint on each row — purely informational.
	registered := map[string]bool{}
	rows, err := s.dbQuery(
		`SELECT DISTINCT name FROM comfy_models`,
	)
	if err == nil {
		defer rows.Close()
		for rows.Next() {
			var name string
			if err := rows.Scan(&name); err == nil {
				registered[name] = true
			}
		}
	}

	type backboneWithRegistered struct {
		dppBackbone
		ComfyRegistered bool `json:"comfy_registered"`
	}
	out := make([]backboneWithRegistered, 0, len(dppBackboneRegistry))
	for _, b := range dppBackboneRegistry {
		out = append(out, backboneWithRegistered{
			dppBackbone:     b,
			ComfyRegistered: registered[b.ID],
		})
	}

	_ = u
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"backbones": out,
	})
}
