package main

// sd.cpp coverage — which model families the stable-diffusion.cpp C++
// backend (`gpunet-sdcpp-worker`) can actually run. Mirrors the SDVersion
// enum + sd_version_is_* helpers in
// third_party/stable-diffusion.cpp/src/model.h.
//
// This is the Go-side source of truth for "is this model sd.cpp-eligible?"
// routing in handleComfyGenerate / handleOAIImageGen and for the UI's
// backend auto-selection. It is intentionally separate from
// dpp_backbones.go: many models are eligible for both, some for only one.

import "strings"

// sdcppFamilyForModel returns the sd.cpp family tag for an arbitrary model
// identifier using the same keyword heuristic as dppFamilyForModel.
// Empty string means "sd.cpp cannot run this model."
//
// Families returned here mirror the broad SDVersion buckets — variants
// like SDXL_INPAINT, FLUX_FILL, WAN2_2_I2V all collapse to the parent
// family tag, since the runtime distinguishes them by inspecting the
// loaded checkpoint, not by a separate ID.
func sdcppFamilyForModel(model string) string {
	if model == "" {
		return ""
	}
	m := strings.ToLower(model)
	switch {
	// FLUX2 must come before FLUX (substring overlap).
	case strings.Contains(m, "flux2"), strings.Contains(m, "flux-2"),
		strings.Contains(m, "flux.2"):
		return "flux2"
	case strings.Contains(m, "flux"):
		return "flux"
	case strings.Contains(m, "chroma-radiance"), strings.Contains(m, "chroma_radiance"):
		return "chroma_radiance"
	case strings.Contains(m, "flex-2"), strings.Contains(m, "flex_2"),
		strings.Contains(m, "flex.2"):
		return "flex_2"
	case strings.Contains(m, "stable-diffusion-3"),
		strings.HasPrefix(m, "sd3"), strings.Contains(m, "sd-3"):
		return "sd3"
	case strings.Contains(m, "stable-video-diffusion"), strings.HasPrefix(m, "svd"):
		return "svd"
	case strings.Contains(m, "wan2"), strings.HasPrefix(m, "wan/"),
		strings.HasPrefix(m, "wan-ai/"), strings.Contains(m, "wan-2"):
		return "wan"
	case strings.Contains(m, "qwen-image"), strings.Contains(m, "qwen_image"):
		return "qwen_image"
	case strings.Contains(m, "anima"):
		return "anima"
	case strings.Contains(m, "ltx"):
		return "ltxav"
	case strings.Contains(m, "hidream"):
		return "hidream"
	case strings.Contains(m, "z-image"), strings.Contains(m, "z_image"),
		strings.HasPrefix(m, "zimage/"):
		return "z_image"
	case strings.Contains(m, "ovis-image"), strings.Contains(m, "ovis_image"):
		return "ovis_image"
	case strings.Contains(m, "ernie-image"), strings.Contains(m, "ernie_image"):
		return "ernie_image"
	case strings.Contains(m, "longcat"):
		return "longcat"
	case strings.Contains(m, "sdxs"):
		// distilled SDXL variants — sd.cpp groups them with SD1/SD2.
		return "sdxs"
	case (strings.Contains(m, "stable-diffusion") || strings.Contains(m, "sdxl")) &&
		strings.Contains(m, "xl"):
		return "sdxl"
	case strings.Contains(m, "stable-diffusion-2"), strings.HasPrefix(m, "sd2"),
		strings.Contains(m, "sd-2"):
		return "sd2"
	case strings.Contains(m, "stable-diffusion"), strings.HasPrefix(m, "sd1"),
		strings.HasPrefix(m, "sd-1"), strings.HasPrefix(m, "sd"):
		return "sd1"
	}
	return ""
}

// isSdcppEligible — convenience wrapper.
func isSdcppEligible(model string) bool {
	return sdcppFamilyForModel(model) != ""
}

// sdcppUnsupportedFamilies enumerates families that DPP/diffusers can
// handle but stable-diffusion.cpp explicitly cannot. Used by the UI to
// gray out the sd.cpp option when the user picks one of these.
var sdcppUnsupportedFamilies = map[string]bool{
	"pixart":        true,
	"animatediff":   true,
	"cogvideox":     true,
	"hunyuan_video": true,
	"mochi":         true,
}

// isSdcppUnsupportedFamily reports whether the given DPP family is known
// to be unsupported by sd.cpp. Empty string returns false.
func isSdcppUnsupportedFamily(family string) bool {
	return sdcppUnsupportedFamilies[family]
}
