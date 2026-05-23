package main

// Paragraph + sentence + token-budget chunker.
//
// Why custom instead of pulling in a third-party tokenizer:
//   - The agent fleet already standardises on a tokenizer for billing
//     (tokenization.go).  A second one here would invite drift.
//   - For RAG retrieval the chunk boundary matters more than exact token
//     count — we want semantically coherent spans.  An approximate
//     ~4-char-per-token heuristic is enough at retrieval time.
//
// Algorithm:
//   1. Split on blank lines → paragraphs.
//   2. For paragraphs that exceed the budget, fall back to sentence
//      splitting (./?/! followed by whitespace).
//   3. For sentences that *still* exceed the budget (very rare — code
//      blobs, dumped JSON), hard-wrap at the budget boundary.
//   4. Pack consecutive units into chunks up to budget, with a small
//      overlap to preserve context across boundaries.

import (
	"strings"
	"unicode"
)

// chunkerConfig controls chunking behaviour.  Sane defaults live on
// defaultChunkerConfig; callers tweak only what they care about.
type chunkerConfig struct {
	TargetTokens int // soft target per chunk (~tokens, not characters)
	MaxTokens    int // hard cap; a chunk above this is split mid-sentence
	OverlapToks  int // tail of prev chunk prepended to next, in tokens
}

func defaultChunkerConfig() chunkerConfig {
	return chunkerConfig{TargetTokens: 256, MaxTokens: 512, OverlapToks: 32}
}

// approxTokens is a cheap stand-in for a real tokenizer.  ~4 chars per
// token is the well-known heuristic from OpenAI's tokenizer docs; close
// enough for chunk sizing.
func approxTokens(s string) int {
	if s == "" {
		return 0
	}
	n := len(s) / 4
	if n == 0 {
		return 1
	}
	return n
}

// chunkText splits text into retrieval-ready chunks.  Returns the chunks
// as plain strings; the caller (rag_ingest.go) wraps them in ragChunk
// once embeddings are computed.
func chunkText(text string, cfg chunkerConfig) []string {
	if cfg.TargetTokens <= 0 {
		cfg.TargetTokens = 256
	}
	if cfg.MaxTokens <= 0 || cfg.MaxTokens < cfg.TargetTokens {
		cfg.MaxTokens = cfg.TargetTokens * 2
	}
	if cfg.OverlapToks < 0 {
		cfg.OverlapToks = 0
	}
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}

	units := splitUnits(text, cfg.MaxTokens)
	chunks := packUnits(units, cfg)
	return chunks
}

// splitUnits walks the text once, emitting semantic units (paragraph >
// sentence > forced span) each at most maxToks tokens.
func splitUnits(text string, maxToks int) []string {
	var out []string
	for _, para := range splitParagraphs(text) {
		if approxTokens(para) <= maxToks {
			out = append(out, para)
			continue
		}
		for _, sent := range splitSentences(para) {
			if approxTokens(sent) <= maxToks {
				out = append(out, sent)
				continue
			}
			out = append(out, hardWrap(sent, maxToks)...)
		}
	}
	return out
}

// splitParagraphs splits on blank-line runs.  Single \n inside a
// paragraph (e.g. wrapped prose) is preserved.
func splitParagraphs(text string) []string {
	var out []string
	var b strings.Builder
	prevBlank := true
	for _, line := range strings.Split(text, "\n") {
		trim := strings.TrimRightFunc(line, unicode.IsSpace)
		if trim == "" {
			if b.Len() > 0 {
				out = append(out, strings.TrimSpace(b.String()))
				b.Reset()
			}
			prevBlank = true
			continue
		}
		if !prevBlank {
			b.WriteByte('\n')
		}
		b.WriteString(trim)
		prevBlank = false
	}
	if b.Len() > 0 {
		out = append(out, strings.TrimSpace(b.String()))
	}
	return out
}

// splitSentences splits on sentence-ending punctuation followed by
// whitespace.  Lossy on edge cases (abbreviations, ellipses) but good
// enough — the retrieval layer is robust to chunk-boundary noise.
func splitSentences(para string) []string {
	var out []string
	var b strings.Builder
	runes := []rune(para)
	for i := 0; i < len(runes); i++ {
		r := runes[i]
		b.WriteRune(r)
		if r != '.' && r != '!' && r != '?' {
			continue
		}
		if i+1 < len(runes) && !unicode.IsSpace(runes[i+1]) {
			continue
		}
		// consume trailing whitespace
		for i+1 < len(runes) && unicode.IsSpace(runes[i+1]) {
			i++
		}
		if s := strings.TrimSpace(b.String()); s != "" {
			out = append(out, s)
		}
		b.Reset()
	}
	if s := strings.TrimSpace(b.String()); s != "" {
		out = append(out, s)
	}
	return out
}

// hardWrap splits a single oversize span into max-token slices.  Used as
// a last resort — a single sentence shouldn't exceed maxToks in practice.
func hardWrap(s string, maxToks int) []string {
	if maxToks <= 0 {
		return []string{s}
	}
	maxChars := maxToks * 4
	if maxChars <= 0 {
		return []string{s}
	}
	var out []string
	for len(s) > maxChars {
		// Cut at a word boundary if one is nearby; otherwise hard cut.
		cut := maxChars
		if i := strings.LastIndexAny(s[:cut], " \t\n"); i > maxChars/2 {
			cut = i
		}
		out = append(out, strings.TrimSpace(s[:cut]))
		s = strings.TrimSpace(s[cut:])
	}
	if s != "" {
		out = append(out, s)
	}
	return out
}

// packUnits joins units up to targetTokens, with an overlap window of
// the last overlapToks tokens of the previous chunk.  Overlap helps
// retrieval recall on queries that span a chunk boundary.
func packUnits(units []string, cfg chunkerConfig) []string {
	if len(units) == 0 {
		return nil
	}
	var out []string
	var cur strings.Builder
	curToks := 0
	flush := func() {
		if cur.Len() == 0 {
			return
		}
		out = append(out, strings.TrimSpace(cur.String()))
		cur.Reset()
		curToks = 0
	}
	for _, u := range units {
		ut := approxTokens(u)
		if curToks > 0 && curToks+ut > cfg.TargetTokens {
			// Save current chunk, then seed the next with an overlap tail.
			prev := strings.TrimSpace(cur.String())
			flush()
			if cfg.OverlapToks > 0 {
				tail := tailTokens(prev, cfg.OverlapToks)
				if tail != "" {
					cur.WriteString(tail)
					cur.WriteByte('\n')
					curToks = approxTokens(tail)
				}
			}
		}
		if cur.Len() > 0 {
			cur.WriteByte('\n')
		}
		cur.WriteString(u)
		curToks += ut
	}
	flush()
	return out
}

// tailTokens returns approximately the last n tokens of s, snapping back
// to a word boundary so we never produce a half-word overlap.
func tailTokens(s string, n int) string {
	if n <= 0 || s == "" {
		return ""
	}
	chars := n * 4
	if chars >= len(s) {
		return s
	}
	start := len(s) - chars
	if i := strings.IndexAny(s[start:], " \t\n"); i >= 0 {
		start += i + 1
	}
	if start >= len(s) {
		return ""
	}
	return s[start:]
}
