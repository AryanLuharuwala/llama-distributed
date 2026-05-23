package main

// Display-name validation/sanitization.
//
// User-controlled, surfaced verbatim in the pools list, the dashboard
// header, the SSE welcome frame, agent welcome frames, and in pool
// member listings.  Three hostile shapes we have to reject:
//   1. Control characters (BiDi overrides, zero-width joiners, NULs,
//      newlines) that let one user impersonate another or break
//      single-line UI rendering.
//   2. Overlong inputs that bloat DB rows / log lines.
//   3. Non-normalized Unicode where the same glyph has multiple byte
//      encodings — without NFC, "café" and "café" can both exist.
//
// We do NOT try to block visual confusables (Cyrillic 'а' vs Latin 'a')
// here; the cost of a UAX#39 skeleton table is high and the threat
// model is closer to phishing-in-username than auth bypass.  Document
// this gap so a future caller doesn't assume the field is anti-spoof.

import (
	"errors"
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

const (
	displayNameMaxRunes = 64
	displayNameMinRunes = 1
)

var (
	errDisplayNameEmpty    = errors.New("display_name required")
	errDisplayNameTooLong  = errors.New("display_name exceeds 64 characters")
	errDisplayNameControl  = errors.New("display_name contains control or invisible characters")
	errDisplayNameInvalid  = errors.New("display_name is not valid UTF-8")
)

// sanitizeDisplayName trims surrounding whitespace, NFC-normalizes,
// validates rune count, and rejects any control / format / surrogate /
// private-use / unassigned codepoint.  Returns the canonical form
// callers should persist, or an error suitable for a 400 response.
func sanitizeDisplayName(raw string) (string, error) {
	if !utf8.ValidString(raw) {
		return "", errDisplayNameInvalid
	}
	// NFC first — composition can shrink rune count below the cap.
	s := norm.NFC.String(raw)
	s = strings.TrimSpace(s)

	if s == "" {
		return "", errDisplayNameEmpty
	}

	runeCount := 0
	for _, r := range s {
		runeCount++
		if runeCount > displayNameMaxRunes {
			return "", errDisplayNameTooLong
		}
		if !isPrintableDisplayRune(r) {
			return "", errDisplayNameControl
		}
	}
	if runeCount < displayNameMinRunes {
		return "", errDisplayNameEmpty
	}
	return s, nil
}

// isPrintableDisplayRune returns true if r is safe to render in a
// single-line UI label.  Rejects:
//   - C0/C1 control codes (newline, NUL, BEL, ESC…)
//   - Unicode "Cf" format chars (BiDi overrides, zero-width joiners,
//     LRE/RLE/PDF, RLM/LRM)
//   - Surrogates (which would have failed UTF-8 anyway, but defense-in-depth)
//   - Private-use area
//   - Unassigned codepoints
func isPrintableDisplayRune(r rune) bool {
	if r == utf8.RuneError {
		return false
	}
	// Block C0 (incl. \t, \n, \r) and DEL.  TrimSpace removed leading/
	// trailing whitespace but interior tabs/newlines are still
	// hostile for single-line rendering.
	if r < 0x20 || r == 0x7f {
		return false
	}
	// Block C1 (0x80–0x9f).
	if r >= 0x80 && r <= 0x9f {
		return false
	}
	// Format chars: ZWJ (200D), ZWNJ (200C), RLM/LRM (200E/200F),
	// LRE/RLE/PDF/LRO/RLO (202A–202E), Mongolian variation selectors,
	// bidi isolates (2066–2069), BOM (FEFF), etc.  unicode.Cf covers
	// the lot.
	if unicode.Is(unicode.Cf, r) {
		return false
	}
	// Surrogates, private-use, unassigned.
	if unicode.Is(unicode.Cs, r) || unicode.Is(unicode.Co, r) {
		return false
	}
	return true
}
