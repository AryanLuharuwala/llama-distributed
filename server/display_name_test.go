package main

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"
	"unicode/utf8"
)

func TestSanitizeDisplayName_Valid(t *testing.T) {
	cases := []struct {
		name, in, want string
	}{
		{"plain ascii", "alice", "alice"},
		{"trim space", "  bob  ", "bob"},
		{"unicode letter", "café", "café"},
		{"emoji", "alice \U0001F980", "alice \U0001F980"},
		{"60 runes", strings.Repeat("a", 60), strings.Repeat("a", 60)},
		{"64 runes (boundary)", strings.Repeat("a", 64), strings.Repeat("a", 64)},
		{"japanese", "山田太郎", "山田太郎"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := sanitizeDisplayName(tc.in)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Errorf("got %q want %q", got, tc.want)
			}
		})
	}
}

func TestSanitizeDisplayName_Rejected(t *testing.T) {
	cases := []struct {
		name, in string
	}{
		{"empty", ""},
		{"whitespace only", "   "},
		{"newline in middle", "ali\nce"},
		{"tab in middle", "ali\tce"},
		{"NUL byte", "alice\x00"},
		{"BEL", "alice\x07"},
		{"DEL", "alice\x7f"},
		{"C1 control", "ali\u009bce"},
		{"BiDi RLO override", "ali\u202ece"},          // RIGHT-TO-LEFT OVERRIDE
		{"BiDi LRO override", "\u202dalice"},          // LEFT-TO-RIGHT OVERRIDE
		{"zero-width joiner", "ali\u200dce"},
		{"BOM in middle", "ali\ufeffce"},
		{"BiDi isolate", "ali\u2066ce"},               // LRI
		{"too long (65 runes)", strings.Repeat("a", 65)},
		{"too long (300 runes)", strings.Repeat("a", 300)},
		{"invalid utf8", string([]byte{0xff, 0xfe, 0xfd})},
		{"private use area", "ali\ue000ce"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := sanitizeDisplayName(tc.in)
			if err == nil {
				t.Errorf("expected rejection for %q but got %q", tc.in, got)
			}
		})
	}
}

func TestSanitizeDisplayName_NFCNormalization(t *testing.T) {
	// "café" with combining acute (e + U+0301) should NFC-compose to
	// the precomposed é (U+00E9), so two different inputs become one
	// canonical string in the DB.
	decomposed := "café"
	precomposed := "café"

	a, err := sanitizeDisplayName(decomposed)
	if err != nil {
		t.Fatalf("decomposed rejected: %v", err)
	}
	b, err := sanitizeDisplayName(precomposed)
	if err != nil {
		t.Fatalf("precomposed rejected: %v", err)
	}
	if a != b {
		t.Errorf("NFC failed: %q != %q (bytes %x vs %x)", a, b, []byte(a), []byte(b))
	}
}

func TestSanitizeDisplayName_NFCShrinksRuneCount(t *testing.T) {
	// 64 base 'e's followed by 64 combining acutes is 128 runes
	// pre-normalize, 64 runes post-NFC (each pair composes to é).
	// Must pass the cap.
	base := strings.Repeat("é", 64)
	got, err := sanitizeDisplayName(base)
	if err != nil {
		t.Fatalf("decomposed 64-rune string was rejected: %v", err)
	}
	if utf8.RuneCountInString(got) != 64 {
		t.Errorf("expected 64 NFC runes, got %d", utf8.RuneCountInString(got))
	}
}

// FuzzSanitizeDisplayName proves that whatever the function accepts,
// the returned string is itself acceptable (idempotence) and contains
// no control / format runes — even on unicode confusables and weird
// byte sequences.
func FuzzSanitizeDisplayName(f *testing.F) {
	seeds := []string{
		"alice", "café", "山田", "", "   ", "a\u202eb",
		strings.Repeat("a", 64),
		strings.Repeat("a", 65),
		"\x00\x01\x02",
		"\ufeff",
		"Аlice", // Cyrillic capital A (confusable)
	}
	for _, s := range seeds {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, in string) {
		out, err := sanitizeDisplayName(in)
		if err != nil {
			return
		}
		// Property 1: idempotent.
		again, err2 := sanitizeDisplayName(out)
		if err2 != nil {
			t.Fatalf("output rejected on re-sanitize: in=%q out=%q err=%v", in, out, err2)
		}
		if again != out {
			t.Fatalf("non-idempotent: %q -> %q -> %q", in, out, again)
		}
		// Property 2: no control / format runes in output.
		for _, r := range out {
			if !isPrintableDisplayRune(r) {
				t.Fatalf("output contains unprintable rune U+%04X: in=%q out=%q", r, in, out)
			}
		}
		// Property 3: rune count within bounds.
		n := utf8.RuneCountInString(out)
		if n < displayNameMinRunes || n > displayNameMaxRunes {
			t.Fatalf("output rune count %d out of [%d,%d]: in=%q out=%q",
				n, displayNameMinRunes, displayNameMaxRunes, in, out)
		}
	})
}

// TestDevLoginRejectsHostileDisplayName covers the wire path: a POST
// to /auth/dev with BiDi-override or overlong payloads must 400, and
// no users row must be created.
func TestDevLoginRejectsHostileDisplayName(t *testing.T) {
	s := newTestServer(t)
	s.cfg.devMode = true

	hostile := []string{
		"\u202eevil",                    // RLO override
		strings.Repeat("x", 200),        // overlong
		"alice\x00",                     // NUL
		"ali\nce",                       // interior newline
		"",                              // empty
		string([]byte{0xff, 0xfe}),      // invalid UTF-8
	}
	for _, name := range hostile {
		t.Run("reject-"+stringTagForTest(name), func(t *testing.T) {
			body := map[string]string{"display_name": name}
			b, _ := json.Marshal(body)
			r := httptest.NewRequest("POST", "/auth/dev", strings.NewReader(string(b)))
			r.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()
			s.handleDevLogin(w, r)
			if w.Code != 400 {
				t.Errorf("hostile display_name %q: status=%d want=400 body=%s", name, w.Code, w.Body.String())
			}
		})
	}

	var rows int
	if err := s.db.QueryRow(`SELECT COUNT(*) FROM users`).Scan(&rows); err != nil {
		t.Fatalf("count users: %v", err)
	}
	if rows != 0 {
		t.Errorf("expected 0 users after %d hostile attempts, got %d", len(hostile), rows)
	}
}

// TestDevLoginAcceptsAndCanonicalizes proves that a valid display
// name with decomposed combining marks is persisted in NFC form so
// downstream SSE / welcome frames emit canonical bytes.
func TestDevLoginAcceptsAndCanonicalizes(t *testing.T) {
	s := newTestServer(t)
	s.cfg.devMode = true

	body := map[string]string{"display_name": "café"} // decomposed
	b, _ := json.Marshal(body)
	r := httptest.NewRequest("POST", "/auth/dev", strings.NewReader(string(b)))
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	s.handleDevLogin(w, r)
	if w.Code != 200 {
		t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
	}

	var stored string
	if err := s.db.QueryRow(`SELECT display_name FROM users LIMIT 1`).Scan(&stored); err != nil {
		t.Fatalf("read back: %v", err)
	}
	if stored != "café" {
		t.Errorf("DB stored %q (bytes %x), expected NFC %q",
			stored, []byte(stored), "café")
	}
}

// stringTagForTest produces a short, run-safe tag for subtest names
// (raw hostile strings can include control bytes or be long).
func stringTagForTest(s string) string {
	if s == "" {
		return "empty"
	}
	out := make([]byte, 0, 16)
	i := 0
	for _, r := range s {
		if i >= 8 {
			break
		}
		i++
		if r >= 0x20 && r < 0x7f {
			out = append(out, byte(r))
		} else {
			out = append(out, '_')
		}
	}
	return string(out)
}
