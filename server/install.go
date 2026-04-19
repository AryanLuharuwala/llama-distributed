package main

import (
	"embed"
	"net/http"
	"strings"
	"time"
)

//go:generate sh -c "cp ../scripts/install/install.sh  assets/install.sh && cp ../scripts/install/install.ps1 assets/install.ps1"

//go:embed assets/install.sh assets/install.ps1
var installFS embed.FS

// handleInstallSh serves the bash bootstrapper at GET /install.sh.
// It is public (no session required) — the script still needs a valid
// pair token at runtime, which is minted by POST /api/install_command.
func (s *server) handleInstallSh(w http.ResponseWriter, _ *http.Request) {
	b, err := installFS.ReadFile("assets/install.sh")
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	w.Header().Set("Content-Type", "text/x-shellscript; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	_, _ = w.Write(b)
}

// handleInstallPs1 serves the PowerShell bootstrapper at GET /install.ps1.
func (s *server) handleInstallPs1(w http.ResponseWriter, _ *http.Request) {
	b, err := installFS.ReadFile("assets/install.ps1")
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	_, _ = w.Write(b)
}

// handleInstallCommand mints a pair token and returns ready-to-paste
// one-liners for each supported OS.  The dashboard renders these so a
// first-time user can copy the exact curl/iwr command for their machine.
//
// POST /api/install_command  (session required)
//
// Response:
//   {
//     "token":      "...",
//     "deep_link":  "distpool://pair?token=...&server=wss://...",
//     "expires_at": 1713500000,
//     "bash":       "curl -fsSL https://.../install.sh | sh -s -- --pair 'distpool://...'",
//     "pwsh":       "iwr -useb https://.../install.ps1 | iex; ..."
//   }
func (s *server) handleInstallCommand(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	token := newRandomToken(20)
	expires := time.Now().Add(5 * time.Minute)
	if _, err := s.db.Exec(
		`INSERT INTO pair_tokens (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)`,
		token, u.ID, nowUnix(), expires.Unix(),
	); err != nil {
		writeErr(w, 500, err.Error())
		return
	}

	base := strings.TrimRight(s.cfg.publicURL, "/")
	wsURL := httpToWS(base) + "/ws/agent"
	deepLink := "distpool://pair?token=" + token + "&server=" + wsURL

	bash := "curl -fsSL " + base + "/install.sh | sh -s -- --pair '" + deepLink + "'"
	pwsh := "$p='" + deepLink + "'; " +
		"iwr -useb " + base + "/install.ps1 -OutFile $env:TEMP\\install.ps1; " +
		"powershell -ExecutionPolicy Bypass -File $env:TEMP\\install.ps1 -Pair $p"

	writeJSON(w, 200, map[string]any{
		"token":      token,
		"deep_link":  deepLink,
		"expires_at": expires.Unix(),
		"bash":       bash,
		"pwsh":       pwsh,
	})
}
