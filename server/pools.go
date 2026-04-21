package main

// Pools — named bags of rigs with membership.
//
// Visibility:
//   private  — owner only; not listed to anyone else
//   invite   — owner + invitees via pool_invites tokens
//   public   — listed to everyone, anyone logged-in can join
//
// The owner is always a member with role='owner'.  Membership grants the
// right to attach your rigs and (later) route inference through the pool.

import (
	"encoding/json"
	"errors"
	"net/http"
	"strconv"
	"strings"
	"time"
)

const poolInviteTTL = 24 * time.Hour

// ─── Helpers ───────────────────────────────────────────────────────────────

// userIsMember returns (role, true) if the user is a member of pool_id.
func (s *server) userIsMember(poolID, userID int64) (string, bool) {
	var role string
	err := s.db.QueryRow(
		`SELECT role FROM pool_members WHERE pool_id = ? AND user_id = ?`,
		poolID, userID,
	).Scan(&role)
	if err != nil {
		return "", false
	}
	return role, true
}

func (s *server) poolVisibility(poolID int64) (string, int64, bool) {
	var vis string
	var owner int64
	err := s.db.QueryRow(
		`SELECT visibility, owner_id FROM pools WHERE id = ?`,
		poolID,
	).Scan(&vis, &owner)
	if err != nil {
		return "", 0, false
	}
	return vis, owner, true
}

// ─── POST /api/pools — create a pool ───────────────────────────────────────

type createPoolReq struct {
	Name        string `json:"name"`
	Visibility  string `json:"visibility"`
	Parallelism string `json:"parallelism,omitempty"` // pp|tp|pp+tp
	PPStages    int    `json:"pp_stages,omitempty"`
	TPSize      int    `json:"tp_size,omitempty"`
}

func (s *server) handleCreatePool(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	var body createPoolReq
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeErr(w, 400, "bad json")
		return
	}
	body.Name = strings.TrimSpace(body.Name)
	if body.Name == "" {
		writeErr(w, 400, "name required")
		return
	}
	switch body.Visibility {
	case "private", "invite", "public":
	case "":
		body.Visibility = "private"
	default:
		writeErr(w, 400, "visibility must be private|invite|public")
		return
	}
	switch body.Parallelism {
	case "pp", "tp", "pp+tp":
	case "":
		body.Parallelism = "pp"
	default:
		writeErr(w, 400, "parallelism must be pp|tp|pp+tp")
		return
	}
	if body.PPStages < 1 {
		body.PPStages = 1
	}
	if body.TPSize < 1 {
		body.TPSize = 1
	}

	slug, err := s.pickPoolSlug(userHandle(u), body.Name, 0)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}

	tx, err := s.db.Begin()
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	defer tx.Rollback()

	res, err := tx.Exec(
		`INSERT INTO pools (owner_id, name, visibility, created_at,
		                    parallelism, pp_stages, tp_size, slug)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
		u.ID, body.Name, body.Visibility, nowUnix(),
		body.Parallelism, body.PPStages, body.TPSize, slug,
	)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	pid, _ := res.LastInsertId()
	if _, err := tx.Exec(
		`INSERT INTO pool_members (pool_id, user_id, role, joined_at) VALUES (?, ?, 'owner', ?)`,
		pid, u.ID, nowUnix(),
	); err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	if err := tx.Commit(); err != nil {
		writeErr(w, 500, err.Error())
		return
	}

	writeJSON(w, 200, map[string]any{
		"id":          pid,
		"name":        body.Name,
		"visibility":  body.Visibility,
		"slug":        slug,
		"base_url":    s.poolBaseURL(slug),
	})
}

// poolBaseURL returns the OpenAI base URL clients should configure
// for the pool.  Uses HTTPS on the apex when publicURL is HTTPS,
// otherwise falls back to the apex's scheme (useful for local dev
// where apex == "localhost:8080" and scheme is http).
func (s *server) poolBaseURL(slug string) string {
	if slug == "" {
		return ""
	}
	scheme := "https"
	if strings.HasPrefix(s.cfg.publicURL, "http://") {
		scheme = "http"
	}
	// Dev shortcut: when apex is a bare hostname like localhost[:port]
	// subdomains won't resolve.  Fall back to the public URL itself
	// plus a ?pool= query — resolvePoolFromHost honours that.
	apex := s.cfg.apexHost
	if apex == "" || strings.HasPrefix(apex, "localhost") ||
		strings.HasPrefix(apex, "127.") {
		return strings.TrimRight(s.cfg.publicURL, "/") + "/v1?pool=" + slug
	}
	return scheme + "://" + slug + "." + apex + "/v1"
}

// ─── GET /api/pools — list mine + public ───────────────────────────────────

type poolInfo struct {
	ID         int64  `json:"id"`
	OwnerID    int64  `json:"owner_id"`
	OwnerName  string `json:"owner_name"`
	Name       string `json:"name"`
	Visibility string `json:"visibility"`
	Slug       string `json:"slug"`
	BaseURL    string `json:"base_url"`
	Role       string `json:"role,omitempty"` // owner|member|"" if not a member
	NMembers   int    `json:"n_members"`
	NRigs      int    `json:"n_rigs"`
	NOnline    int    `json:"n_online"`
	CreatedAt  int64  `json:"created_at"`
}

func (s *server) handleListPools(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	// Pools visible to u: ones they are a member of, PLUS public ones.
	rows, err := s.db.Query(`
		SELECT p.id, p.owner_id, COALESCE(o.display_name, ''), p.name, p.visibility,
		       COALESCE(p.slug, ''), p.created_at,
		       COALESCE(m.role, '') AS my_role
		FROM pools p
		JOIN users o ON o.id = p.owner_id
		LEFT JOIN pool_members m ON m.pool_id = p.id AND m.user_id = ?
		WHERE m.user_id = ? OR p.visibility = 'public'
		ORDER BY p.created_at DESC
	`, u.ID, u.ID)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	defer rows.Close()

	var pools []poolInfo
	for rows.Next() {
		var p poolInfo
		if err := rows.Scan(&p.ID, &p.OwnerID, &p.OwnerName, &p.Name, &p.Visibility, &p.Slug, &p.CreatedAt, &p.Role); err != nil {
			continue
		}
		p.BaseURL = s.poolBaseURL(p.Slug)
		// Member / rig counts.
		_ = s.db.QueryRow(`SELECT COUNT(*) FROM pool_members WHERE pool_id = ?`, p.ID).Scan(&p.NMembers)
		_ = s.db.QueryRow(`SELECT COUNT(*) FROM pool_rigs    WHERE pool_id = ?`, p.ID).Scan(&p.NRigs)
		p.NOnline = s.countOnlineRigsInPool(p.ID)
		pools = append(pools, p)
	}
	writeJSON(w, 200, map[string]any{"pools": pools})
}

// ─── GET /api/pools/{id} — detail ──────────────────────────────────────────

func (s *server) handlePoolDetail(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	pid, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil {
		writeErr(w, 400, "bad id")
		return
	}
	vis, _, ok := s.poolVisibility(pid)
	if !ok {
		writeErr(w, 404, "pool not found")
		return
	}
	role, isMember := s.userIsMember(pid, u.ID)
	if !isMember && vis != "public" {
		writeErr(w, 403, "not a member")
		return
	}

	var p poolInfo
	p.ID = pid
	p.Role = role
	err = s.db.QueryRow(`
		SELECT p.owner_id, COALESCE(o.display_name,''), p.name, p.visibility,
		       COALESCE(p.slug,''), p.created_at
		FROM pools p JOIN users o ON o.id = p.owner_id WHERE p.id = ?`, pid,
	).Scan(&p.OwnerID, &p.OwnerName, &p.Name, &p.Visibility, &p.Slug, &p.CreatedAt)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	p.BaseURL = s.poolBaseURL(p.Slug)
	_ = s.db.QueryRow(`SELECT COUNT(*) FROM pool_members WHERE pool_id = ?`, pid).Scan(&p.NMembers)
	_ = s.db.QueryRow(`SELECT COUNT(*) FROM pool_rigs    WHERE pool_id = ?`, pid).Scan(&p.NRigs)
	p.NOnline = s.countOnlineRigsInPool(pid)

	// Member list and rig list.
	type memberOut struct {
		UserID      int64  `json:"user_id"`
		DisplayName string `json:"display_name"`
		Role        string `json:"role"`
	}
	var members []memberOut
	mrows, _ := s.db.Query(`
		SELECT m.user_id, COALESCE(u.display_name,''), m.role
		FROM pool_members m JOIN users u ON u.id = m.user_id
		WHERE m.pool_id = ? ORDER BY m.joined_at ASC`, pid)
	for mrows.Next() {
		var m memberOut
		if err := mrows.Scan(&m.UserID, &m.DisplayName, &m.Role); err == nil {
			members = append(members, m)
		}
	}
	mrows.Close()

	type rigOut struct {
		RigID    int64  `json:"rig_id"`
		OwnerID  int64  `json:"owner_id"`
		AgentID  string `json:"agent_id"`
		Hostname string `json:"hostname"`
		Online   bool   `json:"online"`
	}
	var rigs []rigOut
	rrows, _ := s.db.Query(`
		SELECT r.id, r.user_id, r.agent_id, r.hostname
		FROM pool_rigs pr JOIN rigs r ON r.id = pr.rig_id
		WHERE pr.pool_id = ?`, pid)
	for rrows.Next() {
		var ro rigOut
		if err := rrows.Scan(&ro.RigID, &ro.OwnerID, &ro.AgentID, &ro.Hostname); err == nil {
			ro.Online = s.hub.agentOnline(ro.OwnerID, ro.AgentID)
			rigs = append(rigs, ro)
		}
	}
	rrows.Close()

	writeJSON(w, 200, map[string]any{
		"pool":    p,
		"members": members,
		"rigs":    rigs,
	})
}

// ─── POST /api/pools/{id}/invite — mint an invite token ────────────────────

func (s *server) handlePoolInvite(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	pid, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil {
		writeErr(w, 400, "bad id")
		return
	}
	role, isMember := s.userIsMember(pid, u.ID)
	if !isMember || role != "owner" {
		writeErr(w, 403, "only the owner can invite")
		return
	}

	token := newRandomToken(20)
	expires := time.Now().Add(poolInviteTTL)
	_, err = s.db.Exec(
		`INSERT INTO pool_invites (token, pool_id, created_by, created_at, expires_at)
		 VALUES (?, ?, ?, ?, ?)`,
		token, pid, u.ID, nowUnix(), expires.Unix(),
	)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	inviteURL := strings.TrimRight(s.cfg.publicURL, "/") + "/?invite=" + token
	writeJSON(w, 200, map[string]any{
		"token":      token,
		"invite_url": inviteURL,
		"expires_at": expires.Unix(),
	})
}

// ─── POST /api/pools/join — accept invite OR join public pool ─────────────

type poolJoinReq struct {
	Invite string `json:"invite,omitempty"`
	PoolID int64  `json:"pool_id,omitempty"`
}

func (s *server) handlePoolJoin(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	var body poolJoinReq
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeErr(w, 400, "bad json")
		return
	}

	var pid int64
	if body.Invite != "" {
		ipid, err := s.consumePoolInvite(body.Invite)
		if err != nil {
			writeErr(w, 400, err.Error())
			return
		}
		pid = ipid
	} else if body.PoolID > 0 {
		vis, _, ok := s.poolVisibility(body.PoolID)
		if !ok {
			writeErr(w, 404, "pool not found")
			return
		}
		if vis != "public" {
			writeErr(w, 403, "pool is not public")
			return
		}
		pid = body.PoolID
	} else {
		writeErr(w, 400, "invite or pool_id required")
		return
	}

	_, err := s.db.Exec(
		`INSERT OR IGNORE INTO pool_members (pool_id, user_id, role, joined_at)
		 VALUES (?, ?, 'member', ?)`,
		pid, u.ID, nowUnix(),
	)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	writeJSON(w, 200, map[string]any{"pool_id": pid, "joined": true})
}

// consumePoolInvite marks the invite used_at (one-shot) and returns the pool id.
func (s *server) consumePoolInvite(token string) (int64, error) {
	var (
		pid     int64
		exp     int64
		used    *int64
	)
	err := s.db.QueryRow(
		`SELECT pool_id, expires_at, used_at FROM pool_invites WHERE token = ?`, token,
	).Scan(&pid, &exp, &used)
	if err != nil {
		return 0, errors.New("invalid invite")
	}
	if exp < nowUnix() {
		return 0, errors.New("invite expired")
	}
	if used != nil {
		return 0, errors.New("invite already used")
	}
	_, err = s.db.Exec(`UPDATE pool_invites SET used_at = ? WHERE token = ? AND used_at IS NULL`,
		nowUnix(), token)
	if err != nil {
		return 0, err
	}
	return pid, nil
}

// ─── POST /api/pools/{id}/rigs — attach one of my rigs ────────────────────

type attachRigReq struct {
	AgentID string `json:"agent_id"`
}

func (s *server) handlePoolAttachRig(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	pid, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil {
		writeErr(w, 400, "bad id")
		return
	}
	if _, isMember := s.userIsMember(pid, u.ID); !isMember {
		writeErr(w, 403, "not a member of this pool")
		return
	}

	var body attachRigReq
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil || body.AgentID == "" {
		writeErr(w, 400, "agent_id required")
		return
	}

	// Rig must belong to the requesting user.
	var rigID int64
	err = s.db.QueryRow(
		`SELECT id FROM rigs WHERE user_id = ? AND agent_id = ?`,
		u.ID, body.AgentID,
	).Scan(&rigID)
	if err != nil {
		writeErr(w, 404, "rig not found or not yours")
		return
	}
	_, err = s.db.Exec(
		`INSERT OR IGNORE INTO pool_rigs (pool_id, rig_id, added_at) VALUES (?, ?, ?)`,
		pid, rigID, nowUnix(),
	)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	writeJSON(w, 200, map[string]any{"pool_id": pid, "rig_id": rigID, "attached": true})
}

// ─── DELETE /api/pools/{id}/rigs/{rigID} — detach a rig ────────────────────

func (s *server) handlePoolDetachRig(w http.ResponseWriter, r *http.Request) {
	u, ok := s.userFromRequest(r)
	if !ok {
		writeErr(w, 401, "not logged in")
		return
	}
	pid, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil {
		writeErr(w, 400, "bad id")
		return
	}
	rid, err := strconv.ParseInt(r.PathValue("rigID"), 10, 64)
	if err != nil {
		writeErr(w, 400, "bad rig id")
		return
	}
	// Only the rig owner or the pool owner can detach.
	var rigOwner int64
	if err := s.db.QueryRow(`SELECT user_id FROM rigs WHERE id = ?`, rid).Scan(&rigOwner); err != nil {
		writeErr(w, 404, "rig not found")
		return
	}
	_, poolOwner, _ := s.poolVisibility(pid)
	if rigOwner != u.ID && poolOwner != u.ID {
		writeErr(w, 403, "not allowed to detach this rig")
		return
	}
	_, err = s.db.Exec(`DELETE FROM pool_rigs WHERE pool_id = ? AND rig_id = ?`, pid, rid)
	if err != nil {
		writeErr(w, 500, err.Error())
		return
	}
	writeJSON(w, 200, map[string]any{"detached": true})
}
