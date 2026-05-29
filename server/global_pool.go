package main

import "log"

// global_pool.go — the open "Global Public Pool" that makes the newcomer
// journey actually work.
//
// Before this, a layman who just wanted to "add my GPU to a public pool" hit a
// wall: there was no public pool to join, and even after install + login +
// connect their rig sat "paired but unattached", serving nothing, with no
// error. This bootstraps a server-owned public pool and auto-attaches any
// freshly-connected rig that isn't already in a pool — so contributing a GPU
// is: install -> login -> connect, done.

const gpunetSystemUser = "GPUnet System"
const globalPoolSlug = "global"

// ensureGlobalPool creates the system-owned public pool if it doesn't exist.
// Idempotent; call once after migrations. Returns the pool id (0 on failure).
func (s *server) ensureGlobalPool() int64 {
	var sysID int64
	if err := s.dbQueryRow(
		`SELECT id FROM users WHERE display_name = ? ORDER BY id LIMIT 1`,
		gpunetSystemUser,
	).Scan(&sysID); err != nil {
		res, e := s.dbExec(
			`INSERT INTO users (display_name, created_at) VALUES (?, ?)`,
			gpunetSystemUser, nowUnix())
		if e != nil {
			log.Printf("ensureGlobalPool: system user: %v", e)
			return 0
		}
		sysID, _ = res.LastInsertId()
	}

	var pid int64
	if s.dbQueryRow(
		`SELECT id FROM pools WHERE slug = ? AND visibility = 'public' LIMIT 1`,
		globalPoolSlug,
	).Scan(&pid) == nil && pid != 0 {
		return pid
	}

	// Other columns (parallelism/pp_stages/tp_size/tp_mode) have NOT NULL
	// defaults from the migrations, so this minimal insert is valid.
	res, e := s.dbExec(
		`INSERT INTO pools (owner_id, name, visibility, created_at, slug)
		 VALUES (?, ?, 'public', ?, ?)`,
		sysID, "Global Public Pool", nowUnix(), globalPoolSlug)
	if e != nil {
		log.Printf("ensureGlobalPool: insert pool: %v", e)
		return 0
	}
	pid, _ = res.LastInsertId()
	_, _ = s.dbExec(
		`INSERT OR IGNORE INTO pool_members (pool_id, user_id, role, joined_at)
		 VALUES (?, ?, 'owner', ?)`, pid, sysID, nowUnix())
	log.Printf("bootstrapped Global Public Pool id=%d (slug=%q)", pid, globalPoolSlug)
	return pid
}

// globalPoolID returns the public global pool's id (0 if absent).
func (s *server) globalPoolID() int64 {
	var pid int64
	_ = s.dbQueryRow(
		`SELECT id FROM pools WHERE slug = ? AND visibility = 'public' ORDER BY id LIMIT 1`,
		globalPoolSlug,
	).Scan(&pid)
	return pid
}

// attachRigToGlobalIfOrphan attaches a just-connected rig to the global public
// pool (auto-joining the user as a member — public pools are open) when the rig
// isn't in any pool yet. This is the universal limbo-killer: it runs on every
// connect path (pair-token or agent-key resume) and is a no-op once the rig
// belongs to any pool, so an operator's later manual pool choices stick.
func (s *server) attachRigToGlobalIfOrphan(uid int64, agentID string) {
	var rigID int64
	if s.dbQueryRow(
		`SELECT id FROM rigs WHERE user_id = ? AND agent_id = ?`, uid, agentID,
	).Scan(&rigID) != nil {
		return
	}
	var n int
	_ = s.dbQueryRow(`SELECT COUNT(*) FROM pool_rigs WHERE rig_id = ?`, rigID).Scan(&n)
	if n > 0 {
		return // already contributing to some pool — leave it alone
	}
	gp := s.globalPoolID()
	if gp == 0 {
		return
	}
	_, _ = s.dbExec(
		`INSERT OR IGNORE INTO pool_members (pool_id, user_id, role, joined_at)
		 VALUES (?, ?, 'member', ?)`, gp, uid, nowUnix())
	if _, err := s.dbExec(
		`INSERT OR IGNORE INTO pool_rigs (pool_id, rig_id, added_at) VALUES (?, ?, ?)`,
		gp, rigID, nowUnix(),
	); err == nil {
		log.Printf("auto-attached rig %d (user %d) to Global Public Pool %d", rigID, uid, gp)
	}
}
