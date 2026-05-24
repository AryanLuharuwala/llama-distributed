-- P5: OIDC federation columns on users.
--
-- oidc_issuer  — the OP's issuer URL (https://dex.example.com).  Stored
--                so we can support multiple OIDC tenants on the same
--                user table without collisions in the subject space.
-- oidc_subject — the `sub` claim from the ID token.  Together with
--                oidc_issuer, this is the OIDC primary key for the user.
--
-- Composite uniqueness on (oidc_issuer, oidc_subject) lets us upsert
-- by (issuer, sub) during /auth/oidc/callback.  NULL is allowed for
-- legacy github_id / google_id users; SQLite's UNIQUE treats every
-- NULL as distinct so multiple legacy users won't collide.

ALTER TABLE users ADD COLUMN oidc_issuer  TEXT;
ALTER TABLE users ADD COLUMN oidc_subject TEXT;

CREATE UNIQUE INDEX IF NOT EXISTS idx_users_oidc
  ON users(oidc_issuer, oidc_subject)
  WHERE oidc_issuer IS NOT NULL AND oidc_subject IS NOT NULL;
