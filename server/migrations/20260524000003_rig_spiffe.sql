-- P6: SPIFFE workload identity column on rigs.
--
-- spiffe_id — the SPIFFE URI from the rig's SVID, e.g.
--             "spiffe://prod.example.com/rig/fleet-east-42".
--             Populated when the rig presents either a JWT-SVID in the
--             /ws/agent hello or an X.509-SVID via envoy XFCC.  Legacy
--             rigs that authenticate with the paired agent_key leave
--             this NULL.
--
-- Partial unique index lets one SVID map to exactly one rig row.  We
-- keep agent_id as the primary join key so the SVID can rotate without
-- a row migration — only the spiffe_id column gets rewritten.

ALTER TABLE rigs ADD COLUMN spiffe_id TEXT;

CREATE UNIQUE INDEX IF NOT EXISTS idx_rigs_spiffe
  ON rigs(spiffe_id)
  WHERE spiffe_id IS NOT NULL;
