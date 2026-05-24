// atlas.hcl — operator config for `atlas migrate diff` against a
// running dist-server database.
//
// This file is for the operator's local atlas CLI; the server itself
// doesn't read it (boot-time application happens via
// server/schema_migrations.go).  Naming follows atlas's standard so
// that `atlas migrate diff <name>` produces files that drop straight
// into server/migrations/.
//
// Typical workflow:
//
//   # Point atlas at a snapshot of the production DB (read-only):
//   atlas migrate diff add_index_on_x \
//     --dir "file://server/migrations" \
//     --to  "postgres://reader:pw@host/dbname?sslmode=disable" \
//     --dev-url "docker://postgres/16/dev"
//
//   # Review the generated SQL, commit, deploy.  dist-server applies
//   # it on its next boot.

env "prod" {
  src = "file://server/migrations"
  url = getenv("DATABASE_URL")
  dev = "docker://postgres/16/dev?search_path=public"

  migration {
    dir    = "file://server/migrations"
    format = atlas
  }

  // Don't let a typo'd migration ship — atlas migrate lint catches
  // common issues (destructive changes, missing indexes on FKs, etc.)
  // before the file gets merged.
  lint {
    destructive {
      error = true
    }
  }
}

env "sqlite" {
  src = "file://server/migrations"
  url = getenv("DATABASE_URL")
  dev = "sqlite://dev?mode=memory"

  migration {
    dir    = "file://server/migrations"
    format = atlas
  }
}
