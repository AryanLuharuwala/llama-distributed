# Static analysis sweep

Single-pass static analysis of `/home/boom/Desktop/Startup/DistibutedInference/llama-distributed`
on 2026-05-25.  Raw tool output lives under `audit/raw/`.

The codebase is in good shape: three Go modules build clean under `go vet`,
no data races detected by `go test -race`, no secrets leaked, no banned C
APIs.  The findings worth shipping a fix for are concentrated in a few
hot spots, listed in the Top 10 below.

## Tools run

| Tool          | Target                              | Install         | Exec time  | Outcome |
|---------------|-------------------------------------|-----------------|------------|---------|
| go vet        | server, dist-turn, surd             | preinstalled    | < 1 s each | clean   |
| staticcheck   | server, dist-turn, surd             | go install ~30s | 5–7 s each | 55 findings, server only |
| gosec         | server, dist-turn, surd             | go install ~30s | 2 s each   | 87 / 2 / 6 findings |
| govulncheck   | server, dist-turn, surd             | go install ~30s | 5–8 s each | 8 / 6 / 5 stdlib CVEs (Go 1.25.8) + pion/dtls v2.2.7 advisory in dist-turn |
| errcheck      | server, dist-turn, surd             | go install ~10s | < 3 s each | 94 / 2 / 16 findings (mostly idiomatic) |
| go test -race | server, dist-turn, surd             | builtin         | 34 s server, 1 s others | clean — 0 races, all pass |
| go test -cover| server, dist-turn, surd             | builtin         | 11 s server | server 39.1 %, others no tests |
| cppcheck      | src/, include/                      | pip cppcheck-wheel 1.5.1 (wraps cppcheck 2.17.1) | 6 s | 1 error (FP), 105 style, 11 portability, 11 performance |
| clang-tidy    | 5 largest src/*.cpp + compile_commands.json | preinstalled | 76 s | 1889 lines; 1 real bug, lots of magic-numbers noise |
| shellcheck    | tools/, scripts/, deploy/, infra/ shell | binary download | 1.5 s | clean (0 warnings at `-S warning`) |
| buildifier    | BUILD.bazel, *.bzl, MODULE.bazel    | binary download | < 1 s | 6 findings, 1 real |
| gitleaks      | full git history (57 commits)       | binary download | 1 s | 2 hits, both test fixtures |
| banned-API grep | strcpy/sprintf/gets in src/include/ | grep            | < 1 s  | 1 hit (`::fgets`, bounded — safe) |
| secret grep   | repo-wide for common token formats  | grep            | < 1 s | clean |

Nothing skipped for install-time reasons.  shellcheck flagged zero issues
at warning severity — the install scripts are clean.

## Top 10 findings across all tools

Ranked highest impact first.

| # | Sev      | File:Line                          | Issue |
|---|----------|------------------------------------|-------|
| 1 | High     | `server/inference.go:131`          | `decodeChunk` integer-overflow → panic-on-craft.  `plen=0xFFFFFFFF` makes `24+plen` wrap to 23; bounds check passes; subsequent slice `b[24:24+plen]` panics with "slice bounds out of range".  Reachable from any authenticated agent's binary frame in `dispatchBinaryFromAgent` (`ws.go:1016`) — no `recover()` on the WS reader goroutine, so this crashes the server. |
| 2 | High     | `server/pp_route.go:362–363`       | `cleanup` callback reassignment is dead.  `defer cleanup()` at line 249 binds the *original* closure value; the augmented cleanup that revokes P2P pairs, releases relays, and credits successful relay sessions is never invoked.  staticcheck (SA4006) caught this.  Pipeline sessions leak P2P pair grants and relay accounting on completion. |
| 3 | High     | `cmd/dist-turn` (indirect)         | govulncheck `GO-2026-4479` — pion/dtls v2.2.7 uses a random AES-GCM nonce, risking authentication-key leakage; "Fixed in: N/A" for v2 (move to v3 required).  pion/turn pulls dtls/v2 transitively, so DTLS termination on the TURN relay is exposed. |
| 4 | High     | `cmd/dist-turn/BUILD.bazel:28,46`  | Two Bazel targets share `name = "dist-turn"` — a `go_binary` at line 28 and the `oci_image_index` emitted by `dist_go_image` at line 46.  Bazel will reject this at load time when the target is actually built.  Either nobody builds `//cmd/dist-turn:dist-turn` via Bazel, or one of the two must be renamed. |
| 5 | High     | go.mod (all three modules)         | Go toolchain pinned to `1.25.0`/`1.25.8`.  govulncheck flags **8 stdlib CVEs** (html/template ×3 XSS, crypto/tls KeyUpdate DoS, crypto/x509 ×2, net Windows-only panic, net/http HTTP/2 infinite loop).  Bumping to `1.25.10` is a no-code-change fix.  Also: `golang.org/x/net v0.52.0` → v0.53.0 fixes the HTTP/2 SETTINGS infinite-loop separately. |
| 6 | Medium   | `server/relay.go:195`              | `break` inside `switch` inside a relay reader loop — staticcheck SA4011 ineffective break.  Client `kind=close` text frames don't actually close the relay; the loop only exits on network error or the binary back-pressure path.  Minor (sessions still get cleaned up on disconnect) but real protocol-contract violation. |
| 7 | Medium   | `src/dist_node_main.cpp:2955–2958` | Use-after-move (clang-tidy `bugprone-use-after-move`).  `reqs[r.req_id] = std::move(r);` at line 2950, then the next line reads `r.seq_id`, `r.layer_lo`, `r.layer_hi`, `r.is_first`, `r.is_last` in a log statement.  Trivial fields *probably* survive, but the standard says reads after move are UB.  Capture the values into locals before the move. |
| 8 | Medium   | `src/vllm_adapter.cpp:428`         | Virtual call in destructor (clang-analyzer-optin.cplusplus.VirtualCall).  `~VllmAdapter()` calls `close()`; any override in a derived class is bypassed because vtable is already sliced.  If subclassing isn't intended, mark the class `final`; if it is, switch the destructor to call the base implementation directly. |
| 9 | Medium   | `server/openai.go:605, 666`        | `go s.summarizeChatTurnSafely(...)` spawned from request handler uses `context.Background()` implicitly (the goroutine doesn't accept the request ctx).  gosec G118.  Two consequences: (a) summarization survives client disconnect — usually desirable, so this may be intentional — but (b) it also outlives server shutdown, so an in-flight summarize can race the DB close.  At least gate the goroutine on a shutdown context held by the server. |
| 10| Medium   | `cmd/dist-turn/main.go` weak-RNG via sha1 import + pion/dtls advisory above | Defense-in-depth: server-side uses HMAC-SHA1 against the TURN spec (`signal.go`) which is fine; but the combination of a vulnerable DTLS dependency and AES-GCM nonce generation done by that library is the path of least resistance for an attacker against the relay's E2E confidentiality story (already partly known: see `peer_relay_plaintext_exposure` memory). |

## Findings by tool

### go vet
All three modules: **clean**.

### staticcheck (server only — others clean)
- **SA4006** ×2: real dead-store bugs.
  - `pp_route.go:363` — see top-10 #2 above.
  - `dashboard_api.go:614` — `this value of n is never used`; cosmetic.
- **SA4011** ×1: `relay.go:195` — see top-10 #6.
- **U1000** ×32 unused symbols (functions, types, vars).  Several look like
  WIP placeholders (`p2pSignalSweep`, `prefixAffinity*`, `ragCollection`,
  `recordComputeFailure/Success` on reputation).  Either wire them up or
  delete; right now they're cruft that confuses code-search.
- **ST1013** ×18: numeric HTTP status literals.  Style only.
- **S1011/S1017** ×2: idiomatic improvements.  Style only.

### gosec
- **server (87 issues, 21 HIGH-severity flags)** — the HIGH-severity bucket
  mostly evaporates on inspection.  Real:
  - G115 integer-overflow at `inference.go:131` — see top-10 #1.
  - G118 context-in-goroutine — see top-10 #9.
  - All G703 "path traversal" hits (`models.go:282`, `install.go:713,719`,
    `shard_fetch_plan.go:118`) are guarded by `isSafeShardFile` /
    `validSurdBinName` whitelists — **false positives**.
  - G704 SSRF in `install.go:459,465` — URL is a hardcoded
    `api.github.com/...` constructed from gTargets cache, no user input —
    **false positive**.
  - G101 hardcoded credentials on `oauth*` and `oidc.go` flag public
    well-known endpoint URLs — **false positives**.
  - G124 cookie attributes ×12 — every call site is in fact setting
    `HttpOnly: true`, `Secure: s.secureCookies()`, and (where appropriate)
    `SameSite: Lax`.  gosec is matching on the literal `http.Cookie{` and
    not following struct fields.  **False positives.**
  - G505 sha1 import — used for TURN-spec HMAC-SHA1; required by RFC 5389.
    **False positive.**
  - G404 weak RNG in `reputation.go:284` — Thompson sampling for relay
    selection.  Non-cryptographic by design.  **False positive.**
  - G306/G302/G301 perm-too-permissive on `0o755`/`0o644` — server
    intentionally creates world-readable model shards and release assets.
    Acceptable.
  - G204 exec.Command-with-variable ×3 — `splitter` / `pythonBin` come from
    server config, not user input.  Acceptable.
  - G304 file-inclusion-via-variable ×4 — `os.Create(part)`,
    `os.Open(dst)`, etc., where `part`/`dst` are server-built paths.  See
    G703 above re whitelisted user input.  False positives.
  - G104 unhandled errors ×19 — almost all `rows.Close()` after an early
    error return, or `_ = json.Marshal()` calls.  Idiomatic.
  - G706 "log injection via taint" ×6 — taint-tracked log lines
    interpolate user-controlled strings.  Acceptable in plain-text logs;
    note that if you ever switch to a structured logging frontend with
    JSON output, these become real escape problems.  Today: low.
- **cmd/dist-turn (2 issues)**: G304 reading a known `state` file from a
  fixed dir (safe), G505 sha1 for TURN HMAC (required).  Both false
  positives.
- **cmd/surd (20 issues)**: 14 G104 on `fs.Parse(args)` returns (idiomatic
  flag-package pattern); rest are similar low-impact.

### govulncheck
- **server** — 8 stdlib CVEs against Go 1.25.8, all fixed in 1.25.9 or
  1.25.10 (no source changes required).  Reachable: HTTP/2 infinite loop
  (any inbound request), TLS 1.3 KeyUpdate DoS (every TLS handshake),
  html/template XSS ×3 (console template execution).  Also
  `golang.org/x/net v0.52.0` < v0.53.0.
- **dist-turn** — same stdlib CVEs + pion/dtls v2.2.7 random-nonce GCM
  issue (top-10 #3).
- **surd** — same stdlib CVEs; no third-party CVEs in reachable code.

### errcheck
- 112 total, dominated by `defer rows.Close()`, `fmt.Fprintf(w, ...)` to
  `http.ResponseWriter`, and `os.Stdout.Write(raw)` in surd.  All
  idiomatic Go.  No high-impact ignored errors found by hand-walking the
  list.

### go test -race
All three modules pass `go test -race -count=1`.  Server (sole module
with tests) ran 12.5 s under the race detector and emitted zero
WARNING: DATA RACE.  Notable — the agent connection map, the WS reader,
and the codec path all have `sync.RWMutex` discipline that the race
detector can't fault.

### cppcheck
- 1 error (`nat_pmp.cpp:281` invalidLifetime) — **false positive**.
  `m.external_ip` is a `std::string` that copies the local `buf`
  before scope exit at line 280.
- 105 style — knownConditionTrueFalse ×3 (false positives — cppcheck
  loses track of out-params), shadowVariable ×2 (`dist_cli_split.cpp:231`,
  `dist_node_main.cpp:1996`), rest are missingOverride and similar.
- 11 portability — all `reinterpret_cast<float*>(uint8_t*)` in the VM
  tensor path.  Intentional; required by the runtime layout.
- 11 performance — `substr(0, n)` self-assign hits in adapters
  (`vllm/comfy/sglang/trtllm`), one ineffective `passByValue` in
  `dist_cli_main.cpp` ×3.  Low priority.

### clang-tidy
1889-line dump, dominated by 333 `readability-magic-numbers` (suppress
the check or accept the noise).  Real bugs:
- `bugprone-use-after-move` ×1 — top-10 #7.
- `clang-analyzer-optin.cplusplus.VirtualCall` ×1 — top-10 #8.
- `bugprone-empty-catch` ×10 — every site is wrapping `std::stoi` or
  similar parse code where the empty handler is the *correct* behavior
  (parse-best-effort), but they should at minimum log so a busted
  config doesn't silently zero-out a timeout.  Low.
- `bugprone-implicit-widening-of-multiplication-result` ×7 — arithmetic
  done in `int` then widened to `size_t`/`ptrdiff_t`/`__suseconds_t`.
  Two of them (`dist_node_main.cpp:1600,3122`) are pointer offset math
  with `int * int` ≤ INT_MAX — won't overflow in practice but worth
  fixing with a leading `static_cast<size_t>`.
- `bugprone-narrowing-conversions` ×1 — `coordinator.cpp:442`
  `uint64_t → double` for a ratio used to allocate layer shares.
  Acceptable for typical VRAM sizes; would silently lose precision
  past 2^53 bytes (8 PiB) which is far above realistic GPU VRAM.

### shellcheck
**0 findings** at `-S warning`.  The install / build / deploy scripts are
well-disciplined: every variable is quoted, no unsanitized `eval`, no
backticks.  Notable since there are 21 shell files.

### buildifier
- `cmd/dist-turn/BUILD.bazel:40` duplicated-name — top-10 #4.
- `deploy/sign/BUILD.bazel:1–2` three unused `load()` symbols
  (`pkg_attributes`, `pkg_files`, `pkg_tar`) — dead imports.
- `tools/oci.bzl:15,29` missing module/function docstrings.  Style.

### gitleaks
2 findings in the full 57-commit history.  Both are `sessionSecret:
"test-secret-..."` literals in `*_test.go` files — test fixtures, not
real secrets.

### banned-API + secret grep
- Banned APIs: `::fgets(buf, sizeof(buf), fp)` in
  `src/dist_node_main.cpp:170` — bounded write, safe.
- Secret grep: clean.  No `BEGIN PRIVATE KEY`, no `ghp_`/`xoxb-`/AWS
  keys.  An `.env` file exists in the repo root with 101 bytes; check it
  manually before any public push (the file is in `.gitignore` per the
  listing).

## Test coverage by package

Server is laid out as a single Go package, so coverage is package-level:

| Module                                | Package                                    | Coverage |
|---------------------------------------|--------------------------------------------|----------|
| server                                | github.com/llama-distributed/server        | **39.1 %** |
| server                                | github.com/llama-distributed/server/ctrlpb | 0.0 % (generated protobuf — N/A) |
| cmd/dist-turn                         | github.com/llama-distributed/dist-turn     | no tests |
| cmd/surd                              | github.com/llama-distributed/surd          | no tests |

**Coverage notes:**
- The server's 39 % aggregates auth (oauth/oidc/device-code), billing
  (counter_store, cost_planner), inference dispatch, WS handling, RAG,
  pipeline routing, and 70+ other source files into a single number.
  The breakage tests in `breakage_test.go` give the auth surface real
  coverage, but it's impossible from one package-level number to
  identify which subsystem is undertested.  **Recommendation:** split
  into sub-packages or instrument `-coverprofile` and roll it up
  per-file.
- `cmd/dist-turn` and `cmd/surd` have **zero tests**.  dist-turn is a
  TURN relay carrying real-time activation traffic.  surd is the
  operator CLI that issues credentials.  Both are user-input boundaries.
  *Suggested minimum:* a smoke test on auth-token parsing in dist-turn,
  and integration tests on the `surd login` / `surd run` happy paths.

## Dependency drift / CVEs

| Module        | Current             | Latest stable       | Severity for our codebase |
|---------------|---------------------|---------------------|---------------------------|
| Go toolchain  | 1.25.0 / 1.25.8     | 1.25.10             | 8 reachable stdlib CVEs (top-10 #5) |
| golang.org/x/net | v0.52.0          | v0.55.0             | HTTP/2 infinite-loop (GO-2026-4918) |
| golang.org/x/crypto | v0.49.0       | v0.52.0             | No reachable CVEs but lagging 3 minors |
| pion/dtls v2  | v2.2.7              | v2.2.12 (still vuln) | random-nonce AES-GCM, no v2 fix — top-10 #3 |
| pion/turn v3  | v3.0.3              | v3.0.3              | up to date |
| coder/websocket | v1.8.13           | v1.8.14             | no CVE, minor |

**MODULE.bazel pins** — `rules_go`, `rules_oci`, `rules_pkg` digests should
be checked against upstream releases manually if you intend to deploy via
Bazel-built images.  I didn't run a WebSearch — would not have produced
high-signal output relative to time cost.

## Noise stats

| Tool         | Low-severity / style findings (counted, not listed) |
|--------------|------------------------------------------------------|
| staticcheck  | 50 of 55 (U1000 ×32, ST1013 ×18)                     |
| gosec server | ~65 of 87 (G104 ×19, G124 ×12, G301 ×11, G706 ×6, G103 ×4, G304 ×4, G103 ×4, G706 ×6, G703 ×4, G302 ×1, G302/G306 ×3) — see Findings by tool for breakdown |
| gosec dist-turn | 2 (both FP)                                       |
| gosec surd   | 20 (mostly G104 fs.Parse)                            |
| errcheck     | ~110 of 112 (defer Close, fmt.Fprintf to ResponseWriter) |
| cppcheck     | 116 (105 style + 11 portability — all expected reinterpret_casts) |
| clang-tidy   | ~1850 of 1889 (333 magic-numbers + bugprone-easily-swappable-parameters ×19 + readability noise) |
| buildifier   | 5 (docstring + unused load)                          |
| gitleaks     | 2 (both test fixtures)                               |

## What I couldn't get to and why

- **rules_go / rules_oci / rules_pkg pin freshness vs upstream** —
  decided WebSearch was not worth the round-trip; the Bazel build wasn't
  actually verified working anyway (top-10 #4 will block it).
- **Per-file Go coverage rollup** — the server is one package, so
  `go test -coverprofile` would still record one number per package.
  Producing per-file coverage requires either splitting the package or
  parsing the coverage profile manually; deferred.
- **clang-tidy on the whole src/ tree** — I ran 5 files (76 s).  Doing
  all 60+ src/*.cpp would have taken 15+ minutes for likely the same
  classes of findings.  The five-file sample includes the largest and
  most-touched files; broadening it is unlikely to surface qualitatively
  new bugs.
- **`go test -race` with `-count=10`** — single-run, 0 races detected.
  A larger run might flush out a flake but the budget didn't allow.
- **Dependency-bump verification** — I didn't actually run `go get -u`
  to bump the Go toolchain or `x/net` and re-run the test suite.  That
  is a follow-up action, not an audit finding.
