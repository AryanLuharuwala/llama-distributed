# Signed builds — Bazel + rules_oci + cosign (P19)

This directory holds the scripts and CI workflow that take a Bazel-built
OCI image and bind a cosign signature to its digest before publication.
The deploy-side verification script (`cosign-verify.sh`) is what
production rigs run before `docker pull` so a tampered image can never
reach a `docker run`.

## Threat model

The supply chain we're hardening:

```
contributor laptop  →  GitHub PR  →  CI runner  →  ghcr.io  →  rig host  →  container runtime
       ↑                  ↑              ↑             ↑           ↑
    code review       review +       build +        registry     pull +
                      approval       sign           push         verify
```

The attacker classes we care about, in order of plausibility:

| class                                  | mitigation                                                              |
| -------------------------------------- | ----------------------------------------------------------------------- |
| Compromised contributor laptop         | Code review + branch protection; signing key never leaves the CI runner. |
| Malicious dependency (typosquatted Go) | `use_repo()` allowlist in MODULE.bazel; gazelle can't add deps silently. |
| Compromised CI runner                  | Keyless signing — no long-lived secret to steal; signatures are public-logged. |
| Compromised registry (ghcr.io)         | Verifying `certificate-identity` at pull time catches digest swaps.     |
| Compromised public Rekor               | Two-of-two: image must be in Rekor *and* certificate must match path.   |
| Compromised cosign itself              | Out of scope — we trust the cosign binary checksum from sigstore.       |

## Sign + verify flow

**Build + sign (CI):**

```
bazel test //:all_go --config=release
bazel build //:all_images --config=release
bazel run //server:dist-server_push -- --tag=v1.2.3
COSIGN_EXPERIMENTAL=1 cosign sign --yes \
    ghcr.io/llama-distributed/dist-server@sha256:<digest>
```

The `--config=release` profile (see `../.bazelrc`) sets
`SOURCE_DATE_EPOCH=0`, strips the host env, and pulls base images by
digest — so two CI runs on the same commit produce byte-identical
images.  This matters because cosign signs the *digest*, and a flaky
build that changes the digest invalidates the signature.

**Verify (rig host):**

```
COSIGN_MODE=keyless \
COSIGN_IDENTITY_REGEX='^https://github.com/AryanLuharuwala/llama-distributed/\.github/workflows/release\.yaml@refs/tags/v[0-9]+\.[0-9]+\.[0-9]+$' \
deploy/sign/cosign-verify.sh \
    --ref ghcr.io/llama-distributed/dist-server@sha256:<digest>
```

The `--certificate-identity-regexp` flag is non-negotiable.  Without
it cosign happily accepts any valid Fulcio cert, including one minted
by an attacker who landed a Workflow file in *their* repo and got an
OIDC token there.  Pinning to our workflow path closes that hole.

## Operating modes

The wrapper scripts (`cosign-sign.sh`, `cosign-verify.sh`) accept two
modes via `$COSIGN_MODE`:

- **`keyless`** (CI default) — uses GitHub Actions' OIDC token to mint
  a short-lived Fulcio certificate.  The signature, certificate, and
  Rekor inclusion proof are pushed to the registry as a sibling
  artifact (`@sha256:<digest>.sig`).  No long-lived secret exists; an
  attacker who briefly compromises the CI runner can sign images
  during that window but the signatures are publicly visible in
  Rekor's append-only log, so detection is straightforward.

- **`key`** (local default) — uses a key pair at `$COSIGN_KEY` /
  `$COSIGN_PUB`.  Faster, no network round-trip to Fulcio, but the
  key is a long-lived secret you have to protect.  We use this for
  local pre-tag testing only; CI always uses keyless.

## Files

| file                          | purpose                                                                            |
| ----------------------------- | ---------------------------------------------------------------------------------- |
| `cosign-sign.sh`              | Signs an `oci_image_index` Bazel output, invoked by `dist_go_image`'s genrule.    |
| `cosign-verify.sh`            | Verifies an image against a pinned cert identity. Runs at deploy time on rigs.     |
| `github-actions-sign.yaml`    | CI workflow — drop into `.github/workflows/release.yaml` when cutting over.        |
| `BUILD.bazel`                 | Exposes the scripts as `sh_binary` rules consumable by `tools/oci.bzl`.            |

## Why Bazel and not Docker buildx

The Dockerfile (`server/Dockerfile`) does produce a working image, but
two of its properties block reproducible signing:

1. `apt-get install ca-certificates sqlite3 libgomp1 libstdc++6` in
   the runtime stage pulls whatever Debian's mirror happens to have
   *that day*.  Two builds 24h apart produce different digests.
2. `COPY --from=builder` doesn't stamp mtime, so layer timestamps
   leak the build clock — another source of digest drift.

rules_oci builds layers with explicit, fixed mtime (the
`SOURCE_DATE_EPOCH=0` action env in `.bazelrc`) and pulls base images
by digest, not tag.  Combined with Bazel's hermetic action sandbox,
two builds of the same commit produce the same digest, which means
a signature minted today still verifies a year from now without a
re-sign.

The legacy `server/Dockerfile` stays in the tree for the duration of
the migration window so docker-compose deploys keep working; new
deploys should adopt the Bazel path.

## Migrating an existing operator

1. Install Bazelisk: `npm i -g @bazel/bazelisk` (or use the
   distro-native package).
2. From the repo root: `bazel build //:all_images` to confirm the
   build works locally.
3. Adopt the workflow at `github-actions-sign.yaml` — copy to
   `.github/workflows/release.yaml`, enable `id-token: write` on the
   repo, tag a release, and watch the Rekor entry land at
   <https://search.sigstore.dev/?email=>... (search by workflow path).
4. Wire `cosign-verify.sh` into the rig boot — the canonical place is
   a systemd `ExecStartPre=` on the `dist-node.service` unit, so a
   verification failure prevents the container from starting.

## What's left for P19+

- **C++ binaries** (`dist-node`, `dist-coordinator`, `dist-client`,
  `dist-cli`, `nat-portmap`) still build with CMake.  A full
  rules_cc + llama.cpp port to Bazel is multi-week work; the C++
  binaries today are shipped as static-link artifacts in GitHub
  Releases and signed at upload time, not via Bazel.
- **SLSA provenance** — cosign signing alone is level 2.  Adding
  `sigstore/sigstore-go`-generated SLSA v1.0 attestations (build
  type, build inputs, materials) gets us to level 3.  That's a
  follow-up; the current scaffold is provenance-ready (each image
  index already has a deterministic digest the attestation can bind
  to).
