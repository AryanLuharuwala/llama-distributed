#!/usr/bin/env bash
# cosign-sign — sign an OCI image produced by `bazel build :<name>`.
#
# Two operating modes, selected by env:
#
#   COSIGN_MODE=keyless   (default in CI; uses GitHub-Actions OIDC token
#                          to mint a short-lived signing cert from
#                          Fulcio, transparency-logs to Rekor)
#   COSIGN_MODE=key       (default locally; expects a key pair at
#                          $COSIGN_KEY / $COSIGN_PUB)
#
# Why both: keyless is the modern best practice — no secret to leak,
# every signature is publicly auditable in Rekor.  But it requires
# OIDC, which doesn't exist on a developer laptop.  Key-based signing
# is what we use locally during development.
#
# The signature is written to `--output`, which Bazel captures as the
# action output.  This file is *not* the signature itself (cosign
# pushes that to the OCI registry as a sibling artifact); rather it's
# an attestation manifest you can `cosign verify` against later.

set -euo pipefail

IMAGE=""
REPOSITORY=""
OUTPUT=""
MODE="${COSIGN_MODE:-keyless}"

while [ $# -gt 0 ]; do
    case "$1" in
        --image)      IMAGE="$2"; shift 2 ;;
        --repository) REPOSITORY="$2"; shift 2 ;;
        --output)     OUTPUT="$2"; shift 2 ;;
        --mode)       MODE="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [ -z "$IMAGE" ] || [ -z "$REPOSITORY" ] || [ -z "$OUTPUT" ]; then
    echo "usage: cosign-sign --image <oci_index_path> --repository <ref> --output <file>" >&2
    exit 2
fi

if ! command -v cosign >/dev/null 2>&1; then
    echo "cosign(1) not found on PATH. Install from https://docs.sigstore.dev/cosign/installation/" >&2
    # Emit a placeholder attestation so the Bazel action still succeeds in dry-run
    # mode (e.g. on a fresh checkout where the dev hasn't installed cosign yet).
    # The genrule's `tags = ["manual"]` keeps this off the default build path so
    # this fallback only fires when the developer explicitly opts in.
    cat > "$OUTPUT" <<EOF
{
  "status":   "skipped",
  "reason":   "cosign binary not found on PATH",
  "image":    "$IMAGE",
  "repository": "$REPOSITORY",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
    exit 0
fi

# Resolve the image's digest.  Bazel writes an oci_image_index to a
# directory; the manifest digest is in `index.json` → manifests[0].digest.
if [ -d "$IMAGE" ] && [ -f "$IMAGE/index.json" ]; then
    DIGEST=$(jq -r '.manifests[0].digest' "$IMAGE/index.json")
else
    echo "image path '$IMAGE' is not an OCI image layout (no index.json)" >&2
    exit 1
fi

REF="${REPOSITORY}@${DIGEST}"
echo "[cosign-sign] signing $REF (mode=$MODE)"

case "$MODE" in
    keyless)
        # In a GitHub Actions workflow, COSIGN_EXPERIMENTAL=1 plus the
        # id-token: write permission gives cosign the OIDC token it
        # needs to mint a short-lived Fulcio certificate.  The
        # signature + cert + Rekor entry are pushed to the registry
        # as a sibling artifact at $REF.sig.
        COSIGN_EXPERIMENTAL=1 cosign sign --yes "$REF" \
            --output-signature "$OUTPUT.sig" \
            --output-certificate "$OUTPUT.cert"
        ;;
    key)
        if [ -z "${COSIGN_KEY:-}" ]; then
            echo "COSIGN_MODE=key requires COSIGN_KEY=<path-to-cosign.key>" >&2
            exit 1
        fi
        cosign sign --key "$COSIGN_KEY" --yes "$REF" \
            --output-signature "$OUTPUT.sig"
        ;;
    *)
        echo "unknown COSIGN_MODE=$MODE (want keyless or key)" >&2
        exit 2
        ;;
esac

# Emit the attestation manifest Bazel captures.
cat > "$OUTPUT" <<EOF
{
  "status":     "signed",
  "mode":       "$MODE",
  "image":      "$IMAGE",
  "repository": "$REPOSITORY",
  "digest":     "$DIGEST",
  "ref":        "$REF",
  "timestamp":  "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
echo "[cosign-sign] wrote attestation → $OUTPUT"
