#!/usr/bin/env bash
# cosign-verify — verify a signed image before deploy.
#
# Run on every dist-server / dist-turn pull at the edge — operators
# wire this into the systemd unit `ExecStartPre=` so a tampered image
# can never reach `docker run`.  Two modes mirror cosign-sign:
#
#   COSIGN_MODE=keyless  → verify against Fulcio + Rekor (CI-built)
#                          using --certificate-identity to bind the
#                          signature to the expected GitHub workflow
#                          identity.  This is the strong check.
#   COSIGN_MODE=key      → verify against $COSIGN_PUB (local builds)

set -euo pipefail

REF=""
MODE="${COSIGN_MODE:-keyless}"
IDENTITY_REGEX="${COSIGN_IDENTITY_REGEX:-^https://github.com/AryanLuharuwala/llama-distributed/\\.github/workflows/release\\.yaml@refs/tags/v[0-9]+\\.[0-9]+\\.[0-9]+$}"
ISSUER="${COSIGN_OIDC_ISSUER:-https://token.actions.githubusercontent.com}"

while [ $# -gt 0 ]; do
    case "$1" in
        --ref)    REF="$2"; shift 2 ;;
        --mode)   MODE="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [ -z "$REF" ]; then
    echo "usage: cosign-verify --ref <repository@sha256:...>" >&2
    exit 2
fi

if ! command -v cosign >/dev/null 2>&1; then
    echo "cosign(1) not found on PATH" >&2
    exit 1
fi

case "$MODE" in
    keyless)
        # The two flags below are the *whole* point: without them
        # cosign verify will accept any signature from any Fulcio cert
        # whose chain validates, including a signature minted by an
        # attacker who got an OIDC token from their own repo.  We pin
        # the cert identity to our release workflow path.
        COSIGN_EXPERIMENTAL=1 cosign verify "$REF" \
            --certificate-identity-regexp "$IDENTITY_REGEX" \
            --certificate-oidc-issuer    "$ISSUER" \
            > /dev/null
        ;;
    key)
        if [ -z "${COSIGN_PUB:-}" ]; then
            echo "COSIGN_MODE=key requires COSIGN_PUB=<path-to-cosign.pub>" >&2
            exit 1
        fi
        cosign verify --key "$COSIGN_PUB" "$REF" > /dev/null
        ;;
    *)
        echo "unknown COSIGN_MODE=$MODE" >&2
        exit 2
        ;;
esac

echo "[cosign-verify] $REF: verified"
