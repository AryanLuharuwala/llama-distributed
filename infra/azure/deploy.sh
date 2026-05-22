#!/usr/bin/env bash
# First-time deploy: provisions a `distpool-server` Container App that lives
# alongside `surd-server` in the same Container Apps environment (surd-env).
# Subsequent image updates are done by .github/workflows/build-and-publish.yml
# so this script only needs to be re-run if you change infra shape.
#
# Image source: ghcr.io (free, fed by GitHub Actions). No ACR needed.
#
# Sizing (matches what we discussed for distpool):
#   - Container Apps Consumption: 0.5 vCPU / 1 GiB
#   - minReplicas=0 (scale to zero when idle)
#   - Azure Files 50 GiB Standard_LRS so model shards + releases cache fit
#   - External HTTPS ingress on :8080
#
# Caveats of scale-to-zero for distpool:
#   - First WS connect after idle pays a 10-30s cold start
#   - SQLite reopens from /data on cold start, no data loss
#   - Native agents reconnect automatically, browsers prompt on disconnect

set -euo pipefail

# ─── required env ────────────────────────────────────────────────────────
: "${GHCR_IMAGE:?Set GHCR_IMAGE to e.g. ghcr.io/youruser/llama-distributed/distpool-server}"
# Docker image refs must be lowercase. GHA's docker/metadata-action lowercases
# automatically, so the pushed image always lives at the lowercase path.
GHCR_IMAGE="${GHCR_IMAGE,,}"

# ─── tuneable ────────────────────────────────────────────────────────────
# Defaults intentionally reuse the Sharing/surd resource group + environment
# so the two apps share a Container Apps environment (option B from the design
# discussion). Override if you want them in separate envs.
RG="${RG:-surd-rg}"
LOCATION="${LOCATION:-centralindia}"
APP="${APP:-distpool-server}"
ENV_NAME="${ENV_NAME:-surd-env}"
# Separate storage account so distpool's 50 GiB share doesn't compete with
# surd-server's 5 GiB. Name is deterministic per subscription so re-runs
# reuse the same account. 3-24 chars, lowercase + digits only.
STORAGE_ACCOUNT="${STORAGE_ACCOUNT:-distpoolstg$(az account show --query id -o tsv | sha256sum | cut -c1-10)}"
FILE_SHARE="${FILE_SHARE:-distpool-data}"
FILE_SHARE_QUOTA_GB="${FILE_SHARE_QUOTA_GB:-50}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# OAuth + session secrets. If DIST_GITHUB_CLIENT / DIST_GITHUB_SECRET are unset
# we still deploy; the server will boot in dev mode (no GitHub login) and you
# can fill them in later via `az containerapp update --set-env-vars`.
DIST_GITHUB_CLIENT="${DIST_GITHUB_CLIENT:-}"
DIST_GITHUB_SECRET="${DIST_GITHUB_SECRET:-}"
DIST_SESSION_SECRET="${DIST_SESSION_SECRET:-$(openssl rand -hex 32)}"

echo "─── config ──────────────────────────────────"
echo "resource group:    $RG"
echo "location:          $LOCATION"
echo "image:             ${GHCR_IMAGE}:${IMAGE_TAG}"
echo "container app:     $APP"
echo "environment:       $ENV_NAME  (shared with surd-server if it exists)"
echo "storage account:   $STORAGE_ACCOUNT"
echo "file share:        $FILE_SHARE  (${FILE_SHARE_QUOTA_GB} GiB)"
echo "compute:           0.5 vCPU / 1 GiB, min=0 max=1"
if [[ -z "$DIST_GITHUB_CLIENT" || -z "$DIST_GITHUB_SECRET" ]]; then
  echo "github oauth:      UNSET — server will boot in dev mode"
else
  echo "github oauth:      configured"
fi
echo "session secret:    (generated, will be stored as a secret on the app)"
echo "────────────────────────────────────────────"

az extension add --name containerapp --only-show-errors --upgrade 2>/dev/null || true
az provider register --namespace Microsoft.App --wait --only-show-errors >/dev/null
az provider register --namespace Microsoft.OperationalInsights --wait --only-show-errors >/dev/null

# ─── 1. resource group ───────────────────────────────────────────────────
az group create --name "$RG" --location "$LOCATION" --output none

# ─── 2. storage for /data ────────────────────────────────────────────────
az storage account create \
  --resource-group "$RG" --name "$STORAGE_ACCOUNT" \
  --location "$LOCATION" --sku Standard_LRS \
  --output none
STORAGE_KEY=$(az storage account keys list -g "$RG" -n "$STORAGE_ACCOUNT" --query '[0].value' -o tsv)
az storage share-rm create \
  --resource-group "$RG" --storage-account "$STORAGE_ACCOUNT" \
  --name "$FILE_SHARE" --quota "$FILE_SHARE_QUOTA_GB" \
  --output none

# ─── 3. container apps environment + volume ──────────────────────────────
# Env may already exist (created by surd-server's deploy.sh). show-then-create
# so re-runs and the shared-env path are both safe.
if ! az containerapp env show -g "$RG" -n "$ENV_NAME" >/dev/null 2>&1; then
  az containerapp env create \
    --resource-group "$RG" --name "$ENV_NAME" \
    --location "$LOCATION" \
    --logs-destination none \
    --output none
fi

# Storage entries in an environment are keyed by storage-name. We use a name
# distinct from surddata so both apps can mount their own shares.
if ! az containerapp env storage show \
       -g "$RG" -n "$ENV_NAME" --storage-name distpooldata >/dev/null 2>&1; then
  az containerapp env storage set \
    --resource-group "$RG" --name "$ENV_NAME" \
    --storage-name distpooldata \
    --azure-file-account-name "$STORAGE_ACCOUNT" \
    --azure-file-account-key "$STORAGE_KEY" \
    --azure-file-share-name "$FILE_SHARE" \
    --access-mode ReadWrite \
    --output none
fi

# ─── 4. deploy the container app (pulls from ghcr.io) ────────────────────
# Two-phase deploy because DIST_PUBLIC_URL needs the FQDN, which we don't know
# until the app exists. Phase 1: create with a placeholder. Phase 2: update
# DIST_PUBLIC_URL + DIST_APEX_HOST to the real FQDN.
#
# ACA's YAML merge replaces full arrays (secrets, registries) wholesale, so
# when GHCR is private we build a single spec that includes both the registry
# credential and the app secrets together.
TMP_YAML=$(mktemp --suffix=.yaml)
{
  cat <<EOF
properties:
  template:
    containers:
      - name: distpool-server
        image: ${GHCR_IMAGE}:${IMAGE_TAG}
        env:
          - name: DIST_ADDR
            value: ":8080"
          - name: DIST_DB
            value: /data/distpool.sqlite
          - name: DIST_MODELS_DIR
            value: /data/models
          - name: DIST_RELEASES_DIR
            value: /data/releases-cache
          # Placeholder — overwritten in phase 2 once the FQDN is known.
          - name: DIST_PUBLIC_URL
            value: "http://placeholder.invalid"
          - name: DIST_APEX_HOST
            value: "placeholder.invalid"
          - name: DIST_GITHUB_CLIENT
            value: "${DIST_GITHUB_CLIENT}"
          - name: DIST_GITHUB_SECRET
            secretRef: github-secret
          - name: DIST_SESSION_SECRET
            secretRef: session-secret
        resources:
          cpu: 0.5
          memory: 1.0Gi
        volumeMounts:
          - volumeName: data
            mountPath: /data
        probes:
          - type: Liveness
            httpGet:
              path: /healthz
              port: 8080
            initialDelaySeconds: 20
            periodSeconds: 60
    scale:
      minReplicas: 0
      maxReplicas: 1
    volumes:
      - name: data
        storageType: AzureFile
        storageName: distpooldata
  configuration:
    ingress:
      external: true
      targetPort: 8080
      transport: auto
      allowInsecure: false
EOF

  if [[ -n "${GHCR_USER:-}" && -n "${GHCR_PAT:-}" ]]; then
    cat <<EOF
    registries:
      - server: ghcr.io
        username: ${GHCR_USER}
        passwordSecretRef: ghcr-pat
    secrets:
      - name: github-secret
        value: "${DIST_GITHUB_SECRET}"
      - name: session-secret
        value: "${DIST_SESSION_SECRET}"
      - name: ghcr-pat
        value: "${GHCR_PAT}"
EOF
  else
    cat <<EOF
    secrets:
      - name: github-secret
        value: "${DIST_GITHUB_SECRET}"
      - name: session-secret
        value: "${DIST_SESSION_SECRET}"
EOF
  fi
} > "$TMP_YAML"

if az containerapp show -g "$RG" -n "$APP" >/dev/null 2>&1; then
  az containerapp update -g "$RG" -n "$APP" --yaml "$TMP_YAML" --output none
else
  az containerapp create -g "$RG" -n "$APP" --environment "$ENV_NAME" --yaml "$TMP_YAML" --output none
fi
rm -f "$TMP_YAML"

FQDN=$(az containerapp show -g "$RG" -n "$APP" --query properties.configuration.ingress.fqdn -o tsv)
PUBLIC_URL="https://$FQDN"

# Phase 2: now that we know the FQDN, fix DIST_PUBLIC_URL and DIST_APEX_HOST.
# OAuth callback construction relies on these matching the actual hostname.
az containerapp update -g "$RG" -n "$APP" \
  --set-env-vars \
    "DIST_PUBLIC_URL=$PUBLIC_URL" \
    "DIST_APEX_HOST=$FQDN" \
  --output none

echo
echo "─── deployed ────────────────────────────────"
echo "URL:           $PUBLIC_URL"
echo "Health:        $PUBLIC_URL/healthz"
echo "Agent WS:      wss://$FQDN/ws/agent"
echo "Browser WS:    wss://$FQDN/ws/browser"
echo
if [[ -n "$DIST_GITHUB_CLIENT" && -n "$DIST_GITHUB_SECRET" ]]; then
  echo "GitHub OAuth callback to register on your OAuth app:"
  echo "  $PUBLIC_URL/auth/github/callback"
  echo
fi
echo "Tear down distpool only:"
echo "  az containerapp delete -g $RG -n $APP --yes"
echo "  az storage account delete -g $RG -n $STORAGE_ACCOUNT --yes"
echo
echo "Tear down everything (this app AND surd-server):"
echo "  az group delete --name $RG --yes --no-wait"
