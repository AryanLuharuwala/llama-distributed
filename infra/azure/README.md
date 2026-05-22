# Deploy distpool-server to Azure (alongside surd-server)

```
GitHub push (server/ changes)
   │
   ▼  (GHA workflow)
Build multi-arch image  ──►  ghcr.io/<you>/<repo>/distpool-server:<sha>  (free)
                                              │
                                              ▼  (GHA workflow)
                                  Azure Container App `distpool-server` pulls
                                  image, rolls a new revision, scales to zero
                                  when idle.
                                              │
                                              ▼
                                  /data persisted on Azure Files (50 GiB)
```

`distpool-server` is deployed as a **second Container App in the same
`surd-env` environment** as `surd-server`. They share the env (and therefore
the same VNet / internal DNS) but each owns its own Azure Files share so
storage isn't fighting.

## One-time setup

### 1. Push the repo to GitHub

The image must be reachable, and the workflow needs a place to live.

### 2. Make the GHCR package public (so Azure can pull anonymously)

After the first successful workflow run, go to:
- your GitHub profile → **Packages** → `distpool-server`
- **Package settings** → **Change visibility** → **Public**

If you'd rather keep it private, set `GHCR_USER` and `GHCR_PAT` env vars when
running `deploy.sh` — the script wires those in as a registry credential on
the Container App.

### 3. (Optional) Create a GitHub OAuth app for the dashboard

Visit https://github.com/settings/applications/new and create an OAuth app.
You won't know the final callback URL until after the first deploy, so just
pick a placeholder for now — you'll come back and fix it in step 5.

Take note of the **Client ID** and generate a **Client Secret**.

### 4. Run `deploy.sh` once (first-time provisioning)

```bash
# from the llama-distributed repo root
export GHCR_IMAGE=ghcr.io/<you>/<repo>/distpool-server

# Optional — leave unset and the server boots in dev mode
export DIST_GITHUB_CLIENT=Iv1.xxxxxxxxxxxxxxxx
export DIST_GITHUB_SECRET=ghcs_xxxxxxxxxxxxxxxx

az login                    # if not already
az account set --subscription <id>   # if you have multiple

bash infra/azure/deploy.sh
```

This creates / reuses:
- resource group `surd-rg` in `centralindia` (shared with surd-server)
- storage account + 50 GiB Azure Files share (`distpool-data`)
- Container Apps environment `surd-env` (created here if surd-server hasn't
  already)
- Container App `distpool-server` with min=0, max=1, 0.5 vCPU, 1.0 GiB

It prints the URL and the WebSocket endpoints native agents and browsers
should point at.

### 5. Set the OAuth callback URL on the GitHub OAuth app

After `deploy.sh` prints the FQDN, go back to your OAuth app's settings and
set the **Authorization callback URL** to:

```
https://<fqdn-from-deploy.sh>/auth/github/callback
```

That URL is also printed at the end of `deploy.sh` for convenience.

### 6. Wire up auto-deploy from GitHub Actions

If you already created the service principal for `surd-server`, reuse the
same `AZURE_CREDENTIALS` secret — it's scoped to the resource group and
covers both apps.

Otherwise, create one:

```bash
SUB=$(az account show --query id -o tsv)
az ad sp create-for-rbac \
  --name distpool-gha-deployer \
  --role contributor \
  --scopes /subscriptions/$SUB/resourceGroups/surd-rg \
  --sdk-auth
```

Copy the entire JSON output. In GitHub:

- **Settings → Secrets and variables → Actions → New repository secret**
  - `AZURE_CREDENTIALS` = (paste the JSON)
- (Optional) **Variables** tab:
  - `AZURE_RG` = `surd-rg`  (only if you changed the default)
  - `AZURE_APP` = `distpool-server`  (only if you changed the default)

From then on, every push to `main` that touches `server/` builds, publishes
to GHCR, and rolls the Azure Container App automatically. The deploy job
no-ops cleanly when `AZURE_CREDENTIALS` isn't set, so the workflow still
works in forks or before you've provisioned Azure.

## Updating secrets after deploy

Add or rotate OAuth credentials without re-running the full script:

```bash
az containerapp secret set -g surd-rg -n distpool-server \
  --secrets "github-secret=<new-secret>" "session-secret=<new-secret>"

az containerapp update -g surd-rg -n distpool-server \
  --set-env-vars DIST_GITHUB_CLIENT=<new-client-id>
```

## Costs (sizing for control-plane workloads)

| Component | Est. monthly |
|---|---|
| Container Apps Consumption (0.5 vCPU / 1.0 GiB, scaled to zero) | a few $/mo at light traffic; free-tier covers idle |
| Azure Files (50 GiB Standard_LRS) | ~$3/mo |
| GHCR (public package) | $0 |
| Egress | dominated by inference relay traffic — sized by your usage |
| **Total** | **~$3-10/mo** for personal/small-team use |

The cost only climbs meaningfully if the server starts relaying large amounts
of inference traffic, or if the models/releases caches on `/data` grow toward
the 50 GiB quota. Both are tunable from the env vars.

## Tear down

Just this app (keeps `surd-server` and the env alive):

```bash
az containerapp delete -g surd-rg -n distpool-server --yes
# storage account name is printed by deploy.sh — grab it from there
az storage account delete -g surd-rg -n <storage-account-name> --yes
```

Everything (both apps, env, storage):

```bash
az group delete --name surd-rg --yes --no-wait
```
