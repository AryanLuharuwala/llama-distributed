---
name: three-rig-bringup
description: End-to-end recipe for bringing up a 3-rig dist-pool spanning the local laptop (NVIDIA), rtxserver (RTX 4090, SSH-accessible), and ParamShakti (IIT KGP HPC, SLURM CPU/GPU partitions). Covers tunnels, pairing, the four landmines (pair-token vs invite-token, conda-gcc cache, dpp_runtime PYTHONPATH, second-local-rig agent_id), and the exact env vars dist-node needs.
---

# Three-rig bring-up — laptop + rtxserver + ParamShakti

This skill collects the actual procedures we learned, with the gotchas that
cost us hours, so the next bring-up takes minutes.

## TL;DR — five things to know

1. **The pair token comes from `/api/install_command`, NOT from `/api/pools/{id}/invite`.**
   The invite endpoint returns a *pool-membership* token; the agent-pairing
   path validates against `pair_tokens`. Symptom of mixing them up:
   `[pair] server reply: {"kind":"error","message":"bad pair token"}`.

2. **`dist-node` needs `DIST_DPP_PYTHONPATH` and `DIST_PYTHON` in its env**,
   otherwise the dpp_caps probe runs `python3 -c "import dpp_runtime.wire"`
   with relative path `../python` and exits 1. The right pair: a venv that
   already has `torch + diffusers`, plus `PYTHONPATH=<repo>/python`. Symptom:
   `[pair] dpp_caps ok=0 err=dpp_runtime not importable (exit=1)`.

3. **A second dist-node on the same host needs `XDG_STATE_HOME` pointed at a
   different dir AND `--id <unique>`** — otherwise it overwrites the first
   rig's saved agent_key and the server treats both connections as the same
   rig. Use `CUDA_VISIBLE_DEVICES=""` if you want it to act as a CPU-only
   stand-in.

4. **On ParamShakti, the cluster build needs gcc-13 *and* a wiped build dir
   *and* `-DGGML_NATIVE=OFF`.** The conda-bundled gcc 7.5 in PATH is missing
   `_mm256_set_m128i`; once cmake caches it via `find_program(CXX)`, no amount
   of `module load gcc/13.2.0` overrides it. The fix is in
   `jobs/shakti-build-v2.sh` (see below) — `rm -rf build`, `export
   CC=/home/apps/gcc-13/bin/gcc CXX=…/g++`, then configure on the *compute
   node* (not the login node), passing `-DCMAKE_C_COMPILER=$CC
   -DCMAKE_CXX_COMPILER=$CXX -DGGML_NATIVE=OFF` plus the AVX-512 off flags.

5. **Files on ParamShakti must live under `/scratch/testuser1`, not `$HOME`.**
   `$HOME` is shared GPFS and small; `/scratch` has 200+G free.

## Host roles

| Rig            | Where                | GPU             | Role in 3-stage DPP                |
|----------------|----------------------|-----------------|------------------------------------|
| laptop primary | local Fedora         | RTX 3050 (4 GB) | typically TE (light VRAM)          |
| rtxserver      | SSH `rtxserver`      | RTX 4090 (24 GB)| UNet stage 0 (heavy)               |
| ParamShakti    | SLURM `cpu`/`gpu`    | A100 / CPU      | UNet stage 1 OR VAE                |

When ParamShakti is offline, a second local dist-node (`shakti-fallback`)
substitutes — it exercises the planner/routing but everything runs on the
laptop's CPU under `CUDA_VISIBLE_DEVICES=""`.

## Step-by-step

### A. The control plane (laptop)

```bash
cd /home/boom/Desktop/Startup/DistibutedInference/llama-distributed
# Build the server (24 MB stripped)
cd server && go build -trimpath -ldflags="-s -w" -o /tmp/dist-int/dist-server . && cd ..

# Make sure no stale process holds :8080
pkill -f /tmp/dist-int/dist-server; sleep 2

# Start with the runtime-bundle dir set so /api/dpp/runtime/manifest works
cd /tmp/dist-int && \
  DIST_COMFY_OUT_DIR=/tmp/dist-int/comfy-out \
  DIST_PUBLIC_URL=http://127.0.0.1:8080 \
  DIST_ADDR=:8080 \
  DIST_DEV_MODE=1 \
  DIST_DB=/tmp/dist-int/distpool.sqlite \
  DIST_SESSION_SECRET=int-test-secret-do-not-use-aaaaaaaa \
  DPP_RUNTIME_DIR=/home/boom/Desktop/Startup/DistibutedInference/llama-distributed/python/dpp_runtime \
  nohup /tmp/dist-int/dist-server >/tmp/dist-int/server.log 2>&1 &
```

Smoke test:

```bash
curl -sf http://127.0.0.1:8080/api/dpp/runtime/manifest | python3 -m json.tool | head
# files: 6, tarball_sha256: 7cc06e3d…, signed: false (unsigned in dev)
```

### B. Mint a pair token (the right way)

```bash
curl -sf -b /tmp/dist-int/cookies.txt \
  -X POST http://127.0.0.1:8080/api/install_command \
  -H 'Content-Type: application/json' \
  -d '{"pool_id":1}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['token'])"
```

The token is **single-use** — each rig needs a fresh mint.

### C. Pair the laptop dist-node

```bash
PAIR=<token>
cat > /tmp/run-dist-laptop.sh <<EOF
#!/bin/bash
export DIST_DPP_PYTHONPATH=/home/boom/Desktop/Startup/DistibutedInference/llama-distributed/python
export DIST_PYTHON=/home/boom/.venv/bin/python3   # venv with torch+diffusers
cd /home/boom/Desktop/Startup/DistibutedInference/llama-distributed
exec ./build/dist-node --pair 'distpool://pair?token=$PAIR&server=ws://127.0.0.1:8080/ws/agent'
EOF
chmod +x /tmp/run-dist-laptop.sh
nohup /tmp/run-dist-laptop.sh >/tmp/dist-node-laptop.log 2>&1 &
disown
# Expect: dpp_caps ok=1 err=
```

### D. Pair rtxserver

Reverse-tunnel laptop's :8080 onto rtxserver:18505:

```bash
ssh -fN -o ExitOnForwardFailure=yes -R 18505:127.0.0.1:8080 rtxserver
# Persists until laptop disconnects
```

Stop any old dist-server / dist-node on rtxserver:

```bash
ssh rtxserver "pkill -f dist-serv; pkill -f dist-node; sleep 2"
```

Pair it. The `omni` conda env on rtxserver has torch 2.6 + diffusers 0.38:

```bash
PAIR=<fresh token from step B>
ssh rtxserver "cat > /tmp/run-dist.sh <<EOF
#!/bin/bash
export DIST_DPP_PYTHONPATH=/home/nishkal/llama-distributed/python
export DIST_PYTHON=/home/nishkal/anaconda3/envs/omni/bin/python
cd /home/nishkal/llama-distributed
exec ./build/dist-node --pair 'distpool://pair?token=$PAIR&server=ws://127.0.0.1:18505/ws/agent'
EOF
chmod +x /tmp/run-dist.sh
nohup /tmp/run-dist.sh >/tmp/dist-node-rtx.log 2>&1 &
disown"
```

### E. ParamShakti — proper build (avoids the conda-gcc trap)

Sync repo to /scratch (NOT $HOME):

```bash
rsync -a --delete \
  --exclude=build/ --exclude=.git/ --exclude=models-store/ \
  --exclude=third_party/llama.cpp/build/ \
  --exclude=third_party/ComfyUI/models/ \
  --exclude=distpool.sqlite* --exclude=audit/ \
  /home/boom/Desktop/Startup/DistibutedInference/llama-distributed/ \
  paramshakti:/scratch/testuser1/llama-distributed/
```

Submit `jobs/shakti-build-v2.sh` (in this skill dir as a sibling). It does:

- `module purge` then `module load gcc/13.2.0 compiler/cmake/3.22.5`
- `export CC=/home/apps/gcc-13/bin/gcc CXX=/home/apps/gcc-13/bin/g++`
- `rm -rf build && mkdir build && cd build` (the cache from any prior
  login-node configure is what burned us — it pinned conda gcc 7.5)
- `cmake .. -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX \`
  `-DDIST_USE_CUDA=OFF -DGGML_CUDA=OFF -DLLAMA_CUDA=OFF \`
  `-DGGML_NATIVE=OFF -DGGML_AVX512=OFF -DGGML_AVX512_VBMI=OFF \`
  `-DGGML_AVX512_VNNI=OFF -DGGML_AVX512_BF16=OFF`
- `cmake --build . -j 8 --target dist-node`

```bash
shakti -- mkdir -p /scratch/testuser1/logs /scratch/testuser1/jobs
rsync -av jobs/shakti-build-v2.sh paramshakti:/scratch/testuser1/jobs/
shakti -- sbatch /scratch/testuser1/jobs/shakti-build-v2.sh
shakti -- tail -F /scratch/testuser1/logs/distbuild-<jobid>.out
```

Expected: ~15 min wall on cpu partition.

### F. ParamShakti — reverse tunnel + pair

The compute node can reach the login node directly (internal IB/Ethernet);
the laptop needs a reverse tunnel to the login node:

```bash
# On laptop, background tunnel:
ssh -fN -R 18000:127.0.0.1:8080 paramshakti
```

In the run sbatch script, the compute node opens a second hop:

```bash
# Inside the sbatch script on the compute node:
ssh -fN -L 18000:127.0.0.1:18000 paramvidya   # paramvidya = login node host
DIST_API_URL=http://127.0.0.1:18000 ./build/dist-node \
    --pair "distpool://pair?token=$DIST_PAIR_TOKEN&server=ws://127.0.0.1:18000/ws/agent"
```

Submit with `--export=DIST_PAIR_TOKEN=<fresh>` so the token reaches the job
without going to disk.

### G. Second-local fallback rig (when ParamShakti isn't ready)

```bash
PAIR=<fresh token>
mkdir -p /tmp/dist-node-2
cat > /tmp/run-dist-laptop2.sh <<EOF
#!/bin/bash
export DIST_DPP_PYTHONPATH=/home/boom/Desktop/Startup/DistibutedInference/llama-distributed/python
export DIST_PYTHON=/home/boom/.venv/bin/python3
export XDG_STATE_HOME=/tmp/dist-node-2/state    # <-- separate state dir
export CUDA_VISIBLE_DEVICES=""                  # <-- act like a CPU rig
cd /home/boom/Desktop/Startup/DistibutedInference/llama-distributed
exec ./build/dist-node --id 'shakti-fallback:laptop2' \
     --pair 'distpool://pair?token=$PAIR&server=ws://127.0.0.1:8080/ws/agent'
EOF
chmod +x /tmp/run-dist-laptop2.sh
nohup /tmp/run-dist-laptop2.sh >/tmp/dist-node-laptop2.log 2>&1 &
disown
```

### H. Confirm the planner sees all three

```bash
curl -sf -b /tmp/dist-int/cookies.txt \
  http://127.0.0.1:8080/api/pools/1/rigs/dpp \
  | python3 -c "
import sys, json
for r in json.load(sys.stdin)['rigs']:
    print(f\"  rig{r['rig_id']:>2}  {r['agent_id']:<30}  online={r['online']!s:<5}  dpp={r.get('dpp_eligible',False)!s:<5}  vram={r.get('vram_free',0)//1024//1024} MiB\")
"
```

All three should show `dpp=True`.

## Triggering a 3-stage DPP run

```bash
curl -sf -b /tmp/dist-int/cookies.txt -X POST \
  http://127.0.0.1:8080/api/comfy/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "pool_id": 1,
    "prompt": "a photorealistic portrait of an astronaut, studio lighting",
    "model": "stabilityai/sdxl-turbo",
    "steps": 20, "cfg": 4.5, "width": 512, "height": 512, "seed": 42,
    "backend": "dpp",
    "unet_stages": 2
  }'
```

`unet_stages=2` means the planner picks 2 rigs for UNet block-split + 1 for
TE+VAE. With three eligible rigs it should spread one stage per rig.

Watch live progress at `http://127.0.0.1:8080/playground` (job appears in
the studio panel; the pipeline node-graph shows stage timing).

## Common pitfalls

- **`bad pair token`**: you used `/api/pools/{id}/invite` instead of
  `/api/install_command` — see point #1 above.
- **`dpp_caps ok=0`**: dist-node was launched without `DIST_DPP_PYTHONPATH`
  or `DIST_PYTHON`. The probe `python3 -c "import dpp_runtime.wire"` is the
  test; run it manually with the same env to debug.
- **Both rigs report the same agent_id**: second dist-node is loading the
  first rig's `agent_key` from `~/.local/state/llama-distributed`. Set
  `XDG_STATE_HOME=/tmp/somethingelse/state` for the second instance.
- **Cluster build dies with `_mm256_set_m128i was not declared`**: the
  conda gcc 7.5 in PATH is cached in `build/CMakeCache.txt`. `rm -rf build`
  + the env exports in step E fix it.
- **Cluster compute node can't reach laptop**: the laptop must open the
  reverse tunnel `ssh -R 18000:127.0.0.1:8080 paramshakti`; the compute
  node then opens `ssh -L 18000:127.0.0.1:18000 paramvidya` to relay.
- **rtxserver dist-node connects but immediately drops**: usually means a
  stale dist-server is still running on rtxserver listening on the port
  the rtxserver dist-node was paired to. Kill it: `ssh rtxserver "pkill
  -f dist-serv"`.

## Files in this skill

- `SKILL.md` — this file
- `jobs/shakti-build-v2.sh` — the working sbatch for paramshakti build
- `jobs/shakti-run.sh` — sbatch to pair + run dist-node on a compute node
  (template, fill in `DIST_PAIR_TOKEN` via `sbatch --export`)
- `scripts/run-dist-laptop.sh` — laptop primary launcher
- `scripts/run-dist-laptop2.sh` — second-local fallback launcher
- `scripts/run-dist-rtx.sh` — rtxserver launcher (deploys via ssh)
