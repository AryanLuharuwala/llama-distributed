# AGENTS.md — instructions for AI coding assistants helping a user install / run this repo

You are reading this because a user opened this repository in an AI coding tool
(Claude Code, Gemini Code Assist, Cursor, Cline, Aider, etc.) and asked for
help installing, building, pairing rigs, or running distributed inference.
Everything below was written after the maintainers hit the same problems
themselves; treat it as authoritative when it conflicts with general
training-data guesses.

If you are not an AI assistant — this is still a readable troubleshooting
guide. Skim "Quick install" then jump to the symptom-keyed sections below.

---

## What this project is, in one paragraph

`llama-distributed` is a control plane (`dist-server`) that orchestrates two
flavors of distributed inference across user-supplied GPU/CPU rigs
(`dist-node`):

1. **LLM pipeline parallel** — transformer layers split across rigs over a
   custom ACTV (activation) wire protocol layered on llama.cpp.
2. **DPP — Diffusion Pipeline Parallel** — text-encoder + UNet stages + VAE
   sharded across rigs, with the UNet itself further block-split. A separate
   Python runtime (`python/dpp_runtime/`) handles the actual diffusion work
   via `diffusers` and `torch`.

The control plane is a single Go binary. The agent is a single C++/CMake
binary. The diffusion runtime is plain Python.

---

## Quick install (the happy path)

```bash
# 1. control-plane prerequisites (Linux/macOS):
#    go >= 1.22, cmake >= 3.22, gcc/clang with C++17, libssl-dev
# 2. clone with submodules (llama.cpp is a submodule)
git clone --recursive https://github.com/<org>/llama-distributed.git
cd llama-distributed

# 3. build the server (Go)
cd server && go build -trimpath -ldflags="-s -w" -o ../build/dist-server .
cd ..

# 4. build the agent (C++)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target dist-node dist-server

# 5. install Python runtime deps (only needed on rigs that will do diffusion)
python3 -m venv .venv
source .venv/bin/activate
pip install torch diffusers transformers accelerate safetensors pillow
```

A successful build leaves you with `build/dist-server`, `build/dist-node`,
and `python/dpp_runtime/` ready to go.

---

## Running locally (single host)

```bash
# 1. Start the control plane
DIST_DEV_MODE=1 \
DIST_DB=/tmp/dist-int/distpool.sqlite \
DIST_PUBLIC_URL=http://127.0.0.1:8080 \
DIST_ADDR=:8080 \
DIST_SESSION_SECRET=dev-secret-change-me-aaaaaaaaaaaaaaaa \
DPP_RUNTIME_DIR=$(pwd)/python/dpp_runtime \
./build/dist-server

# 2. Sign in to the dashboard at http://127.0.0.1:8080 (browser device-code
#    flow in dev mode is automatic for the local user).

# 3. Generate a pair token:
curl -sf -b cookies.txt -X POST http://127.0.0.1:8080/api/install_command \
  -H 'Content-Type: application/json' -d '{"pool_id":1}'

# 4. Pair the agent (in a second terminal):
export DIST_DPP_PYTHONPATH=$(pwd)/python
export DIST_PYTHON=$(pwd)/.venv/bin/python      # venv with torch+diffusers
./build/dist-node --pair 'distpool://pair?token=<TOKEN>&server=ws://127.0.0.1:8080/ws/agent'
```

Look for `[pair] dpp_caps ok=1` in the agent log. If you see `ok=0`, see
the **dpp_caps** section below.

---

## Symptom-keyed troubleshooting

When the user says one of the things on the **left**, the answer is on the
**right**. Do NOT speculate beyond these — these are the actual failure
modes we hit.

### 🔴 `[pair] server reply: {"kind":"error","message":"bad pair token"}`

**Cause:** the user generated the token from `/api/pools/{id}/invite`
(a pool-membership invite) instead of `/api/install_command` (an
agent-pairing token). They are different tables in the DB.

**Fix:** mint via
```bash
curl -sf -b cookies.txt -X POST http://127.0.0.1:8080/api/install_command \
  -H 'Content-Type: application/json' -d '{"pool_id":1}' \
  | jq -r .token
```
Pair tokens are single-use; each rig needs its own mint.

---

### 🔴 `[pair] dpp_caps ok=0 err=dpp_runtime not importable (exit=1)`

**Cause:** the dist-node probe runs `python3 -c "import dpp_runtime.wire"`
with `PYTHONPATH=../python` (relative to its CWD) and a generic `python3`.
On most user installs, one of three things is wrong:
1. `python3` is the system python with no `torch`/`diffusers`
2. `PYTHONPATH` points at the wrong dir (you launched the agent from a
   different cwd than expected)
3. `dpp_runtime/` directory isn't present at all (e.g. they only copied the
   built binary, not the repo)

**Fix:** set explicit env vars on the agent process:
```bash
export DIST_DPP_PYTHONPATH=/absolute/path/to/llama-distributed/python
export DIST_PYTHON=/absolute/path/to/venv/with/torch/bin/python
./build/dist-node --pair '...'
```

**Manual verification (run this BEFORE relaunching the agent):**
```bash
PYTHONPATH=$DIST_DPP_PYTHONPATH $DIST_PYTHON -c "import dpp_runtime.wire; print('ok')"
```
If that prints `ok`, the agent will too.

**Fallback when python is unfixable (no internet, no sudo, ancient
distro):** build `dist-sdcpp-worker` and let the rig serve diffusion via
the vendored C++ backend instead. Same model coverage (SD1/2, SDXL,
PixArt, SD3/3.5, Flux); whole-pipeline single-rig only in phase A.
```bash
git submodule update --init --recursive third_party/stable-diffusion.cpp
cmake -B build -DDIST_USE_SDCPP=ON ...      # plus your usual flags
cmake --build build --target dist-sdcpp-worker -j
./build/dist-sdcpp-worker --probe           # validates ggml backends load
```
Smoke test directly (no agent needed):
```bash
./build/dist-sdcpp-worker \
  --model /path/to/sd15.safetensors \
  --prompt "a cat" --out /tmp/out.png \
  --w 512 --h 512 --steps 20 --cfg 7.0 --seed 42
```
This is the unblock path for **paramshakti cn01** and any other rig where
the python stack (`torch`+`diffusers`) is not reachable. Build-time
gotchas (CFLAGS contamination, missing /usr/include, FetchContent needs
git) all still apply — see the SLURM section below.

---

### 🔴 Two rigs show the same `agent_id` in the dashboard

**Cause:** the user launched a second `dist-node` on the same host. The
agent persists its identity to `$XDG_STATE_HOME/llama-distributed/`
(default `~/.local/state/llama-distributed`), so a second instance loads
the first instance's `agent_key` and the server treats both connections
as the same rig.

**Fix:** for each additional instance on the same host, point
`XDG_STATE_HOME` somewhere unique AND pass `--id <new-name>`:
```bash
XDG_STATE_HOME=/tmp/rig-2/state \
CUDA_VISIBLE_DEVICES="" \
./build/dist-node --id 'cpu-helper:1' --pair '...'
```

---

### 🔴 CMake build fails with `_mm256_set_m128i was not declared in this scope` (in `ggml-cpu/arch/x86/repack.cpp`)

**Cause:** an old gcc (typically gcc 7.5 from a conda sysroot) is being
picked up. cmake caches the compiler path in `build/CMakeCache.txt`, so
later `module load gcc/13.2.0` doesn't override it.

**Fix:**
```bash
rm -rf build
unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS LIBRARY_PATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH
export CC=/path/to/gcc-13
export CXX=/path/to/g++-13
cmake -B build -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX \
  -DGGML_NATIVE=OFF -DGGML_AVX512=OFF -DGGML_AVX512_VBMI=OFF \
  -DGGML_AVX512_VNNI=OFF -DGGML_AVX512_BF16=OFF
cmake --build build -j --target dist-node
```

Why each flag:
- wipe `build/` — cmake caches the compiler choice, so a fresh dir is the
  only reliable override
- `unset CFLAGS …` — conda environments export `-isystem
  /opt/.../conda/include` and `-march=nocona`; both poison the
  feature-detection compile tests
- `GGML_NATIVE=OFF` — stops ggml from baking the build host's CPU
  capabilities into binaries (important on HPC where login and compute
  nodes differ)
- `GGML_AVX512*=OFF` — these intrinsics are the ones gcc 7 chokes on

---

### 🔴 CMake configure fails: `Could NOT find Threads (missing: Threads_FOUND)` and `pthread.h - not found`

**Two distinct causes — diagnose first:**

```bash
test -f /usr/include/pthread.h && echo "system headers PRESENT" || echo "system headers MISSING"
echo "CFLAGS=${CFLAGS:-<unset>}"
```

**Case A — `/usr/include/pthread.h` exists but `CFLAGS` is set:**
Conda's `CFLAGS=-isystem /opt/.../conda/include …` overrides the system
include path with a directory that has no `pthread.h`.
**Fix:** `unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS LIBRARY_PATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH`
before invoking cmake. (Use the full invocation in the AVX section above.)

**Case B — `/usr/include` is absent entirely (stripped HPC compute-node image):**
The login node has glibc headers, the compute node doesn't. Probe with
`echo | $CC -E -Wp,-v - 2>&1 | head -20` — you'll see
`ignoring nonexistent directory "/usr/include"`. The conda installation
ships a full sysroot, so point gcc at it explicitly:
```bash
SYSROOT=/opt/ohpc/pub/apps/conda/x86_64-conda-linux-gnu/sysroot
test -f "$SYSROOT/usr/include/pthread.h" || { echo "no sysroot here"; exit 2; }
cmake -B build \
  -DCMAKE_C_FLAGS="--sysroot=$SYSROOT" \
  -DCMAKE_CXX_FLAGS="--sysroot=$SYSROOT" \
  -DCMAKE_EXE_LINKER_FLAGS="--sysroot=$SYSROOT" \
  -DCMAKE_SYSROOT="$SYSROOT" \
  ...other flags...
```
We borrow conda's *header tree*, NOT its CFLAGS — keep them unset.

---

### 🔴 CMake configure fails: `could not find git for clone of ftxui-populate` (or `LibDataChannel`)

**Cause:** CMake's `FetchContent` invokes `git` to fetch FTXUI (used by
`dist-cli top` TUI) and optionally LibDataChannel (P2P ACTV). HPC
compute nodes typically have no git installed; only the login node does.

**Two fixes, pick by need:**

1. **You don't need the TUI / P2P** (default for agents):
   ```bash
   cmake -B build -DDIST_BUILD_CLI_TUI=OFF -DDIST_USE_P2P=OFF ...
   ```
   `dist-node` (the agent) and `dist-server` don't require FTXUI.

2. **You want them** — pre-populate the sources on a host with git, then
   build on the compute node:
   ```bash
   # on login node (has git):
   cmake -B build -S .   # configure-only: triggers FetchContent → _deps/
   # then on compute node:
   cmake --build build -j --target dist-node
   ```
   The `_deps/ftxui-src` directory is reused; cmake won't re-fetch.

---

### 🔴 `comfy probe failed: comfyui unreachable at 127.0.0.1:8188`

This is not a fatal error — it just means the agent isn't advertising
ComfyUI capability. If the user doesn't need ComfyUI workflows, ignore
it. If they do, start ComfyUI on the agent host:
```bash
cd third_party/ComfyUI && python main.py --listen 127.0.0.1 --port 8188
```
Then re-pair with `DIST_WITH_COMFYUI=1`.

---

### 🔴 Agent connects, then immediately disconnects, with no error

**Most common cause:** a stale `dist-server` is also running locally and the
agent is talking to the wrong one. Check:
```bash
pgrep -fa dist-serv
ss -ltnp | grep -E '8080|18080|18505'
```
Kill the stale process and re-pair.

---

### 🔴 The agent runs but the dashboard shows it offline / `dpp_eligible: false`

The `dpp_eligible` flag requires *both* `dpp_caps ok=1` from the agent
AND `online=true` from the heartbeat. If `online=true` but
`dpp_eligible=false`, the dpp_caps probe failed — see the dpp_caps
section above. If `online=false` despite the agent appearing connected,
the heartbeat thread crashed (rare); restart the agent and capture
`/tmp/dist-node-*.log`.

---

### 🔴 Remote-rig pairing across NAT / firewalls

This repo uses a relay model — agents make outbound WebSocket connections
to the control plane, no inbound ports required. For a remote rig:

1. Open a reverse SSH tunnel from the control-plane host to the rig:
   ```bash
   ssh -fN -R 18505:127.0.0.1:8080 <rig-host>
   ```
2. On the rig, pair against `ws://127.0.0.1:18505/ws/agent` (not the
   control plane's public address — the rig dials its local tunnel
   endpoint).

This avoids opening any ports on either side.

---

### 🔴 HPC / SLURM specifics

If the user mentions SLURM, ParamShakti, or similar HPC clusters, also
read `skills/three-rig-bringup/SKILL.md` — it has the working cluster
build recipe, including the conda-CFLAGS gotcha above and a
`sbatch`-ready template at `skills/three-rig-bringup/jobs/`. Key points:

- Don't compile on the login node — submit a `sbatch` job
- Don't `module purge` (kills system headers); just unset the conda
  flags and load gcc/cmake modules
- Use `/scratch/<user>` for files, never `$HOME` (the GPFS quota is small)
- Reverse-tunnel from your laptop → cluster login node, then have the
  compute node forward through the login node back to the laptop

---

## What to do when nothing here matches

Before guessing:

1. **Read the actual error output.** Don't paraphrase to the user.
   Grep the source if a message is novel:
   ```bash
   grep -rn '<the error string>' server/ src/ python/
   ```
2. **Check `build/CMakeFiles/CMakeError.log`** for cmake failures —
   the configure-time test stdout/stderr lives there, not in the
   user-facing error.
3. **Verify env first.** Print `which gcc`, `which python`, `echo
   $CFLAGS`, `echo $PYTHONPATH` before running any build/agent. Most
   problems are env-vars leaking from a conda or pyenv setup.
4. **If a build worked before and stopped working** after a system
   change (new conda env, new system update, new module), suspect
   `CFLAGS`/`PATH` poisoning before code regressions.

---

## What this file is NOT

- It is **not** product documentation. For "what is DPP" / "how does the
  planner pick rigs" / API reference, see `README.md` and `docs/`.
- It is **not** a comprehensive bug list. It captures the install-time
  failure modes that real users actually hit; runtime/perf issues live
  elsewhere.
- It is **not** a substitute for reading the user's actual logs.
  Paste-and-grep, don't pattern-match.

---

## Telling the user where to file feedback

If the user hits a *new* installation failure not covered above and they
have it diagnosed enough to share:

- Open an issue at the repo, attach `build/CMakeFiles/CMakeError.log`
  and the agent's `/tmp/dist-node-*.log`
- Or, if they prefer, dump both logs into a gist and link it

This file (`AGENTS.md`) gets a new section per real-world failure mode,
so future agents don't repeat the diagnosis from scratch.
