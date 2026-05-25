#!/usr/bin/env bash
#SBATCH --job-name=distbuild
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/testuser1/logs/%x-%j.out
#SBATCH --error=/scratch/testuser1/logs/%x-%j.err
set -uo pipefail
echo "[$(date -Iseconds)] build job $SLURM_JOB_ID on $(hostname)"

# IMPORTANT: conda's activation hooks export CFLAGS/CXXFLAGS that include
# `-isystem /opt/ohpc/pub/apps/conda/include` (which has no pthread.h) and
# `-march=nocona -mtune=haswell` (locked to old arch). That set of flags
# gets baked into every cmake compile test, so pthread.h detection fails.
# Wipe them BEFORE loading any module.
unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS LIBRARY_PATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH

source /etc/profile.d/lmod.sh
module load gcc/13.2.0 compiler/cmake/3.22.5
export CC=/home/apps/gcc-13/bin/gcc
export CXX=/home/apps/gcc-13/bin/g++

# Re-confirm the env is clean after the modules load.
echo "CFLAGS=${CFLAGS:-<unset>}"
echo "CXXFLAGS=${CXXFLAGS:-<unset>}"
echo "CPPFLAGS=${CPPFLAGS:-<unset>}"
echo "PATH=$PATH"
which gcc g++ cmake
gcc --version | head -1
cmake --version | head -1
echo "pthread.h exists? $(test -f /usr/include/pthread.h && echo YES || echo NO)"

cd /scratch/testuser1/llama-distributed
rm -rf build
mkdir build && cd build

echo "--- configure ---"
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=$CC \
  -DCMAKE_CXX_COMPILER=$CXX \
  -DDIST_USE_CUDA=OFF \
  -DGGML_CUDA=OFF \
  -DLLAMA_CUDA=OFF \
  -DGGML_NATIVE=OFF \
  -DGGML_AVX512=OFF \
  -DGGML_AVX512_VBMI=OFF \
  -DGGML_AVX512_VNNI=OFF \
  -DGGML_AVX512_BF16=OFF \
  2>&1 | tail -40
cfg_rc=$?
echo "--- configure rc=$cfg_rc ---"
[ $cfg_rc -ne 0 ] && {
  echo "--- CMakeError.log tail ---"
  tail -30 CMakeFiles/CMakeError.log 2>&1
  exit $cfg_rc
}

echo "--- build ---"
cmake --build . -j 8 --target dist-node 2>&1 | tail -60
rc=$?
echo "--- build rc=$rc ---"
ls -la dist-node 2>&1 || true
file dist-node 2>&1 || true
exit $rc
