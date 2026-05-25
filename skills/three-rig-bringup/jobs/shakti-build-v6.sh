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

# Wipe conda CFLAGS (their -march=nocona/-isystem poison cmake tests)
unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS LIBRARY_PATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH

source /etc/profile.d/lmod.sh
module load gcc/13.2.0 compiler/cmake/3.22.5
export CC=/home/apps/gcc-13/bin/gcc
export CXX=/home/apps/gcc-13/bin/g++

# Compute nodes lack /usr/include — point gcc at the conda sysroot which has
# the full glibc headers. We DO NOT inherit conda CFLAGS; we just borrow the
# header tree.
SYSROOT=/opt/ohpc/pub/apps/conda/x86_64-conda-linux-gnu/sysroot
test -f "$SYSROOT/usr/include/pthread.h" || { echo "FATAL: no pthread.h at $SYSROOT"; exit 2; }
export CFLAGS="--sysroot=$SYSROOT"
export CXXFLAGS="--sysroot=$SYSROOT"
export LDFLAGS="--sysroot=$SYSROOT"

echo "--- env ---"
which gcc g++ cmake
gcc --version | head -1
echo "SYSROOT=$SYSROOT"
echo "pthread.h via sysroot: $(test -f $SYSROOT/usr/include/pthread.h && echo YES)"
echo "test pthread compile:"
echo "#include <pthread.h>" > /tmp/t.c
echo "int main(){return 0;}" >> /tmp/t.c
$CC $CFLAGS /tmp/t.c -lpthread -o /tmp/t.out 2>&1 && echo "  pthread compile OK" || echo "  pthread compile FAILED"

cd /scratch/testuser1/llama-distributed
rm -rf build
mkdir build && cd build

echo "--- configure ---"
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=$CC \
  -DCMAKE_CXX_COMPILER=$CXX \
  -DCMAKE_C_FLAGS="--sysroot=$SYSROOT" \
  -DCMAKE_CXX_FLAGS="--sysroot=$SYSROOT" \
  -DCMAKE_EXE_LINKER_FLAGS="--sysroot=$SYSROOT" \
  -DCMAKE_SYSROOT="$SYSROOT" \
  -DDIST_USE_CUDA=OFF \
  -DGGML_CUDA=OFF \
  -DLLAMA_CUDA=OFF \
  -DGGML_NATIVE=OFF \
  -DGGML_AVX512=OFF \
  -DGGML_AVX512_VBMI=OFF \
  -DGGML_AVX512_VNNI=OFF \
  -DGGML_AVX512_BF16=OFF -DDIST_BUILD_CLI_TUI=OFF \
  2>&1 | tail -50
cfg_rc=${PIPESTATUS[0]}
echo "--- configure rc=$cfg_rc ---"
[ $cfg_rc -ne 0 ] && {
  echo "--- CMakeError.log tail ---"
  tail -50 CMakeFiles/CMakeError.log 2>&1
  exit $cfg_rc
}

echo "--- build ---"
cmake --build . -j 8 --target dist-node 2>&1 | tail -80
rc=${PIPESTATUS[0]}
echo "--- build rc=$rc ---"
ls -la dist-node 2>&1 || true
file dist-node 2>&1 || true
exit $rc
