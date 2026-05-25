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
source /etc/profile.d/lmod.sh
# DO NOT module purge — strips pthread.h from the search path.
module load gcc/13.2.0 compiler/cmake/3.22.5
export CC=/home/apps/gcc-13/bin/gcc
export CXX=/home/apps/gcc-13/bin/g++
# Make sure system headers (pthread.h, etc) are visible to gcc-13.
export C_INCLUDE_PATH="/usr/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
export CPLUS_INCLUDE_PATH="/usr/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
which gcc g++ cmake
gcc --version | head -1
cmake --version | head -1

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
[ $cfg_rc -ne 0 ] && exit $cfg_rc

echo "--- build ---"
cmake --build . -j 8 --target dist-node 2>&1 | tail -50
rc=$?
echo "--- build rc=$rc ---"
ls -la dist-node 2>&1 || true
exit $rc
