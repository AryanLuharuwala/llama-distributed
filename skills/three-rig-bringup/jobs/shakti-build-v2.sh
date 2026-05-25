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
module purge
module load gcc/13.2.0 compiler/cmake/3.22.5
export CC=/home/apps/gcc-13/bin/gcc
export CXX=/home/apps/gcc-13/bin/g++
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
  2>&1 | tail -30
echo "--- build ---"
cmake --build . -j 8 --target dist-node 2>&1 | tail -50
rc=$?
echo "--- build rc=$rc ---"
ls -la dist-node 2>&1 || true
exit $rc
