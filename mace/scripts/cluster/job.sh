#!/usr/bin/env bash
# Submit from login2 with an explicit one-GPU allocation on gpunode3.
# sbatch -p gpu -w gpunode3 --gres=gpu:rtx4090:1 -c 8 --mem=64G ... job.sh native
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/environment.sh"
cd "$MACE_CLUSTER_ROOT"
mkdir -p benchmarks/rtx4090/logs
printf 'Job %s on %s; CUDA_VISIBLE_DEVICES=%s; SLURM_JOB_GPUS=%s\n' \
    "$SLURM_JOB_ID" "$(hostname)" "$CUDA_VISIBLE_DEVICES" "$SLURM_JOB_GPUS"
if [[ "$CUDA_VISIBLE_DEVICES" == *,* || -z "$CUDA_VISIBLE_DEVICES" ]]; then
    echo 'Expected exactly one Slurm-assigned GPU' >&2
    exit 1
fi
nvidia-smi -i "$CUDA_VISIBLE_DEVICES"
gpu_processes=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)
if [[ -n "$gpu_processes" ]]; then
    printf 'Assigned GPU already has compute processes: %s\n' "$gpu_processes" >&2
    exit 1
fi
case "${1:?Specify native or a Python script}" in
    native)
        cmake -S .. -B ../build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \
            -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCUDAToolkit_ROOT=/usr/local/cuda \
            -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11 \
            -DMACE_BUILD_EXPLICIT_REFERENCE=ON
        cmake --build ../build -j4
        srun --cpu-bind=cores ../build/lapis/lapis_tests
        ;;
    *)
        srun --cpu-bind=cores python "$@"
        ;;
esac
