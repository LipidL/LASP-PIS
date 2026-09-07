#!/usr/bin/env bash
# Source from the checkout on the cluster. Keep all writable caches on home15.
export PAIRWISE_CLUSTER_ROOT=/home15/luohz/ppf_gpt_benchmark
export UV_CACHE_DIR="$PAIRWISE_CLUSTER_ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PAIRWISE_CLUSTER_ROOT/.python"
export XDG_CACHE_HOME="$PAIRWISE_CLUSTER_ROOT/.cache"
export MPLCONFIGDIR="$PAIRWISE_CLUSTER_ROOT/.cache/matplotlib"
export TORCHINDUCTOR_CACHE_DIR="$PAIRWISE_CLUSTER_ROOT/.cache/inductor"
export TRITON_CACHE_DIR="$PAIRWISE_CLUSTER_ROOT/.cache/triton"
export CUDA_CACHE_PATH="$PAIRWISE_CLUSTER_ROOT/.cache/cuda"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export TORCHINDUCTOR_COMPILE_THREADS=4
export PATH="$PAIRWISE_CLUSTER_ROOT/.venv-cu126/bin:/home14/luohz/lib/cmake/build/bin:/usr/local/cuda/bin:$PATH"
export PYTHONPATH="$PAIRWISE_CLUSTER_ROOT/python:$PAIRWISE_CLUSTER_ROOT/mace-ref"
export PAIRWISE_LIBRARY="$PAIRWISE_CLUSTER_ROOT/build/libpairwise.so"
# Use the same CUDA runtime and cuBLAS for native-only and PyTorch comparisons.
# The native SM89 binary is compiled with the node's CUDA 12.4 toolkit.
export LD_LIBRARY_PATH="$PAIRWISE_CLUSTER_ROOT/.venv-cu126/lib/python3.12/site-packages/nvidia/cublas/lib:$PAIRWISE_CLUSTER_ROOT/.venv-cu126/lib/python3.12/site-packages/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
