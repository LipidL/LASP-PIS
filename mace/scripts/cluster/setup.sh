#!/usr/bin/env bash
# Run on gpunode9; no GPU work is performed here.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/environment.sh"
cd "$MACE_CLUSTER_ROOT"
mkdir -p .python .cache benchmarks/rtx4090/logs
if [[ ! -x .python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12 ]]; then
    cp -a /home2/luohz/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu .python/
fi
uv_bin=/home2/luohz/.local/bin/uv
if [[ ! -x .venv-cu126/bin/python ]]; then
    "$uv_bin" venv --python "$PWD/.python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12" .venv-cu126
fi
if [[ -f scripts/cluster/requirements-cu126.lock ]]; then
    "$uv_bin" pip install --python .venv-cu126/bin/python -r scripts/cluster/requirements-cu126.lock
    "$uv_bin" pip install --python .venv-cu126/bin/python --no-deps -e ./mace-ref -e .
else
    "$uv_bin" pip install --python .venv-cu126/bin/python \
        torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
    "$uv_bin" pip install --python .venv-cu126/bin/python \
        'cuequivariance==0.11.1' 'cuequivariance-torch==0.11.1' \
        'cuequivariance-ops-torch-cu12==0.10.0' \
        'numpy==2.5.2' 'ase==3.29.0' 'matscipy==1.2.0' \
        nvidia-ml-py setuptools pytest -e ./mace-ref -e .
fi
"$uv_bin" pip check --python .venv-cu126/bin/python
"$uv_bin" pip freeze --python .venv-cu126/bin/python > benchmarks/rtx4090/environment.freeze.txt
python -c 'import torch; import cuequivariance_torch; import cuequivariance_ops_torch; print("torch", torch.__version__, "CUDA", torch.version.cuda)'
