# Benchmark reproduction recipe. The Python lock is the supplied environment;
# this image also builds the independent CUDA inference library from source.
FROM nvidia/cuda:12.6.2-devel-ubuntu22.04
COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake git ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
ENV UV_PYTHON_INSTALL_DIR=/opt/python
RUN uv python install 3.12.13
COPY docker/requirements.lock /tmp/requirements.lock
RUN uv venv --python 3.12.13 /opt/venv && uv pip install --python /opt/venv/bin/python -r /tmp/requirements.lock cmake==4.0.2
ENV PATH="/opt/venv/bin:${PATH}"
COPY . /workspace
RUN uv pip install --python /opt/venv/bin/python --no-deps -e ./mace-ref -e .
ARG CUDA_ARCHITECTURES="80;86;89;90"
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" -DPAIRWISE_BUILD_EXPLICIT_REFERENCE=ON && cmake --build build -j4
ENV PAIRWISE_LIBRARY=/workspace/build/libpairwise.so
ENV MPLCONFIGDIR=/tmp/pairwise-matplotlib
ENV TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
CMD ["python", "tools/benchmark.py", "--quick"]
