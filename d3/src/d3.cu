#include <algorithm>
#include <cmath>
#include <lapis/pair.cuh>

#include "d3/d3.hpp"

namespace d3 {
namespace {
namespace table {
#include "parameters.inc"
}
constexpr int threads = 128;

template<int N>
__device__ void reduce(double (&values)[N], void* scratch) {
    auto p = static_cast<double*>(scratch);
    int  t = threadIdx.x;
    for (int k = 0; k < N; ++k)
        p[k * threads + t] = values[k];
    __syncthreads();
    for (int stride = threads / 2; stride; stride /= 2) {
        if (t < stride)
            for (int k = 0; k < N; ++k)
                p[k * threads + t] += p[k * threads + t + stride];
        __syncthreads();
    }
    if (t == 0)
        for (int k = 0; k < N; ++k)
            values[k] = p[k * threads];
}
struct BaseOp {
    static constexpr int threads = d3::threads, tile_edges = 32, scratch_alignment = 16;
    void                 setup(lapis::Context&, void*) {}
    void                 postprocess(lapis::Context&, void*) {}
};
struct Coordination : BaseOp {
    static const char* name() { return "d3.coordination"; }
    struct State {
        double sum[1] = {};
    };
    const int*           elements;
    const double*        rcov;
    double*              cn;
    int*                 status;
    lapis::WorkspaceSpec plan(int) const { return {0, threads * sizeof(double), 16}; }
    __device__ void      prepare(int, State&, void*) {}
    template<class Tile>
    __device__ void consume(int i, const Tile& tile, State& s, void*) {
        int e = threadIdx.x;
        if (e >= tile.size())
            return;
        double r = sqrt(double(tile.squared_distance(e))) / bohr;
        if (r == 0) {
            atomicCAS(status, 0, 1);
            return;
        }
        double radius = rcov[elements[i]] + rcov[elements[tile.sender(e)]];
        s.sum[0] += 1 / (1 + exp(-16 * (radius / r - 1)));
    }
    __device__ void finalize(int i, State& s, void* scratch) {
        reduce(s.sum, scratch);
        if (threadIdx.x == 0)
            cn[i] = s.sum[0];
    }
};

// The Gaussian interpolation separates into per-atom normalized weights.
// Subtract the largest exponent to remain finite for highly coordinated atoms.
__global__ void weights(int n, const int* elements, const int* offsets, const double* reference_cn,
                        const double* cn, double* w, double* dw) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    int    start = offsets[elements[i]], count = offsets[elements[i] + 1] - start;
    double exponent[table::max_reference], sum = 0, mean = 0, largest = -INFINITY;
    for (int a = 0; a < count; ++a) {
        double delta = cn[i] - reference_cn[start + a];
        exponent[a]  = -4 * delta * delta;
        largest      = fmax(largest, exponent[a]);
    }
    for (int a = 0; a < count; ++a) {
        exponent[a] = exp(exponent[a] - largest);
        sum += exponent[a];
        mean += exponent[a] * reference_cn[start + a];
    }
    mean /= sum;
    for (int a = 0; a < count; ++a) {
        w[table::max_reference * i + a] = exponent[a] / sum;
        dw[table::max_reference * i + a] =
            8 * (reference_cn[start + a] - mean) * w[table::max_reference * i + a];
    }
}

struct OutputState {
    double value[14] = {};
}; // energy, force, virial, dE/dCN
__device__ void radial(OutputState& s, float3 d, double derivative_over_r, bool self) {
    double x[3] = {d.x, d.y, d.z};
    for (int k = 0; k < 3; ++k) {
        // A periodic self-image contributes strain but no position derivative.
        if (!self)
            s.value[1 + k] += derivative_over_r * x[k];
        for (int l = 0; l < 3; ++l)
            s.value[4 + 3 * k + l] -= .5 * derivative_over_r * x[k] * x[l];
    }
}
struct Dispersion : BaseOp {
    static const char* name() { return "d3.dispersion"; }
    using State = OutputState;
    Parameters           parameters;
    const int *          elements, *offsets;
    const double *       c6, *r2r4, *w, *dw;
    double*              adjoint;
    float *              energy, *force, *virial;
    int*                 status;
    lapis::WorkspaceSpec plan(int) const { return {0, threads * 14 * sizeof(double), 16}; }
    __device__ void      prepare(int, State&, void*) {}
    template<class Tile>
    __device__ void consume(int i, const Tile& tile, State& s, void*) {
        int e = threadIdx.x;
        if (e >= tile.size())
            return;
        int    j = tile.sender(e), zi = elements[i], zj = elements[j];
        double r2 = double(tile.squared_distance(e)) / (bohr * bohr);
        if (r2 == 0) {
            atomicCAS(status, 0, 1);
            return;
        }
        double coefficient = 0, derivative = 0;
        for (int a = offsets[zi]; a < offsets[zi + 1]; ++a)
            for (int b = offsets[zj]; b < offsets[zj + 1]; ++b) {
                int    hi = max(a, b), lo = min(a, b);
                double c = c6[hi * (hi + 1) / 2 + lo];
                coefficient += w[table::max_reference * i + a - offsets[zi]]
                               * w[table::max_reference * j + b - offsets[zj]] * c;
                derivative += dw[table::max_reference * i + a - offsets[zi]]
                              * w[table::max_reference * j + b - offsets[zj]] * c;
            }
        double ratio   = 3 * r2r4[zi] * r2r4[zj];
        double damping = parameters.a1 * sqrt(ratio) + parameters.a2;
        double d2 = damping * damping, d6 = d2 * d2 * d2;
        double r6 = r2 * r2 * r2;
        double t6 = 1 / (r6 + d6), t8 = 1 / (r6 * r2 + d6 * d2);
        double factor = -hartree * (parameters.s6 * t6 + parameters.s8 * ratio * t8);
        s.value[0] += .5 * coefficient * factor;
        // Reverse directed edges supply the other half, including self-images.
        s.value[13] += derivative * factor;
        double gradient =
            hartree * coefficient / (bohr * bohr)
            * (6 * parameters.s6 * r2 * r2 * t6 * t6 + 8 * parameters.s8 * ratio * r6 * t8 * t8);
        radial(s, tile.displacement(e), gradient, i == j);
    }
    __device__ void finalize(int i, State& s, void* scratch) {
        reduce(s.value, scratch);
        if (threadIdx.x == 0) {
            energy[i] = float(s.value[0]);
            for (int k = 0; k < 3; ++k)
                force[3 * i + k] = float(s.value[1 + k]);
            for (int k = 0; k < 9; ++k)
                virial[9 * i + k] = float(s.value[4 + k]);
            adjoint[i] = s.value[13];
        }
    }
};
struct CoordinationBackward : BaseOp {
    static const char* name() { return "d3.coordination_backward"; }
    using State = OutputState;
    const int*           elements;
    const double *       rcov, *adjoint;
    float *              force, *virial;
    int*                 status;
    lapis::WorkspaceSpec plan(int) const { return {0, threads * 14 * sizeof(double), 16}; }
    __device__ void      prepare(int, State&, void*) {}
    template<class Tile>
    __device__ void consume(int i, const Tile& tile, State& s, void*) {
        int e = threadIdx.x;
        if (e >= tile.size())
            return;
        int    j = tile.sender(e);
        double r = sqrt(double(tile.squared_distance(e)));
        if (r == 0) {
            atomicCAS(status, 0, 1);
            return;
        }
        double radius = bohr * (rcov[elements[i]] + rcov[elements[j]]);
        double count  = 1 / (1 + exp(-16 * (radius / r - 1)));
        double gradient =
            -(adjoint[i] + adjoint[j]) * 16 * radius * count * (1 - count) / (r * r * r);
        radial(s, tile.displacement(e), gradient, i == j);
    }
    __device__ void finalize(int i, State& s, void* scratch) {
        reduce(s.value, scratch);
        if (threadIdx.x == 0) {
            for (int k = 0; k < 3; ++k)
                force[3 * i + k] += float(s.value[1 + k]);
            for (int k = 0; k < 9; ++k)
                virial[9 * i + k] += float(s.value[4 + k]);
        }
    }
};

template<class T, size_t N>
const T* upload(lapis::Context& context, const char* name, const T (&values)[N]) {
    auto out = context.array<T>(name, N);
    lapis::check(
        cudaMemcpyAsync(out, values, sizeof(values), cudaMemcpyHostToDevice, context.stream));
    return out;
}
} // namespace

Calculator::Calculator(Parameters parameters, Cutoffs cutoffs, lapis::Options options)
    : lapis::Calculator(std::max(cutoffs.coordination, cutoffs.dispersion), options),
      parameters_(parameters), cutoffs_(cutoffs) {
    for (double p : {parameters.s6, parameters.s8, parameters.a1, parameters.a2})
        if (!std::isfinite(p) || p < 0)
            throw lapis::Error("Invalid D3(BJ) damping parameters");
    if (parameters.a1 == 0 && parameters.a2 == 0)
        throw lapis::Error("D3(BJ) requires a positive damping radius");
    for (float c : {cutoffs.coordination, cutoffs.dispersion})
        if (!std::isfinite(c) || c <= 0 || !std::isfinite(c * c))
            throw lapis::Error("Invalid D3 cutoff");
    offsets_      = upload(context_, "d3.table.offsets", table::offsets);
    reference_cn_ = upload(context_, "d3.table.reference_cn", table::reference_cn);
    c6_           = upload(context_, "d3.table.c6", table::c6);
    rcov_         = upload(context_, "d3.table.rcov", table::rcov);
    r2r4_         = upload(context_, "d3.table.r2r4", table::r2r4);
    lapis::check(cudaStreamSynchronize(context_.stream));
}

lapis::Result Calculator::evaluate() {
    for (int z : context_.original_elements)
        if (z > table::max_element)
            throw lapis::Error("D3 parameters support atomic numbers 1 through 94");
    size_t n       = context_.geometry.n;
    auto   cn      = context_.array<double>("d3.cn", n);
    auto   w       = context_.array<double>("d3.weights", table::max_reference * n);
    auto   dw      = context_.array<double>("d3.weight_derivatives", table::max_reference * n);
    auto   adjoint = context_.array<double>("d3.adjoint", n);
    auto   e       = context_.array<float>("d3.e", n);
    auto   f       = context_.array<float>("d3.f", 3 * n);
    auto   v       = context_.array<float>("d3.v", 9 * n);
    auto   z       = context_.geometry.elements;
    auto   status  = context_.geometry.status;
    lapis::execute(context_, lapis::Traversal{cutoffs_.coordination},
                   Coordination{{}, z, rcov_, cn, status});
    if (n)
        weights<<<(n + 127) / 128, 128, 0, context_.stream>>>(int(n), z, offsets_, reference_cn_,
                                                              cn, w, dw);
    lapis::check(cudaGetLastError());
    lapis::execute(
        context_, lapis::Traversal{cutoffs_.dispersion},
        Dispersion{{}, parameters_, z, offsets_, c6_, r2r4_, w, dw, adjoint, e, f, v, status});
    lapis::execute(context_, lapis::Traversal{cutoffs_.coordination},
                   CoordinationBackward{{}, z, rcov_, adjoint, f, v, status});
    return context_.finish(e, f, v);
}
} // namespace d3
