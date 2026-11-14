#include <algorithm>
#include <climits>
#include <cmath>
#include <cub/device/device_scan.cuh>

#include "lapis/runtime.hpp"

namespace lapis {
#ifdef LAPIS_EXPLICIT_REFERENCE
void build_explicit_reference(Context&);
#endif
bool Buffer::reserve(size_t bytes) {
    if (bytes <= capacity)
        return false;
    void* fresh = nullptr;
    check(cudaMalloc(&fresh, bytes));
    if (data)
        cudaFree(data);
    data     = fresh;
    capacity = bytes;
    return true;
}
Context::Context() {
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));
    check(cudaEventCreateWithFlags(&input_copied_, cudaEventDisableTiming));
}
Context::~Context() {
    cudaStreamSynchronize(stream);
    if (pinned_)
        cudaFreeHost(pinned_);
    if (input_pinned_)
        cudaFreeHost(input_pinned_);
    cudaEventDestroy(input_copied_);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
}
KernelResources Context::kernel_resources(const void* kernel, dim3 block, size_t shared) {
    int device;
    check(cudaGetDevice(&device));
    KernelKey key{device, kernel, shared, block.x, block.y, block.z};
    auto      cached = kernel_resources_.find(key);
    if (cached != kernel_resources_.end())
        return cached->second;
    cudaFuncAttributes attr{};
    check(cudaFuncGetAttributes(&attr, kernel));
    cudaDeviceProp props{};
    check(cudaGetDeviceProperties(&props, device));
    unsigned threads = block.x * block.y * block.z;
    if (block.x > unsigned(props.maxThreadsDim[0]) || block.y > unsigned(props.maxThreadsDim[1])
        || block.z > unsigned(props.maxThreadsDim[2]))
        throw Error("PairOp block dimension exceeds device limit");
    if (props.major < 7 || threads > unsigned(attr.maxThreadsPerBlock)
        || shared + attr.sharedSizeBytes > props.sharedMemPerBlockOptin
        || size_t(attr.numRegs) * threads > size_t(props.regsPerBlock))
        throw Error("PairOp exceeds device kernel resource limits");
    // The attribute belongs to the kernel, not this Context. Always opt in to
    // the device limit so another calculator cannot lower a cached launch limit.
    if (shared + attr.sharedSizeBytes > props.sharedMemPerBlock)
        check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   int(props.sharedMemPerBlockOptin - attr.sharedSizeBytes)));
    int active;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, threads, shared));
    if (active < 1)
        throw Error("PairOp has no resident block capacity");
    KernelResources result{"", attr.numRegs, int(threads), active, attr.sharedSizeBytes, shared};
    kernel_resources_.emplace(key, result);
    return result;
}
void* Context::reserve(const std::string& name, size_t bytes) {
    auto&  b             = buffers_[name];
    size_t previous      = b.capacity;
    size_t during_growth = diagnostics.allocated_bytes + (bytes > previous ? bytes : 0);
    if (b.reserve(bytes))
        ++allocation_generation_;
    diagnostics.allocated_bytes += b.capacity - previous;
    diagnostics.peak_bytes =
        std::max({diagnostics.peak_bytes, diagnostics.allocated_bytes, during_growth});
    return b.data;
}
__global__ void histogram(const int* bins, int* counts, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        atomicAdd(counts + bins[i], 1);
}
__global__ void identity_order(int n, int* io, int* oi) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        io[i] = oi[i] = i;
}
__global__ void scatter(const float* x, const int* z, const int* b, const int* w, int n,
                        const int* offsets, int* cursor, float* xo, int* zo, int* bo, int* wo,
                        int* io, int* oi) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    int j = offsets[b[i]] + atomicAdd(cursor + b[i], 1);
    io[j] = i;
    oi[i] = j;
    zo[j] = z[i];
    bo[j] = b[i];
    for (int k = 0; k < 3; ++k) {
        xo[3 * j + k] = x[3 * i + k];
        wo[3 * j + k] = w[3 * i + k];
    }
}
__global__ void restore(const float* e, const float* f, const float* v, const int* map, int n,
                        const int* status, float* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0)
        reinterpret_cast<int*>(out)[size_t(n) * 13] = *status;
    if (i >= n)
        return;
    int j  = map[i];
    out[j] = e[i];
    for (int k = 0; k < 3; ++k)
        out[n + 3 * j + k] = f[3 * i + k];
    for (int k = 0; k < 9; ++k)
        out[4 * n + 9 * j + k] = v[9 * i + k];
}
void Context::prepare(const Structure& s, float cutoff, const Options& opts) {
    // Calculator calls finish before reusing staging. Direct Context clients
    // may prepare again sooner; wait only for the preceding input transfer.
    if (input_pending_) {
        check(cudaEventSynchronize(input_copied_));
        input_pending_ = false;
    }
    prepared_cutoff_ = 0;
    resources.clear();
    if (s.size < 0 || s.size > INT_MAX / 16 || !std::isfinite(cutoff) || cutoff <= 0
        || !std::isfinite(cutoff * cutoff))
        throw Error("Invalid atom count or cutoff");
    if (!s.pbc[0] || !s.pbc[1] || !s.pbc[2])
        throw Error("v1 requires three periodic axes");
    if (s.size && (!s.positions || !s.elements))
        throw Error("Null structure data");
    if (opts.grid_to_atom_budget < 1 || opts.grid_byte_budget < 16)
        throw Error("Invalid grid budget");
    double h[9];
    for (int k = 0; k < 9; ++k) {
        h[k] = s.cell[k];
        if (!std::isfinite(h[k]))
            throw Error("Nonfinite cell");
    }
    double det   = h[0] * (h[4] * h[8] - h[5] * h[7]) - h[1] * (h[3] * h[8] - h[5] * h[6])
                   + h[2] * (h[3] * h[7] - h[4] * h[6]);
    double scale = 0;
    for (double a : h)
        scale = std::max(scale, std::abs(a));
    if (!std::isfinite(det) || scale == 0 || std::abs(det) < 1e-10 * scale * scale * scale)
        throw Error("Singular or ill-conditioned cell");
    double inv[9] = {(h[4] * h[8] - h[5] * h[7]) / det, (h[2] * h[7] - h[1] * h[8]) / det,
                     (h[1] * h[5] - h[2] * h[4]) / det, (h[5] * h[6] - h[3] * h[8]) / det,
                     (h[0] * h[8] - h[2] * h[6]) / det, (h[2] * h[3] - h[0] * h[5]) / det,
                     (h[3] * h[7] - h[4] * h[6]) / det, (h[1] * h[6] - h[0] * h[7]) / det,
                     (h[0] * h[4] - h[1] * h[3]) / det};
    int    dims[3], images[3];
    std::array<double, 3> heights;
    bool                  grid_ok = true;
    for (int k = 0; k < 3; ++k) {
        double height =
            1 / std::sqrt(inv[k] * inv[k] + inv[3 + k] * inv[3 + k] + inv[6 + k] * inv[6 + k]);
        heights[k]   = height;
        double ratio = height / cutoff;
        grid_ok &= ratio >= 1;
        dims[k]   = int(std::max(1.0, std::min(double(INT_MAX / 4), std::floor(ratio))));
        double im = std::ceil(cutoff / height);
        if (im > 1000000)
            throw Error("Excessive periodic image range");
        images[k] = std::max(1, int(im));
    }
    if (double(2 * images[0] + 1) * (2 * images[1] + 1) * (2 * images[2] + 1) > INT_MAX)
        throw Error("Periodic image tuples exceed v1 int32 capacity");
    if (opts.backend == NeighborBackend::CellList && !grid_ok)
        throw Error("CellList requires all reciprocal plane heights >= cutoff");
    size_t cap = std::max(size_t(1),
                          std::min({opts.grid_byte_budget / 16,
                                    size_t(std::max(s.size, 1)) * size_t(opts.grid_to_atom_budget),
                                    size_t(INT_MAX / 4)}));
    auto   bins_product = [&]() { return double(dims[0]) * dims[1] * dims[2]; };
    bool   coarsened    = false;
    while (bins_product() > double(cap)) {
        int axis   = int(std::max_element(dims, dims + 3) - dims);
        dims[axis] = (dims[axis] + 1) / 2;
        coarsened  = true;
    }
    int    nb = int(bins_product()), n = s.size;
    size_t input_bytes = std::max(size_t(1), size_t(n) * 8) * 4;
    if (input_bytes > input_capacity_) {
        void* fresh;
        check(cudaMallocHost(&fresh, input_bytes));
        if (input_pinned_)
            cudaFreeHost(input_pinned_);
        input_pinned_   = fresh;
        input_capacity_ = input_bytes;
    }
    auto* xyz  = static_cast<float*>(input_pinned_);
    auto* z    = reinterpret_cast<int*>(xyz + size_t(n) * 3);
    auto* bins = z + n;
    auto* wind = bins + n;
    occupancy_.assign(nb, 0);
    auto& occ = occupancy_;
    for (int i = 0; i < n; ++i) {
        if (s.elements[i] < 1 || s.elements[i] > 118)
            throw Error("Invalid atomic number");
        z[i] = s.elements[i];
        double f[3];
        int    b[3];
        for (int k = 0; k < 3; ++k) {
            double v = 0;
            for (int d = 0; d < 3; ++d) {
                float p = s.positions[3 * i + d];
                if (!std::isfinite(p))
                    throw Error("Nonfinite coordinate");
                v += p * inv[3 * d + k];
            }
            double w = std::floor(v);
            if (std::abs(w) > INT_MAX / 4)
                throw Error("Coordinate winding exceeds int32 range");
            wind[3 * i + k] = int(w);
            f[k]            = v - w;
            b[k]            = std::min(dims[k] - 1, int(f[k] * dims[k]));
        }
        for (int k = 0; k < 3; ++k)
            xyz[3 * i + k] = float(f[0] * h[k] + f[1] * h[3 + k] + f[2] * h[6 + k]);
        bins[i] = (b[0] * dims[1] + b[1]) * dims[2] + b[2];
        ++occ[bins[i]];
    }
    double work = 0;
    for (int b = 0; b < nb; ++b)
        if (occ[b]) {
            int z = b % dims[2], y = (b / dims[2]) % dims[1], x = b / (dims[1] * dims[2]);
            for (int dx = -1; dx <= 1; ++dx)
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dz = -1; dz <= 1; ++dz) {
                        int c = (((x + dx + dims[0]) % dims[0]) * dims[1]
                                 + (y + dy + dims[1]) % dims[1])
                                    * dims[2]
                                + (z + dz + dims[2]) % dims[2];
                        work += double(occ[b]) * occ[c];
                    }
        }
    double brute = double(n) * n;
    for (int k = 0; k < 3; ++k)
        brute *= 2 * images[k] + 1;
    auto        selected = opts.backend;
    std::string reason   = "explicit backend";
    if (selected == NeighborBackend::Auto) {
        selected =
            grid_ok && work < brute * .9 ? NeighborBackend::CellList : NeighborBackend::BruteForce;
        reason = !grid_ok ? "reciprocal plane height below cutoff"
                 : selected == NeighborBackend::CellList
                     ? (coarsened ? "coarsened grid has lower occupancy-estimated work"
                                  : "grid has lower occupancy-estimated work")
                     : "brute force has comparable or lower estimated work";
    }
    diagnostics.selected      = selected;
    diagnostics.reason        = reason;
    diagnostics.grid          = {dims[0], dims[1], dims[2]};
    diagnostics.max_occupancy = *std::max_element(occ.begin(), occ.end());
    diagnostics.occupied_bins =
        int(std::count_if(occ.begin(), occ.end(), [](int a) { return a > 0; }));
    diagnostics.mean_occupancy   = double(n) / nb;
    diagnostics.grid_candidates  = work;
    diagnostics.brute_candidates = brute;
    original_elements.clear();
    if (n)
        original_elements.assign(s.elements, s.elements + n);
    check(cudaEventRecord(start, stream));
    float* input    = static_cast<float*>(reserve("input.packed", input_bytes));
    int*   elements = reinterpret_cast<int*>(input + size_t(n) * 3);
    int*   ib       = elements + n;
    int*   iw       = ib + n;
    if (n) {
        check(cudaMemcpyAsync(input, input_pinned_, size_t(n) * 8 * 4, cudaMemcpyHostToDevice,
                              stream));
        // Mark pending before recording so error recovery also protects staging.
        input_pending_ = true;
        check(cudaEventRecord(input_copied_, stream));
    }
    float* sorted        = input;
    int*   sz            = elements;
    int*   sb            = ib;
    int*   sw            = iw;
    internal_to_original = array<int>("geometry.i2o", n);
    original_to_internal = array<int>("geometry.o2i", n);
    int* offsets         = nullptr;
    if (selected == NeighborBackend::BruteForce) {
        if (n)
            identity_order<<<(n + 255) / 256, 256, 0, stream>>>(n, internal_to_original,
                                                                original_to_internal);
    } else {
        sorted      = array<float>("geometry.xyz", 3 * size_t(n));
        sz          = array<int>("geometry.z", n);
        sb          = array<int>("geometry.bin", n);
        sw          = array<int>("geometry.winding", 3 * size_t(n));
        int* counts = array<int>("grid.count", nb + 1);
        offsets     = array<int>("grid.offset", nb + 1);
        int* cursor = array<int>("grid.cursor", nb);
        clear(counts, size_t(nb + 1) * 4);
        clear(cursor, size_t(nb) * 4);
        if (n)
            histogram<<<(n + 255) / 256, 256, 0, stream>>>(ib, counts, n);
        size_t scan_bytes = 0;
        check(cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, counts, offsets, nb + 1, stream));
        void* scan = reserve("grid.scan", scan_bytes);
        check(cub::DeviceScan::ExclusiveSum(scan, scan_bytes, counts, offsets, nb + 1, stream));
        if (n)
            scatter<<<(n + 255) / 256, 256, 0, stream>>>(input, elements, ib, iw, n, offsets,
                                                         cursor, sorted, sz, sb, sw,
                                                         internal_to_original,
                                                         original_to_internal);
    }
    check(cudaGetLastError());
    int* status = array<int>("status", 1);
    clear(status, 4);
    geometry          = {};
    geometry.n        = n;
    geometry.xyz      = sorted;
    geometry.elements = sz;
    geometry.bins     = sb;
    geometry.offsets  = offsets;
    geometry.winding  = sw;
    geometry.status   = status;
    geometry.backend  = selected;
    geometry.cutoff2  = cutoff * cutoff;
    std::copy(s.cell.begin(), s.cell.end(), geometry.h);
    std::copy(dims, dims + 3, geometry.grid);
    std::copy(images, images + 3, geometry.images);
    plane_heights_   = heights;
    prepared_cutoff_ = cutoff;
#ifdef LAPIS_EXPLICIT_REFERENCE
    build_explicit_reference(*this);
    diagnostics.reason += "; benchmark explicit connectivity";
#endif
}
DeviceGeometry Context::traversal_geometry(Traversal traversal) const {
    if (!std::isfinite(traversal.cutoff) || traversal.cutoff <= 0
        || traversal.cutoff > prepared_cutoff_ || prepared_cutoff_ <= 0)
        throw Error("Traversal cutoff must be finite, positive and <= prepared cutoff");
    auto view    = geometry;
    view.cutoff2 = traversal.cutoff * traversal.cutoff;
    if (view.backend == NeighborBackend::BruteForce)
        for (int k = 0; k < 3; ++k)
            // Use the same double-precision heights and formula as prepare().
            // r <= R guarantees that these bounds fit the prepared limits.
            view.images[k] = std::max(1, int(std::ceil(traversal.cutoff / plane_heights_[k])));
    return view;
}
void Context::synchronize() {
    check(cudaGetLastError());
    check(cudaStreamSynchronize(stream));
    int status = 0;
    check(cudaMemcpy(&status, geometry.status, 4, cudaMemcpyDeviceToHost));
    if (status)
        throw Error("Model detected invalid device state (code " + std::to_string(status) + ")");
}
Result Context::finish(float* e, float* f, float* v) {
    int    n       = geometry.n;
    size_t bytes   = (size_t(n) * 13 + 1) * 4;
    float* ordered = array<float>("output.ordered", size_t(n) * 13 + 1);
    restore<<<std::max(1, (n + 255) / 256), 256, 0, stream>>>(e, f, v, internal_to_original, n,
                                                              geometry.status, ordered);
    if (bytes > pinned_capacity_) {
        void* fresh;
        check(cudaMallocHost(&fresh, bytes));
        if (pinned_)
            cudaFreeHost(pinned_);
        pinned_          = fresh;
        pinned_capacity_ = bytes;
    }
    if (bytes)
        check(cudaMemcpyAsync(pinned_, ordered, bytes, cudaMemcpyDeviceToHost, stream));
    check(cudaEventRecord(stop, stream));
    check(cudaGetLastError());
    check(cudaStreamSynchronize(stream));
    input_pending_ = false;
    int status     = static_cast<const int*>(pinned_)[size_t(n) * 13];
    if (status)
        throw Error("Model detected invalid device state (code " + std::to_string(status) + ")");
    check(cudaEventElapsedTime(&diagnostics.device_ms, start, stop));
    Result r;
    r.energies.resize(n);
    r.forces.resize(size_t(n) * 3);
    auto p = static_cast<const float*>(pinned_);
    if (n) {
        std::copy(p, p + n, r.energies.begin());
        std::copy(p + n, p + 4 * n, r.forces.begin());
    }
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(p[i]))
            throw Error("Nonfinite model output");
        r.energy += p[i];
        for (int k = 0; k < 3; ++k)
            if (!std::isfinite(p[n + 3 * i + k]))
                throw Error("Nonfinite force");
    }
    for (int k = 0; k < 9; ++k) {
        double sum = 0;
        for (int i = 0; i < n; ++i)
            sum += p[4 * n + 9 * i + k];
        if (!std::isfinite(sum))
            throw Error("Nonfinite virial");
        r.virial[k] = float(sum);
    }
    return r;
}
Result Calculator::calculate(const Structure& s) {
    if (busy_.test_and_set(std::memory_order_acquire))
        throw Error("Calculator is single-flight: concurrent call rejected");
    struct Guard {
        std::atomic_flag& b;
        ~Guard() { b.clear(std::memory_order_release); }
    } guard{busy_};
    try {
        context_.prepare(s, cutoff_, options_);
        return evaluate();
    } catch (...) {
        cudaStreamSynchronize(context_.stream);
        throw;
    }
}
} // namespace lapis
