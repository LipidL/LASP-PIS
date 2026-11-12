// Minimal CUDA benchmark runner for one model and one structure per process.
//
// The executable is intentionally small: it reads the canonical binary input,
// loads the requested model, performs warmup calls, records wall and device
// time, and emits a self-describing JSON record.  Fixture generation, reference
// comparison, and baseline selection live in the Python driver.
#include <lapis/runtime.hpp>

#if defined(LAPIS_BENCH_HPNN) && defined(LAPIS_BENCH_MACE)
#error "Build one benchmark executable per model library"
#endif
#if !defined(LAPIS_BENCH_HPNN) && !defined(LAPIS_BENCH_MACE)
#error "Define LAPIS_BENCH_HPNN or LAPIS_BENCH_MACE"
#endif
#ifdef LAPIS_BENCH_HPNN
#include <hpnn/hpnn.hpp>
#endif
#ifdef LAPIS_BENCH_MACE
#include <mace/mh1.hpp>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string engine, model, input, backend = "auto", output, head;
    int         intervals = 0, warmup = 5, samples = 20;
};

struct Input {
    uint64_t              n = 0;
    std::array<double, 9> cell{};
    std::vector<int>      elements;
    std::vector<float>    positions;
};

void usage(std::ostream& out) {
    out << "Usage: lapis_bench_<engine> --engine hpnn|mace --model FILE "
           "--input FILE.bin [--backend auto|brute|cell] [--intervals N] "
           "[--head NAME] [--warmup N] [--samples N] [--output FILE]\n";
}

int integer(const std::string& text, const char* name) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error(std::string("invalid ") + name + ": " + text);
    unsigned long long value = std::stoull(text);
    if (value > unsigned(INT_MAX))
        throw std::runtime_error(std::string(name) + " is too large");
    return int(value);
}

Args parse(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string flag = argv[i];
        if (flag == "--help" || flag == "-h")
            continue;
        if (i + 1 == argc)
            throw std::runtime_error("missing value for " + flag);
        std::string value = argv[++i];
        if (flag == "--engine")
            args.engine = value;
        else if (flag == "--model")
            args.model = value;
        else if (flag == "--input")
            args.input = value;
        else if (flag == "--backend")
            args.backend = value;
        else if (flag == "--output")
            args.output = value;
        else if (flag == "--head")
            args.head = value;
        else if (flag == "--intervals")
            args.intervals = integer(value, "intervals");
        else if (flag == "--warmup")
            args.warmup = integer(value, "warmup");
        else if (flag == "--samples")
            args.samples = integer(value, "samples");
        else
            throw std::runtime_error("unknown option: " + flag);
    }
    if (args.engine != "hpnn" && args.engine != "mace")
        throw std::runtime_error("--engine must be hpnn or mace");
    if (args.model.empty() || args.input.empty())
        throw std::runtime_error("--model and --input are required");
    if (args.backend != "auto" && args.backend != "brute" && args.backend != "cell")
        throw std::runtime_error("--backend must be auto, brute, or cell");
    if (args.samples < 1)
        throw std::runtime_error("--samples must be at least one");
    return args;
}

void read_exact(std::ifstream& in, void* data, size_t bytes, const char* what) {
    in.read(static_cast<char*>(data), std::streamsize(bytes));
    if (!in)
        throw std::runtime_error(std::string("short or unreadable ") + what);
}

Input read_input(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot open input: " + path);
    char     magic[8];
    uint32_t version = 0, reserved = 0;
    uint64_t n = 0;
    read_exact(in, magic, sizeof(magic), "magic");
    read_exact(in, &version, sizeof(version), "version");
    read_exact(in, &reserved, sizeof(reserved), "reserved");
    read_exact(in, &n, sizeof(n), "atom count");
    if (std::string(magic, sizeof(magic)) != "LAPISB01")
        throw std::runtime_error("input is not a LAPISB01 structure");
    if (version != 1)
        throw std::runtime_error("unsupported structure version");
    if (reserved != 0)
        throw std::runtime_error("unexpected reserved header value");
    if (n == 0 || n > uint64_t(INT_MAX))
        throw std::runtime_error("invalid atom count");
    if (n > (SIZE_MAX / 3) / sizeof(float) || n > SIZE_MAX / sizeof(int))
        throw std::runtime_error("atom count overflows host buffers");
    Input input;
    input.n = n;
    read_exact(in, input.cell.data(), input.cell.size() * sizeof(double), "cell");
    input.elements.resize(size_t(n));
    read_exact(in, input.elements.data(), input.elements.size() * sizeof(int), "elements");
    input.positions.resize(size_t(3 * n));
    read_exact(in, input.positions.data(), input.positions.size() * sizeof(float), "positions");
    for (double value : input.cell)
        if (!std::isfinite(value))
            throw std::runtime_error("input cell contains a non-finite value");
    for (float value : input.positions)
        if (!std::isfinite(value))
            throw std::runtime_error("input positions contain a non-finite value");
    return input;
}

lapis::NeighborBackend backend(const std::string& name) {
    if (name == "brute")
        return lapis::NeighborBackend::BruteForce;
    if (name == "cell")
        return lapis::NeighborBackend::CellList;
    return lapis::NeighborBackend::Auto;
}

const char* backend_name(lapis::NeighborBackend value) {
    return value == lapis::NeighborBackend::CellList ? "cell" : "brute";
}

double median(std::vector<double> values) {
    if (values.empty())
        return 0;
    std::sort(values.begin(), values.end());
    size_t n = values.size();
    return n % 2 ? values[n / 2] : 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

double percentile(std::vector<double> values, double fraction) {
    if (values.empty())
        return 0;
    std::sort(values.begin(), values.end());
    if (fraction <= 0)
        return values.front();
    if (fraction >= 1)
        return values.back();
    size_t index = size_t(std::ceil(fraction * double(values.size()))) - 1;
    return values[std::min(index, values.size() - 1)];
}

std::string escape(const std::string& value) {
    std::string result;
    result.reserve(value.size());
    for (char c : value) {
        switch (c) {
        case '\\':
            result += "\\\\";
            break;
        case '"':
            result += "\\\"";
            break;
        case '\n':
            result += "\\n";
            break;
        case '\r':
            result += "\\r";
            break;
        case '\t':
            result += "\\t";
            break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                char buffer[7];
                std::snprintf(buffer, sizeof(buffer), "\\u%04x", static_cast<unsigned char>(c));
                result += buffer;
            } else {
                result += c;
            }
        }
    }
    return result;
}

std::string version_text(int version) {
    int         major  = version / 1000;
    int         minor  = (version % 1000) / 10;
    int         patch  = version % 10;
    std::string result = std::to_string(major) + "." + std::to_string(minor);
    if (patch)
        result += "." + std::to_string(patch);
    return result;
}

void write_device(std::ostream& out) {
    int device = 0;
    lapis::check(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    lapis::check(cudaGetDeviceProperties(&prop, device));
    int driver = 0, runtime = 0;
    lapis::check(cudaDriverGetVersion(&driver));
    lapis::check(cudaRuntimeGetVersion(&runtime));
    out << "\"device\":{\"name\":\"" << escape(prop.name) << "\""
        << ",\"compute_capability\":\"" << prop.major << "." << prop.minor << "\""
        << ",\"memory_bytes\":" << prop.totalGlobalMem << ",\"clock_khz\":" << prop.clockRate
        << ",\"memory_clock_khz\":" << prop.memoryClockRate
        << ",\"sms\":" << prop.multiProcessorCount << ",\"driver_cuda\":" << driver
        << ",\"driver_version\":\"" << version_text(driver) << "\""
        << ",\"runtime_cuda\":" << runtime << ",\"runtime_version\":\"" << version_text(runtime)
        << "\"}";
}

void write_summary(std::ostream& out, const std::vector<double>& wall) {
    double mean = std::accumulate(wall.begin(), wall.end(), 0.0) / double(wall.size());
    out << "\"summary\":{\"median_ms\":" << median(wall) << ",\"p95_ms\":" << percentile(wall, 0.95)
        << ",\"min_ms\":" << percentile(wall, 0.0) << ",\"max_ms\":" << percentile(wall, 1.0)
        << ",\"mean_ms\":" << mean << "}";
}

void run(const Args& args, std::ostream& out) {
    Input              input = read_input(args.input);
    std::vector<float> xyz   = input.positions;
    lapis::Structure   structure;
    structure.size      = int(input.n);
    structure.positions = xyz.data();
    structure.elements  = input.elements.data();
    for (size_t k = 0; k < 9; ++k)
        structure.cell[k] = float(input.cell[k]);
    structure.pbc = {true, true, true};

    lapis::Options options;
    options.backend = backend(args.backend);

    std::unique_ptr<lapis::Calculator> calculator;
    auto                               start = std::chrono::steady_clock::now();
#ifdef LAPIS_BENCH_HPNN
    if (args.engine == "hpnn")
        calculator = std::make_unique<hpnn::HPNN>(args.model, options, args.intervals);
#endif
#ifdef LAPIS_BENCH_MACE
    if (args.engine == "mace") {
        if (args.head.empty())
            calculator = std::make_unique<mace::MH1>(args.model, options);
        else
            calculator = std::make_unique<mace::MH1>(args.model, options, args.head);
    }
#endif
    if (!calculator)
        throw std::runtime_error("this executable was not built with engine " + args.engine);
    double init_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

    lapis::Result result;
    for (int i = 0; i < args.warmup; ++i)
        result = calculator->calculate(structure);

    std::vector<double> wall, device;
    wall.reserve(size_t(args.samples));
    device.reserve(size_t(args.samples));
    for (int i = 0; i < args.samples; ++i) {
        auto begin = std::chrono::steady_clock::now();
        result     = calculator->calculate(structure);
        auto end   = std::chrono::steady_clock::now();
        wall.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
        device.push_back(double(calculator->diagnostics().device_ms));
    }
    lapis::check(cudaDeviceSynchronize());

    double                force_max = 0, force_l2 = 0;
    std::array<double, 3> force_sum{};
    double                energies_sum = 0;
    for (size_t i = 0; i < result.forces.size(); i += 3) {
        for (int c = 0; c < 3; ++c) {
            double value = double(result.forces[i + size_t(c)]);
            force_max    = std::max(force_max, std::abs(value));
            force_l2 += value * value;
            force_sum[size_t(c)] += value;
        }
    }
    for (float value : result.energies)
        energies_sum += double(value);

    const auto& diagnostics = calculator->diagnostics();
    out << std::setprecision(17);
    out << "{\"schema\":1,\"status\":\"ok\""
        << ",\"engine\":\"" << escape(args.engine) << "\""
        << ",\"model\":\"" << escape(args.model) << "\""
        << ",\"input\":\"" << escape(args.input) << "\""
        << ",\"atoms\":" << input.n << ",\"init_ms\":" << init_ms << ",\"options\":{\"backend\":\""
        << escape(args.backend) << "\",\"intervals\":" << args.intervals
        << ",\"warmup\":" << args.warmup << ",\"samples\":" << args.samples << "},";
    write_device(out);
    out << ",\"samples_ms\":[";
    for (size_t i = 0; i < wall.size(); ++i)
        out << (i ? "," : "") << wall[i];
    out << "],\"device_ms\":[";
    for (size_t i = 0; i < device.size(); ++i)
        out << (i ? "," : "") << device[i];
    out << "],";
    write_summary(out, wall);
    out << ",\"output\":{\"energy\":" << result.energy << ",\"energies_sum\":" << energies_sum
        << ",\"virial\":[";
    for (size_t i = 0; i < result.virial.size(); ++i)
        out << (i ? "," : "") << result.virial[i];
    out << "],\"force_max_abs\":" << force_max << ",\"force_l2\":" << std::sqrt(force_l2)
        << ",\"force_sum\":[" << force_sum[0] << "," << force_sum[1] << "," << force_sum[2] << "]}";
    out << ",\"diagnostics\":{\"selected\":\"" << backend_name(diagnostics.selected)
        << "\",\"grid\":[" << diagnostics.grid[0] << "," << diagnostics.grid[1] << ","
        << diagnostics.grid[2] << "]"
        << ",\"max_occupancy\":" << diagnostics.max_occupancy
        << ",\"occupied_bins\":" << diagnostics.occupied_bins
        << ",\"mean_occupancy\":" << diagnostics.mean_occupancy
        << ",\"grid_candidates\":" << diagnostics.grid_candidates
        << ",\"brute_candidates\":" << diagnostics.brute_candidates
        << ",\"allocated_bytes\":" << diagnostics.allocated_bytes
        << ",\"peak_bytes\":" << diagnostics.peak_bytes << ",\"reason\":\""
        << escape(diagnostics.reason) << "\"}";
    out << ",\"resources\":[";
    const auto& resources = calculator->resources();
    for (size_t i = 0; i < resources.size(); ++i) {
        const auto& r = resources[i];
        out << (i ? "," : "") << "{\"name\":\"" << escape(r.name)
            << "\",\"registers\":" << r.registers << ",\"block_threads\":" << r.block_threads
            << ",\"active_blocks_per_sm\":" << r.active_blocks_per_sm
            << ",\"static_shared\":" << r.static_shared
            << ",\"dynamic_shared\":" << r.dynamic_shared << "}";
    }
    out << "]}\n";
    out.flush();
    if (!out)
        throw std::runtime_error("failed to write benchmark output");
}

} // namespace

int main(int argc, char** argv) {
    Args args;
    try {
        args = parse(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "lapis_bench: " << error.what() << "\n";
        usage(std::cerr);
        return 2;
    }
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
            usage(std::cout);
            return 0;
        }

    std::ofstream file;
    std::ostream* out = &std::cout;
    if (!args.output.empty()) {
        file.open(args.output);
        if (!file) {
            std::cerr << "lapis_bench: cannot open output: " << args.output << "\n";
            return 2;
        }
        out = &file;
    }
    try {
        run(args, *out);
        return 0;
    } catch (const std::exception& error) {
        *out << "{\"schema\":1,\"status\":\"error\",\"error\":\"" << escape(error.what())
             << "\"}\n";
        out->flush();
        std::cerr << "lapis_bench: " << error.what() << "\n";
        return 1;
    }
}
