#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>

#include "d3/d3.hpp"

struct Fixture {
    const char*           name;
    std::vector<int>      z;
    std::vector<float>    xyz;
    std::array<float, 9>  cell;
    d3::Cutoffs           cutoffs;
    double                energy;
    std::vector<double>   forces;
    std::array<double, 9> virial;
    lapis::Structure      structure() const { return {int(z.size()), xyz.data(), z.data(), cell}; }
};
#include "reference.inc"
void require(bool condition, const char* message) {
    if (!condition)
        throw lapis::Error(message);
}
void near(double actual, double expected, double atol, const char* message) {
    if (std::abs(actual - expected) > atol + 1e-5 * std::abs(expected)) {
        std::cerr << message << ": actual=" << actual << " expected=" << expected << '\n';
        throw lapis::Error(message);
    }
}
void compare(const lapis::Result& a, const lapis::Result& b, double tolerance) {
    near(a.energy, b.energy, tolerance, "energy changed");
    require(a.forces.size() == b.forces.size(), "force size changed");
    for (size_t k = 0; k < a.forces.size(); ++k)
        near(a.forces[k], b.forces[k], tolerance, "force changed");
    for (int k = 0; k < 9; ++k)
        near(a.virial[k], b.virial[k], tolerance, "virial changed");
}
template<class F>
void rejects(F operation) {
    bool rejected = false;
    try {
        operation();
    } catch (const lapis::Error&) {
        rejected = true;
    }
    require(rejected, "invalid D3 input accepted");
}
int main() {
    try {
        for (const auto& fixture : fixtures) {
            std::cout << fixture.name << '\n';
            lapis::Options opts;
            opts.backend = lapis::NeighborBackend::BruteForce;
            d3::Calculator calc({}, fixture.cutoffs, opts);
            auto           result = calc.calculate(fixture.structure());
            near(result.energy, fixture.energy, 2e-6, "reference energy");
            for (size_t k = 0; k < result.forces.size(); ++k)
                near(result.forces[k], fixture.forces[k], 2e-6, "reference force");
            for (int k = 0; k < 9; ++k)
                near(result.virial[k], fixture.virial[k], 3e-6, "reference virial");
            near(std::accumulate(result.energies.begin(), result.energies.end(), 0.), result.energy,
                 1e-10, "atomic energy sum");
            for (int k = 0; k < 3; ++k) {
                double sum = 0;
                for (size_t i = 0; i < fixture.z.size(); ++i)
                    sum += result.forces[3 * i + k];
                near(sum, 0, 2e-7, "momentum conservation");
            }
            auto capacity = calc.diagnostics().allocated_bytes;
            compare(result, calc.calculate(fixture.structure()), 1e-8);
            require(calc.diagnostics().allocated_bytes == capacity, "allocation not reused");
            d3::Calculator automatic({}, fixture.cutoffs);
            compare(result, automatic.calculate(fixture.structure()), 1e-7);
            if (fixture.cell[0]
                    >= std::max(fixture.cutoffs.coordination, fixture.cutoffs.dispersion)
                && fixture.cell[3] == 0 && fixture.cell[6] == 0) {
                opts.backend = lapis::NeighborBackend::CellList;
                d3::Calculator grid({}, fixture.cutoffs, opts);
                compare(result, grid.calculate(fixture.structure()), 1e-7);
            }
            auto moved = fixture;
            std::reverse(moved.z.begin(), moved.z.end());
            for (size_t i = 0; i < moved.z.size(); ++i)
                for (int k = 0; k < 3; ++k)
                    moved.xyz[3 * i + k] = fixture.xyz[3 * (moved.z.size() - 1 - i) + k]
                                           + (i % 2 ? 1 : -1) * fixture.cell[k];
            auto transformed = calc.calculate(moved.structure());
            near(result.energy, transformed.energy, 2e-7, "permutation/wrapping energy");
            for (size_t i = 0; i < moved.z.size(); ++i)
                for (int k = 0; k < 3; ++k)
                    near(result.forces[3 * i + k],
                         transformed.forces[3 * (moved.z.size() - 1 - i) + k], 5e-7,
                         "permutation/wrapping force");
            // Full coordinate and cell derivatives, including the CN chain rule.
            for (size_t k = 0; k < fixture.xyz.size(); ++k) {
                auto  plus = fixture, minus = fixture;
                float step = .002f;
                plus.xyz[k] += step;
                minus.xyz[k] -= step;
                double diff = (calc.calculate(plus.structure()).energy
                               - calc.calculate(minus.structure()).energy)
                              / (plus.xyz[k] - minus.xyz[k]);
                near(result.forces[k], -diff, 1e-5, "force finite difference");
            }
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b) {
                    auto  plus = fixture, minus = fixture;
                    float step = .0005f;
                    for (size_t i = 0; i < fixture.z.size(); ++i) {
                        plus.xyz[3 * i + b] += step * fixture.xyz[3 * i + a];
                        minus.xyz[3 * i + b] -= step * fixture.xyz[3 * i + a];
                    }
                    for (int row = 0; row < 3; ++row) {
                        plus.cell[3 * row + b] += step * fixture.cell[3 * row + a];
                        minus.cell[3 * row + b] -= step * fixture.cell[3 * row + a];
                    }
                    double diff = (calc.calculate(plus.structure()).energy
                                   - calc.calculate(minus.structure()).energy)
                                  / (2 * step);
                    near(result.virial[3 * a + b], -diff, 3e-5, "virial finite difference");
                }
        }
        auto           fixture = fixtures.front();
        d3::Calculator calc;
        auto           empty = fixture.structure();
        empty.size           = 0;
        auto zero            = calc.calculate(empty);
        require(zero.energy == 0 && zero.forces.empty() && zero.energies.empty(), "empty result");
        d3::Calculator isolated({}, {.1f, .2f});
        require(isolated.calculate(fixture.structure()).energy == 0, "empty neighborhood");
        d3::Calculator disabled({0, 0, .4289, 4.4407}, fixture.cutoffs);
        zero = disabled.calculate(fixture.structure());
        require(zero.energy == 0, "zero scales energy");
        for (float f : zero.forces)
            require(f == 0, "zero scales force");
        rejects([] { d3::Calculator bad({-1, 1, 1, 1}); });
        rejects([] { d3::Calculator bad({1, 1, 0, 0}); });
        rejects([] { d3::Calculator bad({}, {0, 1}); });
        rejects([] { d3::Calculator bad({}, {1, std::numeric_limits<float>::infinity()}); });
        rejects([] { d3::Calculator bad({1, 1, std::numeric_limits<double>::quiet_NaN(), 1}); });
        fixture.z[0] = 95;
        rejects([&] { calc.calculate(fixture.structure()); });
        fixture = fixtures.front();
        std::copy(fixture.xyz.begin(), fixture.xyz.begin() + 3, fixture.xyz.begin() + 3);
        rejects([&] { calc.calculate(fixture.structure()); });
        // A device error must not poison the next structure.
        calc.calculate(fixtures.front().structure());
        std::cout << "D3 references, derivatives, invariance, reuse and errors passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
