#include "physics.hpp"

#include <algorithm>
#include <chemio/structure.hpp>
#include <cmath>
#include <filesystem>
#include <limits>
#include <map>
#include <numeric>
#include <random>

namespace potential_test {
namespace {
constexpr Matrix box(float x, float y, float z) {
    return {x, 0, 0, 0, y, 0, 0, 0, z};
}
constexpr Matrix identity = box(1, 1, 1);

template<class Container>
Container transform(const Container& rows, const Matrix& m) {
    Container out = rows;
    for (size_t i = 0; i < rows.size(); i += 3)
        for (int j = 0; j < 3; ++j) {
            double value = 0;
            for (int k = 0; k < 3; ++k)
                value += double(rows[i + k]) * m[3 * k + j];
            out[i + j] = float(value);
        }
    return out;
}
Matrix transpose(const Matrix& a) {
    Matrix out{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            out[3 * i + j] = a[3 * j + i];
    return out;
}
Structure deform(Structure s, const Matrix& m) {
    s.positions = transform(s.positions, m);
    s.cell      = transform(s.cell, m);
    return s;
}
void near(double actual, double expected, double absolute, double relative = 2e-4) {
    EXPECT_TRUE(std::isfinite(actual));
    EXPECT_NEAR(actual, expected, absolute + relative * std::abs(expected));
}
void valid(const lapis::Result& r, size_t n) {
    ASSERT_EQ(r.energies.size(), n);
    ASSERT_EQ(r.forces.size(), 3 * n);
    EXPECT_TRUE(std::isfinite(r.energy));
    for (float v : r.energies)
        EXPECT_TRUE(std::isfinite(v));
    for (float v : r.forces)
        EXPECT_TRUE(std::isfinite(v));
    for (float v : r.virial)
        EXPECT_TRUE(std::isfinite(v));
}
} // namespace

double volume(const Matrix& h) {
    return std::abs(double(h[0]) * (double(h[4]) * h[8] - double(h[5]) * h[7])
                    - double(h[1]) * (double(h[3]) * h[8] - double(h[5]) * h[6])
                    + double(h[2]) * (double(h[3]) * h[7] - double(h[4]) * h[6]));
}

Structure supercell(const Structure& s, std::array<int, 3> repeats) {
    Structure out{s.name, {}, {}, s.cell};
    for (int a = 0; a < repeats[0]; ++a)
        for (int b = 0; b < repeats[1]; ++b)
            for (int c = 0; c < repeats[2]; ++c)
                for (size_t i = 0; i < s.elements.size(); ++i) {
                    out.elements.push_back(s.elements[i]);
                    for (int k = 0; k < 3; ++k)
                        out.positions.push_back(s.positions[3 * i + k] + a * s.cell[k]
                                                + b * s.cell[3 + k] + c * s.cell[6 + k]);
                }
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            out.cell[3 * i + j] *= repeats[i];
    return out;
}

std::vector<Structure> structures() {
    std::vector<Structure> out{
        {"Empty", {}, {}, box(14, 15, 16)},
        {"IsolatedO", {8}, {2.1f, 3.2f, 4.3f}, box(18, 19, 20)},
        {"Water", {8, 1, 1}, {2, 2, 2, 2.97f, 2.02f, 2.01f, 1.76f, 2.94f, 2.03f}, box(16, 17, 18)},
        {"Methane",
         {6, 1, 1, 1, 1},
         {3, 3, 3, 3.64f, 3.63f, 3.62f, 2.37f, 2.35f, 3.63f, 2.36f, 3.63f, 2.38f, 3.64f, 2.37f,
          2.36f},
         box(18, 19, 20)},
        // A distorted crystal gives nonzero forces and off-diagonal stresses.
        {"BCCFe", {26, 26}, {0.08f, 0.03f, 0.02f, 1.43f, 1.47f, 1.41f}, box(2.87f, 2.91f, 2.83f)},
        {"TriclinicTiO2",
         {22, 8, 8},
         {0.21f, 0.32f, 0.43f, 1.95f, 1.12f, 0.85f, 0.91f, 2.11f, 2.41f},
         {4.7f, 0, 0, 0.7f, 4.9f, 0, -0.4f, 0.6f, 5.1f}},
        {"PeriodicSelfImages",
         {14},
         {0.23f, 0.37f, 0.41f},
         {3.2f, 0, 0, 0.4f, 3.5f, 0, -0.2f, 0.3f, 3.8f}},
        {"SiSlab",
         {14, 14, 14, 14},
         {0.1f, 0.1f, 5, 1.45f, 1.43f, 6.36f, 2.79f, 2.75f, 5.03f, 4.1f, 4.12f, 6.35f},
         {5.43f, 0, 0, 0.2f, 5.51f, 0, 0, 0, 20}}
    };
    // A dilute mixed-species fixture spans multiple grid bins and crosses a
    // periodic boundary, exercising the output permutation of the cell backend.
    auto mixed = supercell(out[2], {2, 1, 1});
    mixed.name = "SeparatedWaters";
    mixed.positions[0] -= mixed.cell[0];
    mixed.positions[10] += 0.07f;
    out.push_back(mixed);

    std::vector<std::filesystem::path> paths;
    for (const auto& entry : std::filesystem::directory_iterator(TEST_STRUCTURE_DIR))
        if (entry.path().extension() == ".arc")
            paths.push_back(entry.path());
    std::sort(paths.begin(), paths.end());
    for (const auto& path : paths) {
        auto frames = chemio::read(path.string());
        for (size_t i = 0; i < frames.size(); ++i) {
            const auto& f = frames[i];
            if (f.pbc != std::array<bool, 3>{true, true, true})
                throw std::runtime_error("Test fixture must be fully periodic: " + path.string());
            Structure s{"ARC_" + path.stem().string() + "_" + std::to_string(i),
                        f.elements,
                        std::vector<float>(f.positions.begin(), f.positions.end()),
                        {}};
            if (s.name.find_first_not_of(
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_")
                != std::string::npos)
                throw std::runtime_error("Test fixture filename must be alphanumeric: "
                                         + path.string());
            std::copy(f.cell.begin(), f.cell.end(), s.cell.begin());
            out.push_back(std::move(s));
        }
    }
    return out;
}

lapis::Calculator& model(lapis::NeighborBackend backend) {
    // Reuse models within a binary run; --gtest_list_tests never initializes CUDA.
    static std::map<lapis::NeighborBackend, std::unique_ptr<lapis::Calculator>> models;
    auto& ptr = models[backend];
    if (!ptr)
        ptr = make_model(backend);
    return *ptr;
}

void compare(const lapis::Result& actual, const lapis::Result& expected) {
    ASSERT_NO_FATAL_FAILURE(valid(actual, expected.energies.size()));
    ASSERT_EQ(actual.forces.size(), expected.forces.size());
    // Total energies contain large atomic baselines: do not use a relative
    // tolerance on total energy that could conceal interaction-energy errors.
    near(actual.energy, expected.energy, 2e-4 * std::max(size_t(1), expected.energies.size()), 0);
    for (size_t i = 0; i < actual.energies.size(); ++i) {
        SCOPED_TRACE("atomic energy " + std::to_string(i));
        near(actual.energies[i], expected.energies[i], 2e-4,
             2 * std::numeric_limits<float>::epsilon());
    }
    for (size_t i = 0; i < actual.forces.size(); ++i) {
        SCOPED_TRACE("force " + std::to_string(i));
        near(actual.forces[i], expected.forces[i], 3e-4);
    }
    for (size_t i = 0; i < 9; ++i) {
        SCOPED_TRACE("virial " + std::to_string(i));
        near(actual.virial[i], expected.virial[i], 5e-4);
    }
}

class Physics : public testing::TestWithParam<Structure> {
  protected:
    lapis::Result calculate(const Structure& s) { return model().calculate(s.view()); }
};

TEST_P(Physics, EnergyAndOutputContract) {
    const auto& s = GetParam();
    const auto  r = calculate(s);
    ASSERT_NO_FATAL_FAILURE(valid(r, s.elements.size()));
    if (s.elements.empty()) {
        EXPECT_DOUBLE_EQ(r.energy, 0);
        for (float v : r.virial)
            EXPECT_FLOAT_EQ(v, 0);
    } else {
        EXPECT_LT(r.energy, 0);
    }
    // Per-atom outputs are float32, while total energy adds baselines in double.
    double sum      = std::accumulate(r.energies.begin(), r.energies.end(), 0.0);
    double rounding = 0;
    for (float e : r.energies)
        rounding += std::abs(e) * std::numeric_limits<float>::epsilon();
    EXPECT_NEAR(sum, r.energy, rounding + 1e-6);
}

TEST_P(Physics, TranslationInvariance) {
    auto       s         = GetParam();
    const auto reference = calculate(s);
    for (const auto shift : {
             std::array<float, 3>{0.37f, -0.23f, 0.41f},
             std::array<float, 3>{-3.1f, 2.7f,   -1.9f}
    }) {
        auto moved = s;
        for (size_t i = 0; i < moved.positions.size(); ++i)
            moved.positions[i] += shift[i % 3];
        compare(calculate(moved), reference);
    }
}

TEST_P(Physics, RotationCovariance) {
    const auto& s         = GetParam();
    const auto  reference = calculate(s);
    // Proper non-axis-aligned rotation from Rodrigues' formula, axis (1,2,3).
    const double                angle = 0.73, c = std::cos(angle), sn = std::sin(angle);
    const std::array<double, 3> axis{1 / std::sqrt(14.), 2 / std::sqrt(14.), 3 / std::sqrt(14.)};
    Matrix                      rotation{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            rotation[3 * i + j] = float((i == j ? c : 0) + (1 - c) * axis[i] * axis[j]);
    rotation[1] -= sn * axis[2];
    rotation[2] += sn * axis[1];
    rotation[3] += sn * axis[2];
    rotation[5] -= sn * axis[0];
    rotation[6] -= sn * axis[1];
    rotation[7] += sn * axis[0];
    auto expected   = reference;
    expected.forces = transform(reference.forces, rotation);
    // Row-vector geometry x' = x R implies virial' = R^T virial R.
    expected.virial =
        transform(transpose(transform(transpose(reference.virial), rotation)), rotation);
    compare(calculate(deform(s, rotation)), expected);
}

TEST_P(Physics, PermutationInvariance) {
    const auto&         s         = GetParam();
    const auto          reference = calculate(s);
    std::vector<size_t> order(s.elements.size());
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 rng(20261106);
    std::shuffle(order.begin(), order.end(), rng);
    if (order.size() > 1 && std::is_sorted(order.begin(), order.end()))
        std::reverse(order.begin(), order.end());
    auto permuted = s;
    auto expected = reference;
    for (size_t i = 0; i < order.size(); ++i) {
        permuted.elements[i] = s.elements[order[i]];
        expected.energies[i] = reference.energies[order[i]];
        for (int k = 0; k < 3; ++k) {
            permuted.positions[3 * i + k] = s.positions[3 * order[i] + k];
            expected.forces[3 * i + k]    = reference.forces[3 * order[i] + k];
        }
    }
    compare(calculate(permuted), expected);
}

TEST_P(Physics, PeriodicImageInvariance) {
    auto       s         = GetParam();
    const auto reference = calculate(s);
    for (size_t i = 0; i < s.elements.size(); ++i)
        for (int a = 0; a < 3; ++a)
            for (int k = 0; k < 3; ++k)
                s.positions[3 * i + k] += (int((i + 2 * a) % 5) - 2) * s.cell[3 * a + k];
    compare(calculate(s), reference);
}

TEST_P(Physics, NumericalForces) {
    const auto& s         = GetParam();
    const auto  reference = calculate(s);
    ASSERT_EQ(reference.forces.size(), s.positions.size());
    for (size_t i = 0; i < s.positions.size(); ++i) {
        SCOPED_TRACE("atom " + std::to_string(i / 3) + " axis " + std::to_string(i % 3));
        double derivatives[2];
        for (int level = 0; level < 2; ++level) {
            const float h    = level == 0 ? 0.004f : 0.002f;
            auto        plus = s, minus = s;
            plus.positions[i] += h;
            minus.positions[i] -= h;
            derivatives[level] = -(calculate(plus).energy - calculate(minus).energy)
                                 / (double(plus.positions[i]) - minus.positions[i]);
        }
        // Fourth-order central differences reduce truncation in narrow radial
        // features without using steps dominated by float32 energy roundoff.
        near((4 * derivatives[1] - derivatives[0]) / 3, reference.forces[i], 3e-3, 5e-3);
    }
}

TEST_P(Physics, NumericalStress) {
    const auto&  s         = GetParam();
    const auto   reference = calculate(s);
    const double v         = volume(s.cell);
    ASSERT_GT(v, 0);
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
            SCOPED_TRACE("strain " + std::to_string(a) + "," + std::to_string(b));
            double derivatives[2];
            for (int level = 0; level < 2; ++level) {
                const float h    = level == 0 ? 0.002f : 0.001f;
                auto        plus = identity, minus = identity;
                plus[3 * a + b] += h;
                minus[3 * a + b] -= h;
                // Deform BOTH lattice and atoms, keeping fractional coordinates fixed.
                derivatives[level] =
                    (calculate(deform(s, plus)).energy - calculate(deform(s, minus)).energy)
                    / (double(plus[3 * a + b]) - minus[3 * a + b]);
            }
            const double numerical = (4 * derivatives[1] - derivatives[0]) / (3 * v);
            near(numerical, -reference.virial[3 * a + b] / v, 4e-3 / v, 5e-3);
        }
}

TEST_P(Physics, SupercellConsistency) {
    const auto& s         = GetParam();
    const auto  reference = calculate(s);
    for (auto repeats : {
             std::array<int, 3>{2, 1, 1},
              std::array<int, 3>{1, 2, 3},
             std::array<int, 3>{2, 2, 2}
    }) {
        SCOPED_TRACE(testing::Message() << "repeats " << repeats[0] << repeats[1] << repeats[2]);
        auto       expanded = supercell(s, repeats);
        const auto actual   = calculate(expanded);
        ASSERT_NO_FATAL_FAILURE(valid(actual, expanded.elements.size()));
        const int copies = repeats[0] * repeats[1] * repeats[2];
        near(actual.energy / copies, reference.energy,
             2e-4 * std::max(size_t(1), s.elements.size()), 0);
        for (size_t i = 0; i < actual.energies.size(); ++i) {
            const size_t original = i % s.elements.size();
            near(actual.energies[i], reference.energies[original], 2e-4,
                 2 * std::numeric_limits<float>::epsilon());
            for (int k = 0; k < 3; ++k)
                near(actual.forces[3 * i + k], reference.forces[3 * original + k], 3e-4);
        }
        for (int k = 0; k < 9; ++k)
            near(-actual.virial[k] / volume(expanded.cell), -reference.virial[k] / volume(s.cell),
                 5e-4 / volume(s.cell));
    }
}

TEST_P(Physics, ForceConservationAndStressSymmetry) {
    const auto& s = GetParam();
    const auto  r = calculate(s);
    ASSERT_NO_FATAL_FAILURE(valid(r, s.elements.size()));
    for (int k = 0; k < 3; ++k) {
        double sum = 0, magnitude = 0;
        for (size_t i = k; i < r.forces.size(); i += 3) {
            sum += r.forces[i];
            magnitude += std::abs(r.forces[i]);
        }
        EXPECT_NEAR(sum, 0, 3e-5 + 1e-5 * magnitude);
    }
    for (int i = 0; i < 3; ++i)
        for (int j = i + 1; j < 3; ++j)
            near(r.virial[3 * i + j], r.virial[3 * j + i], 3e-4);
    if (s.elements.size() == 1)
        for (float f : r.forces)
            EXPECT_NEAR(f, 0, 3e-5);
}

TEST_P(Physics, RepeatAndReuseBuffers) {
    const auto& s         = GetParam();
    const auto  reference = calculate(s);
    for (int i = 0; i < 3; ++i)
        compare(calculate(s), reference);
    calculate(supercell(s, {2, 2, 2}));
    auto empty = s;
    empty.elements.clear();
    empty.positions.clear();
    auto zero = calculate(empty);
    EXPECT_DOUBLE_EQ(zero.energy, 0);
    EXPECT_TRUE(zero.energies.empty());
    EXPECT_TRUE(zero.forces.empty());
    for (float v : zero.virial)
        EXPECT_FLOAT_EQ(v, 0);
    compare(calculate(s), reference);
}

TEST_P(Physics, NeighborBackendsAgree) {
    const auto& s         = GetParam();
    const auto  reference = model(lapis::NeighborBackend::BruteForce).calculate(s.view());
    compare(calculate(s), reference);
    auto& cell = model(lapis::NeighborBackend::CellList);
    // Small-cell fixtures intentionally require periodic self images. A forced
    // grid must reject them explicitly, while Auto must still match brute force.
    try {
        compare(cell.calculate(s.view()), reference);
        EXPECT_EQ(cell.diagnostics().selected, lapis::NeighborBackend::CellList);
    } catch (const lapis::Error& e) {
        ASSERT_NE(std::string(e.what()).find(
                      "CellList requires all reciprocal plane heights >= cutoff"),
                  std::string::npos)
            << e.what();
        EXPECT_EQ(model().diagnostics().selected, lapis::NeighborBackend::BruteForce);
    }
}

TEST_P(Physics, RejectsInvalidInputAndRecovers) {
    const auto& s         = GetParam();
    const auto  reference = calculate(s);
    auto        reject    = [&](lapis::Structure bad) {
        EXPECT_THROW(model().calculate(bad), lapis::Error);
        compare(calculate(s), reference);
    };
    auto bad = s.view();
    bad.size = -1;
    reject(bad);
    bad      = s.view();
    bad.cell = {};
    reject(bad);
    bad         = s.view();
    bad.cell[0] = std::numeric_limits<float>::infinity();
    reject(bad);
    bad        = s.view();
    bad.pbc[2] = false;
    reject(bad);
    if (!s.elements.empty()) {
        bad           = s.view();
        bad.positions = nullptr;
        reject(bad);
        bad          = s.view();
        bad.elements = nullptr;
        reject(bad);
        auto invalid        = s;
        invalid.elements[0] = 0;
        reject(invalid.view());
        invalid             = s;
        invalid.elements[0] = 119;
        reject(invalid.view());
        invalid              = s;
        invalid.positions[0] = std::numeric_limits<float>::quiet_NaN();
        reject(invalid.view());
        // Exact overlapping distinct atoms must not be mistaken for self edges.
        invalid = s;
        invalid.elements.push_back(s.elements[0]);
        invalid.positions.insert(invalid.positions.end(), s.positions.begin(),
                                 s.positions.begin() + 3);
        reject(invalid.view());
    }
}

INSTANTIATE_TEST_SUITE_P(Structures, Physics, testing::ValuesIn(structures()),
                         [](const testing::TestParamInfo<Structure>& p) { return p.param.name; });
} // namespace potential_test
