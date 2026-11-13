#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mace/mh1.hpp>
#include <map>
#include <sstream>

#include "physics.hpp"
#include "weights.hpp"

namespace {
using Archive = std::map<std::string, std::vector<char>>;

Archive archive() {
    std::istringstream in(mace_detail::weight_data(TEST_MODEL_PATH, {}), std::ios::binary);
    in.ignore(8);
    uint32_t count;
    in.read(reinterpret_cast<char*>(&count), 4);
    Archive result;
    for (uint32_t k = 0; k < count; ++k) {
        uint32_t length;
        uint64_t bytes;
        in.read(reinterpret_cast<char*>(&length), 4);
        in.read(reinterpret_cast<char*>(&bytes), 8);
        std::string name(length, '\0');
        in.read(name.data(), length);
        auto& data = result[name];
        data.resize(bytes);
        in.read(data.data(), bytes);
    }
    if (!in)
        throw std::runtime_error("Failed to read test archive");
    return result;
}

struct TemporaryArchive {
    std::filesystem::path path;
    explicit TemporaryArchive(const Archive& data) {
        static int serial = 0;
        auto       stamp  = std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path()
               / ("mace-tp-" + std::to_string(stamp) + "-" + std::to_string(serial++) + ".pwm");
        std::ofstream out(path, std::ios::binary);
        out.write("PWMH1001", 8);
        uint32_t count = uint32_t(data.size());
        out.write(reinterpret_cast<const char*>(&count), 4);
        for (const auto& [name, bytes] : data) {
            uint32_t length = uint32_t(name.size());
            uint64_t size   = bytes.size();
            out.write(reinterpret_cast<const char*>(&length), 4);
            out.write(reinterpret_cast<const char*>(&size), 8);
            out.write(name.data(), length);
            out.write(bytes.data(), size);
        }
        if (!out)
            throw std::runtime_error("Failed to write test archive");
    }
    ~TemporaryArchive() {
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
    }
};

void change_weights(std::vector<char>& data) {
    for (size_t i = 0; i < data.size(); i += sizeof(float)) {
        float value;
        std::memcpy(&value, data.data() + i, sizeof(value));
        value *= 0.8f + 0.01f * float((i / sizeof(float)) % 7);
        std::memcpy(data.data() + i, &value, sizeof(value));
    }
}

void reverse_records(std::vector<char>& data, size_t width) {
    for (size_t a = 0, b = data.size() - width; a < b; a += width, b -= width)
        for (size_t k = 0; k < width; ++k)
            std::swap(data[a + k], data[b + k]);
}

class TensorProductCompatibility : public ::testing::TestWithParam<bool> {};

TEST_P(TensorProductCompatibility, ReorderedTermsAgreeWithSpecializedPaths) {
    static const Archive original  = archive();
    auto                 canonical = original;
    if (GetParam()) {
        // Exercise runtime CG coefficients and trained radial projections in
        // both layers. Specialization must never bake in checkpoint values.
        for (int layer = 0; layer < 2; ++layer) {
            auto prefix = "layer" + std::to_string(layer);
            for (const auto* suffix : {".tp.values", ".radial.9.weight", ".radial.9.bias"})
                change_weights(canonical.at(prefix + suffix));
        }
    }
    TemporaryArchive canonical_file(canonical);
    mace::MH1        specialized(canonical_file.path.string());
    auto             fixtures = potential_test::structures();
    auto fixture = std::find_if(fixtures.begin(), fixtures.end(),
                                [](const auto& s) { return s.name == "TriclinicTiO2"; });
    ASSERT_NE(fixture, fixtures.end());
    auto expected = specialized.calculate(fixture->view());
    if (GetParam()) {
        auto unchanged = potential_test::model().calculate(fixture->view());
        EXPECT_GT(std::abs(expected.energy - unchanged.energy), 1e-4);
    }
    for (int mask : {1, 2, 3}) {
        SCOPED_TRACE(mask);
        auto reordered = canonical;
        for (int layer = 0; layer < 2; ++layer) {
            if (!(mask & (1 << layer)))
                continue;
            auto prefix = "layer" + std::to_string(layer) + ".tp";
            reverse_records(reordered.at(prefix + ".terms"), 4 * sizeof(int));
            reverse_records(reordered.at(prefix + ".values"), sizeof(float));
        }
        TemporaryArchive fallback_file(reordered);
        mace::MH1        generic(fallback_file.path.string());
        auto             actual = generic.calculate(fixture->view());
        EXPECT_NEAR(actual.energy, expected.energy, 1e-5 * fixture->elements.size());
        for (size_t k = 0; k < actual.energies.size(); ++k)
            EXPECT_NEAR(actual.energies[k], expected.energies[k], 1e-5);
        for (size_t k = 0; k < actual.forces.size(); ++k)
            EXPECT_NEAR(actual.forces[k], expected.forces[k], 1e-4);
        for (size_t k = 0; k < actual.virial.size(); ++k)
            EXPECT_NEAR(actual.virial[k], expected.virial[k], 5e-4);
    }
}

INSTANTIATE_TEST_SUITE_P(MH1, TensorProductCompatibility, ::testing::Bool());
} // namespace
