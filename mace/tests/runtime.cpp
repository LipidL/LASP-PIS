#include <algorithm>
#include <cstdlib>
#include <mace/mh1.hpp>

#include "physics.hpp"

namespace {
struct Environment {
    std::string name, old;
    bool        present;
    Environment(const char* key, const char* value) : name(key) {
        const char* previous = std::getenv(key);
        present              = previous != nullptr;
        if (previous)
            old = previous;
        if (value)
            setenv(key, value, 1);
        else
            unsetenv(key);
    }
    ~Environment() {
        if (present)
            setenv(name.c_str(), old.c_str(), 1);
        else
            unsetenv(name.c_str());
    }
};

TEST(MH1Runtime, GraphReplayTracksInputsGeometryAndCapacity) {
    Environment               enable("MACE_DISABLE_GRAPHS", nullptr);
    Environment               profiling("MACE_PROFILE_MH1", nullptr);
    Environment               default_math("MACE_DEFAULT_MATH", "1");
    mace::MH1                 accelerated(TEST_MODEL_PATH);
    Environment               disable("MACE_DISABLE_GRAPHS", "1");
    Environment               pedantic("MACE_DEFAULT_MATH", nullptr);
    mace::MH1                 reference(TEST_MODEL_PATH);
    potential_test::Structure small;
    small.cell = {5.5f, 0, 0, 0, 5.5f, 0, 0, 0, 5.5f};
    for (int x = 0; x < 2; ++x)
        for (int y = 0; y < 2; ++y)
            for (int z = 0; z < 2; ++z) {
                small.elements.push_back(14);
                small.positions.insert(small.positions.end(),
                                       {0.5f + x * 2.7f, 0.5f + y * 2.6f, 0.5f + z * 2.5f});
            }
    auto check = [&](const potential_test::Structure& s) {
        // Warm, capture, and replay each new configuration. Between iterations
        // change coordinates without changing shape, exercising live input data.
        auto moving = s;
        for (int call = 0; call < 4; ++call) {
            if (!moving.positions.empty())
                moving.positions[0] += .001f;
            auto expected = reference.calculate(moving.view());
            auto actual   = accelerated.calculate(moving.view());
            EXPECT_NEAR(actual.energy, expected.energy,
                        1e-5 * std::max(size_t(1), s.elements.size()));
            ASSERT_EQ(actual.energies.size(), expected.energies.size());
            ASSERT_EQ(actual.forces.size(), expected.forces.size());
            for (size_t i = 0; i < actual.energies.size(); ++i)
                EXPECT_NEAR(actual.energies[i], expected.energies[i], 1e-5);
            for (size_t i = 0; i < actual.forces.size(); ++i)
                EXPECT_NEAR(actual.forces[i], expected.forces[i], 1e-4);
            for (int i = 0; i < 9; ++i)
                EXPECT_NEAR(actual.virial[i], expected.virial[i], 5e-4);
        }
    };
    check(small);
    EXPECT_EQ(accelerated.diagnostics().selected, lapis::NeighborBackend::BruteForce);
    auto changed        = small;
    changed.elements[0] = 8;
    check(changed);
    changed.cell = {12, 0, 0, 1, 13, 0, 0, 0, 14};
    check(changed);
    EXPECT_EQ(accelerated.diagnostics().selected, lapis::NeighborBackend::CellList);
    changed.cell[1] = .125f; // Same count and grid, different by-value kernel geometry.
    check(changed);
    check(potential_test::supercell(changed, {2, 2, 2}));
    check(small); // Shrink without releasing the larger capacities.
    auto empty = small;
    empty.positions.clear();
    empty.elements.clear();
    check(empty);
    check(changed);
    auto overlap = changed;
    std::copy(overlap.positions.begin(), overlap.positions.begin() + 3,
              overlap.positions.begin() + 3);
    // Same geometry/cache key: device error must survive graph replay and the
    // packed status transfer, and the next valid call must clear that status.
    EXPECT_THROW(accelerated.calculate(overlap.view()), lapis::Error);
    check(changed);
}
} // namespace
