#include <dlfcn.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace {
struct Library {
    void* library    = nullptr;
    void* calculator = nullptr;
    using Calculate  = int (*)(void*, int, const float*, const int*, const float*, double*, float*,
                               float*, float*);
    Calculate calculate    = nullptr;
    void (*destroy)(void*) = nullptr;
    const char* (*error)() = nullptr;
    template<class Function>
    Function symbol(const std::string& name) {
        auto ptr = dlsym(library, name.c_str());
        if (!ptr)
            throw std::runtime_error(dlerror());
        return reinterpret_cast<Function>(ptr);
    }
    Library() {
        library = dlopen(MACE_LIBRARY_PATH, RTLD_NOW | RTLD_GLOBAL);
        if (!library)
            throw std::runtime_error(dlerror());
        try {
            destroy    = symbol<void (*)(void*)>("mace_destroy");
            error      = symbol<const char* (*)()>("mace_last_error");
            calculate  = symbol<Calculate>("mace_calculate");
            calculator = symbol<void* (*)(const char*, int)>("mace_create_mh1")(MACE_MODEL_PATH, 0);
            if (!calculator)
                throw std::runtime_error(error());
        } catch (...) {
            dlclose(library);
            throw;
        }
    }
    Library(const Library&) = delete;
    ~Library() {
        if (calculator)
            destroy(calculator);
        if (library)
            dlclose(library);
    }
    std::array<double, 22> evaluate() {
        const float xyz[]{2, 2, 2, 2.97f, 2.02f, 2.01f, 1.76f, 2.94f, 2.03f};
        const int   elements[]{8, 1, 1};
        const float cell[]{16, 0, 0, 0, 17, 0, 0, 0, 18};
        double      energy;
        float       atomic[3], force[9], virial[9];
        if (calculate(calculator, 3, xyz, elements, cell, &energy, atomic, force, virial))
            throw std::runtime_error(error());
        std::array<double, 22> result{};
        result[0] = energy;
        std::copy(atomic, atomic + 3, result.begin() + 1);
        std::copy(force, force + 9, result.begin() + 4);
        std::copy(virial, virial + 9, result.begin() + 13);
        return result;
    }
};
TEST(CAPI, InterleavedCalculators) {
    Library    first;
    const auto before = first.evaluate();
    {
        Library    second;
        const auto other = second.evaluate();
        for (int run = 0; run < 3; ++run) {
            const auto actual_first = first.evaluate(), actual_second = second.evaluate();
            for (size_t i = 0; i < before.size(); ++i) {
                ASSERT_TRUE(std::isfinite(actual_first[i]));
                ASSERT_TRUE(std::isfinite(actual_second[i]));
                EXPECT_NEAR(actual_first[i], before[i], 3e-4);
                EXPECT_NEAR(actual_second[i], other[i], 3e-4);
            }
        }
    }
    const auto after = first.evaluate();
    for (size_t i = 0; i < before.size(); ++i)
        EXPECT_NEAR(after[i], before[i], 3e-4);
}
} // namespace
