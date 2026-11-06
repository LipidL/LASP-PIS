#pragma once
#include <gtest/gtest.h>

#include <lapis/runtime.hpp>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace potential_test {
using Matrix = std::array<float, 9>;
struct Structure {
    std::string        name;
    std::vector<int>   elements;
    std::vector<float> positions;
    Matrix             cell;
    lapis::Structure   view() const {
        return {int(elements.size()), positions.data(), elements.data(), cell};
    }
};
inline void PrintTo(const Structure& s, std::ostream* out) {
    *out << s.name << " (" << s.elements.size() << " atoms)";
}
std::vector<Structure> structures();
Structure              supercell(const Structure&, std::array<int, 3> repeats);
double                 volume(const Matrix&);
// Each model supplies only this adapter; every structure gets the same tests.
std::unique_ptr<lapis::Calculator> make_model(lapis::NeighborBackend);
lapis::Calculator&                 model(lapis::NeighborBackend = lapis::NeighborBackend::Auto);
void                               compare(const lapis::Result&, const lapis::Result&);
} // namespace potential_test
