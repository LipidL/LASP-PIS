#pragma once
#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace chemio {
// Cartesian Angstrom coordinates and row-vector lattice; owns all storage.
struct Structure {
  std::vector<double> positions;
  std::vector<int> elements;
  std::array<double, 9> cell{};
  std::array<bool, 3> pbc{false, false, false};
  std::string title;
};
std::array<double, 9> cell_from_parameters(double a, double b, double c,
                                         double alpha, double beta, double gamma);
// ARC frames are zero based. A malformed or incomplete frame is an error.
std::vector<Structure> read(const std::string& path);
Structure read_frame(const std::string& path, std::size_t frame = 0);
}  // namespace chemio
