#include <chemio/structure.hpp>
#include <climits>
#include <cmath>
#include <mace/mh1.hpp>

namespace mace {
lapis::Result MH1::calculate_file(const std::string &path, size_t frame) {
  auto owned = chemio::read_frame(path, frame);
  if (!owned.pbc[0] || !owned.pbc[1] || !owned.pbc[2])
    throw lapis::Error("mace requires three periodic axes");
  if (owned.elements.size() > INT_MAX)
    throw lapis::Error("too many atoms");
  std::vector<float> xyz;
  xyz.reserve(owned.positions.size());
  for (double x : owned.positions) {
    if (!std::isfinite(float(x)))
      throw lapis::Error("coordinate exceeds float32 range");
    xyz.push_back(float(x));
  }
  lapis::Structure s;
  s.size = int(owned.elements.size());
  s.positions = xyz.data();
  s.elements = owned.elements.data();
  s.pbc = owned.pbc;
  for (size_t k = 0; k < 9; ++k) {
    if (!std::isfinite(float(owned.cell[k])))
      throw lapis::Error("cell exceeds float32 range");
    s.cell[k] = float(owned.cell[k]);
  }
  return calculate(s);
}
} // namespace mace
