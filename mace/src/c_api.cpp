#include "pairwise/c_api.h"
#include "pairwise/mh1.hpp"
#include <algorithm>
#include <cstring>
#include <sstream>
using namespace pairwise;
namespace {
thread_local std::string error;
Options options(int b) {
  if (b < 0 || b > 2)
    throw Error("Invalid backend");
  Options o;
  o.backend = NeighborBackend(b);
  return o;
}
} // namespace
extern "C" {
void *pw_create_lj(float e, float s, float c, int b) {
  try {
    error.clear();
    return new LennardJones(e, s, c, options(b));
  } catch (const std::exception &e) {
    error = e.what();
    return nullptr;
  }
}
void *pw_create_mh1(const char *path, int b) {
  try {
    error.clear();
    if (!path)
      throw Error("Null archive path");
    return new MH1(path, options(b));
  } catch (const std::exception &e) {
    error = e.what();
    return nullptr;
  }
}
void pw_destroy(void *h) {
  delete static_cast<Calculator *>(h);
}
const char *pw_last_error() {
  return error.c_str();
}
int pw_calculate(void *h, int n, const float *xyz, const int *z, const float *cell, double *energy,
                 float *per_atom, float *force, float *virial) {
  try {
    error.clear();
    if (!h || !cell || !energy || !virial || (n && (!per_atom || !force)))
      throw Error("Null calculator or result pointer");
    Structure s;
    s.size = n;
    s.positions = xyz;
    s.elements = z;
    std::copy(cell, cell + 9, s.cell.begin());
    auto r = static_cast<Calculator *>(h)->calculate(s);
    *energy = r.energy;
    std::copy(r.energies.begin(), r.energies.end(), per_atom);
    std::copy(r.forces.begin(), r.forces.end(), force);
    std::copy(r.virial.begin(), r.virial.end(), virial);
    return 0;
  } catch (const std::exception &e) {
    error = e.what();
    return -1;
  }
}
int pw_diagnostics(void *h, char *buffer, size_t capacity) {
  try {
    if (!h || !buffer)
      throw Error("Null diagnostics pointer");
    auto *c = static_cast<Calculator *>(h);
    auto &d = c->diagnostics();
    std::ostringstream s;
    size_t weights = 0;
    if (auto *m = dynamic_cast<MH1 *>(c))
      weights = m->parameter_bytes();
    s << "{\"backend\":\"" << (d.selected == NeighborBackend::CellList ? "CellList" : "BruteForce")
      << "\",\"grid\":[" << d.grid[0] << "," << d.grid[1] << "," << d.grid[2] << "],\"reason\":\""
      << d.reason << "\",\"max_occupancy\":" << d.max_occupancy
      << ",\"occupied_bins\":" << d.occupied_bins << ",\"mean_occupancy\":" << d.mean_occupancy
      << ",\"grid_candidates\":" << d.grid_candidates
      << ",\"brute_candidates\":" << d.brute_candidates
      << ",\"context_bytes\":" << d.allocated_bytes << ",\"context_peak_bytes\":" << d.peak_bytes
      << ",\"parameter_bytes\":" << weights << ",\"device_ms\":" << d.device_ms << ",\"kernels\":[";
    bool first = true;
    for (auto &r : c->resources()) {
      if (!first)
        s << ",";
      first = false;
      s << "{\"name\":\"" << r.name << "\",\"registers\":" << r.registers
        << ",\"threads\":" << r.block_threads << ",\"static_shared\":" << r.static_shared
        << ",\"dynamic_shared\":" << r.dynamic_shared
        << ",\"blocks_per_sm\":" << r.active_blocks_per_sm << "}";
    }
    s << "]}";
    auto str = s.str();
    if (str.size() + 1 > capacity)
      throw Error("Diagnostics buffer too small");
    memcpy(buffer, str.c_str(), str.size() + 1);
    return 0;
  } catch (const std::exception &e) {
    error = e.what();
    return -1;
  }
}
}
