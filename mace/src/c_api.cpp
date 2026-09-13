#include "mace/c_api.h"
#include "mace/mh1.hpp"
#include <algorithm>
#include <cstring>
#include <sstream>
using namespace lapis;
using mace::MH1;
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
void *mace_create_mh1(const char *path, int b) {
  return mace_create_mh1_head(path, b, nullptr);
}
void *mace_create_mh1_head(const char *path, int b, const char *head) {
  try {
    error.clear();
    if (!path)
      throw Error("Null archive path");
    return new MH1(path, options(b), head ? head : "");
  } catch (const std::exception &e) {
    error = e.what();
    return nullptr;
  }
}
void mace_destroy(void *h) {
  delete static_cast<MH1 *>(h);
}
const char *mace_last_error() {
  return error.c_str();
}
int mace_calculate(void *h, int n, const float *xyz, const int *z, const float *cell, double *energy,
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
    auto r = static_cast<MH1 *>(h)->calculate(s);
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
int mace_diagnostics(void *h, char *buffer, size_t capacity) {
  try {
    if (!h || !buffer)
      throw Error("Null diagnostics pointer");
    auto *c = static_cast<MH1 *>(h);
    auto &d = c->diagnostics();
    std::ostringstream s;
    const size_t weights = c->parameter_bytes();
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
