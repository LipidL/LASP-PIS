#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mace/mh1.hpp>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
void usage() {
  std::cout << "Usage: mace-main MODEL STRUCTURE.arc [--frame N] [--backend auto|brute|cell] "
               "[--head NAME] [--json] [--output FILE]\n";
}
size_t frame_index(const std::string &s) {
  if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos)
    throw std::runtime_error("frame must be a nonnegative integer");
  auto n = std::stoull(s);
  if (n > std::numeric_limits<size_t>::max())
    throw std::runtime_error("frame index too large");
  return size_t(n);
}
template <class Values> void array(std::ostream &out, const Values &values) {
  out << '[';
  bool first = true;
  for (auto v : values) {
    if (!first)
      out << ',';
    out << v;
    first = false;
  }
  out << ']';
}
} // namespace
int main(int argc, char **argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--help") {
      usage();
      return 0;
    }
    if (argc < 3) {
      usage();
      return 2;
    }
    std::string model = argv[1], structure = argv[2], head, output;
    size_t frame = 0;
    bool json = false;
    lapis::Options options;
    for (int i = 3; i < argc; ++i) {
      std::string flag = argv[i];
      if (flag == "--json") {
        json = true;
        continue;
      }
      if (flag != "--frame" && flag != "--backend" && flag != "--output" && flag != "--head")
        throw std::runtime_error("unknown option: " + flag);
      if (++i == argc)
        throw std::runtime_error("missing value for " + flag);
      std::string value = argv[i];
      if (flag == "--frame")
        frame = frame_index(value);
      else if (flag == "--output")
        output = value;
      else if (flag == "--head")
        head = value;
      else if (value == "auto")
        options.backend = lapis::NeighborBackend::Auto;
      else if (value == "brute")
        options.backend = lapis::NeighborBackend::BruteForce;
      else if (value == "cell")
        options.backend = lapis::NeighborBackend::CellList;
      else
        throw std::runtime_error("unknown backend: " + value);
    }
    if (!output.empty() && std::filesystem::exists(output) &&
        (std::filesystem::equivalent(output, model) ||
         std::filesystem::equivalent(output, structure)))
      throw std::runtime_error("output would overwrite an input file");
    mace::MH1 calculator(model, options, head);
    auto result = calculator.calculate_file(structure, frame);
    std::ofstream file;
    if (!output.empty()) {
      file.open(output);
      if (!file)
        throw std::runtime_error("cannot open output: " + output);
    }
    std::ostream &out = output.empty() ? std::cout : file;
    out << std::setprecision(17);
    std::ifstream model_file(model, std::ios::binary);
    char signature[2]{};
    model_file.read(signature, 2);
    std::cerr << "MACE head: "
              << (signature[0] == 'P' && signature[1] == 'K'
                      ? (head.empty() ? "matpes_r2scan" : head)
                      : "fixed by processed archive")
              << '\n';
    if (json) {
      out << "{\"frame\":" << frame << ",\"atoms\":" << result.energies.size()
          << ",\"units\":{\"energy\":\"eV\",\"forces\":\"eV/"
             "Angstrom\",\"virial\":\"eV\"},\"energy\":"
          << result.energy << ",\"atomic_energies\":";
      array(out, result.energies);
      out << ",\"forces\":[";
      for (size_t k = 0; k < result.forces.size(); k += 3) {
        if (k)
          out << ',';
        out << '[' << result.forces[k] << ',' << result.forces[k + 1] << ',' << result.forces[k + 2]
            << ']';
      }
      out << "],\"virial\":";
      array(out, result.virial);
      out << "}\n";
    } else {
      out << "Frame: " << frame << "\nAtoms: " << result.energies.size()
          << "\nEnergy (eV): " << result.energy << "\nForces (eV/Angstrom), input atom order:\n";
      for (size_t k = 0; k < result.forces.size(); k += 3)
        out << k / 3 << ' ' << result.forces[k] << ' ' << result.forces[k + 1] << ' '
            << result.forces[k + 2] << '\n';
      out << "Virial (eV), -dE/dstrain, row major:\n";
      for (int k = 0; k < 9; k += 3)
        out << result.virial[k] << ' ' << result.virial[k + 1] << ' ' << result.virial[k + 2]
            << '\n';
    }
    out.flush();
    if (!out)
      throw std::runtime_error("failed to write inference output");
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "mace-main: " << e.what() << '\n';
    return 1;
  }
}
