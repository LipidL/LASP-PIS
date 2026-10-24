#include "weights.hpp"
#include <fstream>
#include <iostream>
int main(int argc, char **argv) {
  try {
    if (argc < 3 || argc > 4)
      throw std::runtime_error("usage: mace_weights_dump MODEL OUTPUT [HEAD]");
    auto bytes = mace_detail::weight_data(argv[1], argc == 4 ? argv[3] : "");
    std::ofstream out(argv[2], std::ios::binary);
    if (!out.write(bytes.data(), bytes.size()))
      throw std::runtime_error("write failed");
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
