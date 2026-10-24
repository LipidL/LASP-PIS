#include <checkpoint/checkpoint.hpp>

#include <cstring>
#include <iostream>
#include <stdexcept>

void require(bool b) {
  if (!b)
    throw std::runtime_error("checkpoint test failed");
}
int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::runtime_error("expected ZIP fixture");
    checkpoint::Archive archive{std::string(argv[1])};
    require(archive.root()->kind == checkpoint::Value::Null);
    // A reducer naming a Python callable stays an inert record; it is not
    // invoked.
    std::string inert = "\x80\x02"
                        "cbuiltins\neval\n)R.";
    auto object = checkpoint::parse({inert.begin(), inert.end()}, archive);
    require(object->kind == checkpoint::Value::Object &&
            object->text == "builtins eval");
    for (const auto &bad : {std::string("\x80\x02"
                                        "h\x01.",
                                        5),
                            std::string("\x80\x02"
                                        "N"),
                            std::string("\x80\x02"
                                        "u."),
                            std::string("\x80\x02"
                                        "X\xff\xff\xff\x7f.",
                                        8)}) {
      bool rejected = false;
      try {
        checkpoint::parse({bad.begin(), bad.end()}, archive);
      } catch (const std::exception &) {
        rejected = true;
      }
      require(rejected);
    }
    // Storage offset and noncontiguous row strides must be respected.
    checkpoint::Tensor t;
    t.dtype = "FloatStorage";
    t.offset = 1;
    t.shape = {2, 2};
    t.stride = {3, 1};
    float storage[] = {0, 1, 2, 3, 4, 5};
    t.storage = std::make_shared<std::vector<char>>(sizeof(storage));
    std::memcpy(t.storage->data(), storage, sizeof(storage));
    require(t.doubles() == std::vector<double>({1, 2, 4, 5}));
    t.offset = 2;
    bool rejected = false;
    try {
      t.doubles();
    } catch (const std::exception &) {
      rejected = true;
    }
    require(rejected);
    std::cout << "Archive, inert reducers, malformed pickle, and strided "
                 "storage passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
