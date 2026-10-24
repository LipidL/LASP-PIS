#include <checkpoint/checkpoint.hpp>
#include <iostream>
void print(checkpoint::Ptr p, int depth = 0) {
  if (depth > 4)
    return;
  std::cout << "kind=" << p->kind << " text=" << p->text.substr(0, 80)
            << " number=" << p->number << " items=" << p->items.size()
            << " fields=" << p->fields.size() << "\n";
  for (auto &x : p->items) {
    std::cout << std::string(depth * 2, ' ') << "item ";
    print(x, depth + 1);
  }
  for (auto &[k, v] : p->fields) {
    std::cout << std::string(depth * 2, ' ') << k << " ";
    print(v, depth + 1);
  }
}
int main(int argc, char **argv) {
  try {
    if (argc < 2 || argc > 3)
      throw std::runtime_error("usage: checkpoint_inspect MODEL [ATTRIBUTE]");
    checkpoint::Archive a(argv[1]);
    auto root = a.root();
    if (argc > 2) {
      print(checkpoint::module(root, argv[2]));
      return 0;
    }
    auto tensors = checkpoint::state_dict(root);
    std::cout << root->text << " " << tensors.size() << " tensors\n";
    for (auto &[k, t] : tensors) {
      auto d = t.doubles();
      std::cout << k << " [";
      for (auto n : t.shape)
        std::cout << n << ",";
      std::cout << "] " << (d.empty() ? 0 : d[0]) << "\n";
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
