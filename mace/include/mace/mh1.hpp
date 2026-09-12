#pragma once
#include "runtime.hpp"
#include <memory>
namespace pairwise {
class MH1 final : public Calculator {
  struct Model;
  std::unique_ptr<Model> model_;
  Result evaluate() override;

public:
  explicit MH1(const std::string &archive, Options options = {});
  ~MH1() override;
  size_t parameter_bytes() const;
};
} // namespace pairwise
