#pragma once
#include <mace/export.h>
#include <lapis/runtime.hpp>
#include <memory>
namespace mace {
class MACE_EXPORT MH1 final : public lapis::Calculator {
  struct Model;
  std::unique_ptr<Model> model_;
  lapis::Result evaluate() override;

public:
  explicit MH1(const std::string &archive, lapis::Options options = {});
  ~MH1() override;
  size_t parameter_bytes() const;
};
} // namespace mace
