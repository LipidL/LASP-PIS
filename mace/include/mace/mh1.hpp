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
  // Original MH1 checkpoint or PWMH1001 archive. Empty head selects
  // matpes_r2scan for checkpoints; processed archives already contain one head.
  explicit MH1(const std::string &model, lapis::Options options = {});
  MH1(const std::string &model, lapis::Options options, const std::string &head);
  ~MH1() override;
  size_t parameter_bytes() const;
  lapis::Result calculate_file(const std::string &path, size_t frame = 0);
};
} // namespace mace
