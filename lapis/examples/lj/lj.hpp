#pragma once
#include <lapis/runtime.hpp>
namespace lapis {
class LennardJones final : public Calculator {
  float epsilon_, sigma_;
  Result evaluate() override;

public:
  LennardJones(float epsilon = 1, float sigma = 1, float cutoff = 2.5, Options options = {});
};
}
