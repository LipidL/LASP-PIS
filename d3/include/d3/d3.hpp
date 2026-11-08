#pragma once
#include <lapis/runtime.hpp>

namespace d3 {
// Atomic-unit conversion used by the original DFT-D3 parameterization.
inline constexpr double bohr = 0.52917726, hartree = 27.21138505;
struct Parameters {
    // PBE-D3(BJ) by default. a2 is in bohr; the other parameters are dimensionless.
    double s6 = 1.0, s8 = 0.7875, a1 = 0.4289, a2 = 4.4407;
};
struct Cutoffs {
    float coordination = float(40 * bohr), dispersion = float(95 * bohr); // Angstrom
};

// Two-body DFT-D3(BJ), including derivatives through coordination-dependent C6.
// H through Pu (Z=1..94); three periodic axes, as required by Lapis.
// Inputs: Angstrom. Outputs: eV, eV/Angstrom, and virial in eV.
class Calculator final : public lapis::Calculator {
    Parameters    parameters_;
    Cutoffs       cutoffs_;
    const int*    offsets_;
    const double *reference_cn_, *c6_, *rcov_, *r2r4_;
    lapis::Result evaluate() override;

  public:
    explicit Calculator(Parameters parameters = {}, Cutoffs cutoffs = {},
                        lapis::Options options = {});
};
} // namespace d3
