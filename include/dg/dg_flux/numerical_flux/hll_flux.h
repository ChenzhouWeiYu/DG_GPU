// dg/dg_flux/numerical_flux/hll_flux.h
#pragma once
#include "dg/dg_flux/numerical_flux/numerical_flux_base.h"

template<uInt N>
class HLLFlux final : public NumericalFlux<N> {
    using Base = NumericalFlux<N>;
    using PhyVector = DenseMatrix<5 + N, 1>;

public:
    explicit HLLFlux(const Thermodynamics<N>& t, const PhysicalFlux<N>& pflux)
        : Base(t, pflux) {}

    HostDevice ForceInline
    PhyVector compute(
        const PhyVector& UL,
        const PhyVector& UR,
        const Vector3& normal
    ) const override {
        auto prim_L = this->thermo.reconstruct(UL);
        auto prim_R = this->thermo.reconstruct(UR);

        Scalar uL = prim_L.u*normal[0] + prim_L.v*normal[1] + prim_L.w*normal[2];
        Scalar uR = prim_R.u*normal[0] + prim_R.v*normal[1] + prim_R.w*normal[2];
        Scalar aL = prim_L.a, aR = prim_R.a;
        Scalar SL = min(uL - aL, uR - aR);
        Scalar SR = max(uL + aL, uR + aR);

        Scalar SL0 = min(0.0, SL), SR0 = max(0.0, SR);
        auto FL = this->physical_flux.compute_dot(prim_L, normal);
        auto FR = this->physical_flux.compute_dot(prim_R, normal);
        // if (SR0 - SL0 < 1e-12) {
        //     return 0.5 * (FL + FR);
        // }
        return (SR0 * FL - SL0 * FR + SL0 * SR0 * (UR - UL)) / (SR0 - SL0);
    }
};