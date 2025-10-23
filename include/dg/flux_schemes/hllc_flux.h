#pragma once
#include "dg/flux_schemes/flux_scheme_base.h"
#include "dg/flux_schemes/lax_friedrichs_flux.h"

template<typename Physics>
class HLLCFlux : public FluxSchemeBase<Physics> {
public:
    // static constexpr uInt NEQN = Physics::NEQN;
    using Base = FluxSchemeBase<Physics>;
    // using Base::compute;
    using Base::NEQN;

    HostDevice
    static DenseMatrix<NEQN, 1> compute(
        const Physics& physics,
        const DenseMatrix<NEQN, 1>& UL,
        const DenseMatrix<NEQN, 1>& UR,
        Scalar nx, Scalar ny, Scalar nz) {
        
        // 简化实现：回退到 Lax-Friedrichs
        // 实际 HLLC 需要 Roe 平均或波速估计
        return LaxFriedrichsFlux<Physics>::compute(physics, UL, UR, nx, ny, nz);
    }
    HostDevice
    static DenseMatrix<NEQN, 1> compute(
        const Physics& physics,
        const DenseMatrix<NEQN, 1>& UL,
        const DenseMatrix<NEQN, 1>& UR,
        vector3f vec) {
            return compute(physics, UL, UR, vec[0], vec[1], vec[2]);
        }
        
};