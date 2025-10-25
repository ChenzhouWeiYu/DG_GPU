// rotated_flux_scheme.h
#pragma once
#include "dg/flux_schemes/flux_scheme_base.h"
template<typename FluxImpl, typename Physics>
class RotatedFluxScheme : public FluxSchemeBase<Physics> {
public:
    using PhysicsType = Physics;
    using Base = FluxSchemeBase<Physics>;
    using Base::NEQN;

    HostDevice __forceinline__
    static DenseMatrix<NEQN, 1> compute(
        const Physics& physics,
        const DenseMatrix<NEQN, 1>& U_L,
        const DenseMatrix<NEQN, 1>& U_R,
        Scalar nx, Scalar ny, Scalar nz) {
        
        auto Q = Base::build_rotation_matrix(nx, ny, nz);
        DenseMatrix<NEQN, 1> QU_L = U_L;
        DenseMatrix<NEQN, 1> QU_R = U_R;
        Base::rotate_conserved(QU_L, Q);
        Base::rotate_conserved(QU_R, Q);

        // 关键：直接调用 FluxImpl::compute_1d
        DenseMatrix<NEQN, 1> F_prime = FluxImpl::compute_1d(physics, QU_L, QU_R);

        Base::inverse_rotate_flux(F_prime, Q);
        return F_prime;
    }

    HostDevice __forceinline__
    static DenseMatrix<NEQN, 1> compute(
        const Physics& physics,
        const DenseMatrix<NEQN, 1>& UL,
        const DenseMatrix<NEQN, 1>& UR,
        vector3f vec) {
        return compute(physics, UL, UR, vec[0], vec[1], vec[2]);
    }
};