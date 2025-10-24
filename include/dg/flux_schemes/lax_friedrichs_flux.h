#pragma once
#include "dg/flux_schemes/flux_scheme_base.h"

template<typename Physics>
class LaxFriedrichsFlux : public FluxSchemeBase<Physics> {
public:
    // static constexpr uInt NEQN = Physics::NEQN;
    using Base = FluxSchemeBase<Physics>;
    // using Base::compute; 
    using Base::NEQN;

    HostDevice
    static DenseMatrix<NEQN, 1> compute(
        const Physics& physics,
        const DenseMatrix<NEQN, 1>& U_L,
        const DenseMatrix<NEQN, 1>& U_R,
        Scalar nx, Scalar ny, Scalar nz) {
        
        auto flux_L = physics.compute_flux_dot_vec(U_L, nx, ny, nz);
        auto flux_R = physics.compute_flux_dot_vec(U_R, nx, ny, nz);

        Scalar rho_L = positive(U_L[0]), rho_R = positive(U_R[0]);
        Scalar un_L = (U_L[1]*nx + U_L[2]*ny + U_L[3]*nz) / rho_L;
        Scalar un_R = (U_R[1]*nx + U_R[2]*ny + U_R[3]*nz) / rho_R;
        Scalar a_L = physics.compute_sound_speed(U_L);
        Scalar a_R = physics.compute_sound_speed(U_R);
        Scalar lambda = fmax(fabs(un_L) + a_L, fabs(un_R) + a_R);

        DenseMatrix<NEQN, 1> flux;
        for (uInt i = 0; i < NEQN; ++i) {
            flux[i] = 0.5 * (flux_L[i] + flux_R[i]) - 0.5 * lambda * (U_R[i] - U_L[i]);
        }
        return flux;
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