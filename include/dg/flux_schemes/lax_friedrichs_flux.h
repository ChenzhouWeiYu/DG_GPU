#pragma once
#include "dg/flux_schemes/flux_scheme_base.h"

template<typename Physics>
class LaxFriedrichsFlux : public FluxSchemeBase<Physics> {
public:
    using Base = FluxSchemeBase<Physics>;
    static constexpr uInt NEQN = Base::NEQN;

    HostDevice
    static DenseMatrix<NEQN, 1> compute(
        const Physics& physics,
        const DenseMatrix<NEQN, 1>& UL,
        const DenseMatrix<NEQN, 1>& UR,
        Scalar nx, Scalar ny, Scalar nz) {
        
        auto FL = physics.computeFlux(UL);
        auto FR = physics.computeFlux(UR);
        DenseMatrix<NEQN, 1> flux_L, flux_R;
        for (uInt i = 0; i < NEQN; ++i) {
            flux_L[i] = FL(i,0)*nx + FL(i,1)*ny + FL(i,2)*nz;
            flux_R[i] = FR(i,0)*nx + FR(i,1)*ny + FR(i,2)*nz;
        }

        // 计算最大波速
        Scalar aL = physics.computeSoundSpeed(UL);
        Scalar aR = physics.computeSoundSpeed(UR);
        Scalar rhoL = positive(UL[0]), rhoR = positive(UR[0]);
        Scalar uL = UL[1]/rhoL, vL = UL[2]/rhoL, wL = UL[3]/rhoL;
        Scalar uR = UR[1]/rhoR, vR = UR[2]/rhoR, wR = UR[3]/rhoR;
        Scalar unL = std::abs(uL*nx + vL*ny + wL*nz);
        Scalar unR = std::abs(uR*nx + vR*ny + wR*nz);
        Scalar lambda = std::max(unL + aL, unR + aR);

        DenseMatrix<NEQN, 1> flux;
        for (uInt i = 0; i < NEQN; ++i) {
            flux[i] = 0.5 * (flux_L[i] + flux_R[i]) - 0.5 * lambda * (UR[i] - UL[i]);
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