#pragma once
#include "dg/physics/physics_base.h"

/// 单组分理想气体物理模型（运行时 gamma）
class IdealGasPhysics : public PhysicsBase<IdealGasPhysics, 5> {
public:
    static constexpr uInt NEQN = 5;
    Scalar gamma; // 运行时参数

    HostDevice
    IdealGasPhysics(Scalar g = 1.4) : gamma(g) {}

    HostDevice
    DenseMatrix<5, 3> computeFluxImpl(const DenseMatrix<5, 1>& U) const;

    HostDevice
    Scalar computePressureImpl(const DenseMatrix<5, 1>& U) const;

    HostDevice
    Scalar computeSoundSpeedImpl(const DenseMatrix<5, 1>& U) const;

    HostDevice
    Scalar get_gamma() const { return gamma; }
};

// 实现（内联）
HostDevice inline DenseMatrix<5, 3> IdealGasPhysics::computeFluxImpl(const DenseMatrix<5, 1>& U) const {
    const Scalar rho = positive(U[0]);
    const Scalar u = U[1]/rho, v = U[2]/rho, w = U[3]/rho;
    const Scalar E = U[4]/rho;
    const Scalar p = computePressureImpl(U);
    return {
        rho*u, rho*v, rho*w,
        rho*u*u + p, rho*v*u, rho*w*u,
        rho*u*v, rho*v*v + p, rho*w*v,
        rho*u*w, rho*v*w, rho*w*w + p,
        u*(rho*E + p), v*(rho*E + p), w*(rho*E + p)
    };
}

HostDevice inline Scalar IdealGasPhysics::computePressureImpl(const DenseMatrix<5, 1>& U) const {
    const Scalar rho = positive(U[0]);
    const Scalar u = U[1]/rho, v = U[2]/rho, w = U[3]/rho;
    const Scalar E = U[4]/rho;
    const Scalar ke = 0.5*(u*u + v*v + w*w);
    return (gamma - 1.0) * rho * (E - ke);
}

HostDevice inline Scalar IdealGasPhysics::computeSoundSpeedImpl(const DenseMatrix<5, 1>& U) const {
    const Scalar p = computePressureImpl(U);
    const Scalar rho = positive(U[0]);
    return sqrt(gamma * p / rho);
}