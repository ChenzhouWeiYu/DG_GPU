#include "dg/physics/multi_component_physics.h"

template<uInt Ns>
HostDevice Scalar MultiComponentPhysics<Ns>::mixtureGamma(const std::array<Scalar, Ns>& Y) const {
    Scalar sum_Y_over_R = 0, sum_Y_Cv = 0;
    for (uInt i = 0; i < Ns; ++i) {
        Scalar R_i = 8314.462618 / M_list[i]; // R_universal / M_i
        Scalar Cv_i = R_i / (gamma_list[i] - 1.0);
        sum_Y_over_R += Y[i] / R_i;
        sum_Y_Cv += Y[i] * Cv_i;
    }
    Scalar R_mix = 1.0 / sum_Y_over_R;
    Scalar Cv_mix = sum_Y_Cv;
    Scalar Cp_mix = Cv_mix + R_mix;
    return Cp_mix / Cv_mix;
}

template<uInt Ns>
HostDevice DenseMatrix<Ns+4, 3> MultiComponentPhysics<Ns>::computeFluxImpl(const DenseMatrix<Ns+4, 1>& U) const {
    Scalar rho = 0;
    std::array<Scalar, Ns> Y;
    for (uInt i = 0; i < Ns; ++i) {
        rho += U[i];
        Y[i] = U[i] / rho;
    }
    rho = positive(rho);

    const Scalar u = U[Ns]/rho, v = U[Ns+1]/rho, w = U[Ns+2]/rho;
    const Scalar E = U[Ns+3]/rho;
    const Scalar ke = 0.5*(u*u + v*v + w*w);
    const Scalar e = E - ke;
    const Scalar p = computePressureImpl(U);

    DenseMatrix<Ns+4, 3> F;
    // ρY_i 通量
    for (uInt i = 0; i < Ns; ++i) {
        F(i,0) = U[i] * u;
        F(i,1) = U[i] * v;
        F(i,2) = U[i] * w;
    }
    // 动量通量
    F(Ns,0) = rho*u*u + p;   F(Ns,1) = rho*u*v;     F(Ns,2) = rho*u*w;
    F(Ns+1,0) = rho*v*u;     F(Ns+1,1) = rho*v*v + p; F(Ns+1,2) = rho*v*w;
    F(Ns+2,0) = rho*w*u;     F(Ns+2,1) = rho*w*v;     F(Ns+2,2) = rho*w*w + p;
    // 能量通量
    F(Ns+3,0) = (rho*E + p) * u;
    F(Ns+3,1) = (rho*E + p) * v;
    F(Ns+3,2) = (rho*E + p) * w;

    return F;
}

template<uInt Ns>
HostDevice Scalar MultiComponentPhysics<Ns>::computePressureImpl(const DenseMatrix<Ns+4, 1>& U) const {
    Scalar rho = 0;
    std::array<Scalar, Ns> Y;
    for (uInt i = 0; i < Ns; ++i) {
        rho += U[i];
        Y[i] = U[i] / rho;
    }
    rho = positive(rho);

    const Scalar u = U[Ns]/rho, v = U[Ns+1]/rho, w = U[Ns+2]/rho;
    const Scalar E = U[Ns+3]/rho;
    const Scalar ke = 0.5*(u*u + v*v + w*w);
    const Scalar e = E - ke;

    // 简化：假设 e = Cv_mix * T, p = rho * R_mix * T
    Scalar sum_Y_over_R = 0, sum_Y_Cv = 0;
    for (uInt i = 0; i < Ns; ++i) {
        Scalar R_i = 8314.462618 / M_list[i];
        Scalar Cv_i = R_i / (gamma_list[i] - 1.0);
        sum_Y_over_R += Y[i] / R_i;
        sum_Y_Cv += Y[i] * Cv_i;
    }
    Scalar R_mix = 1.0 / sum_Y_over_R;
    Scalar Cv_mix = sum_Y_Cv;
    Scalar T = e / Cv_mix;
    return rho * R_mix * T;
}

template<uInt Ns>
HostDevice Scalar MultiComponentPhysics<Ns>::computeSoundSpeedImpl(const DenseMatrix<Ns+4, 1>& U) const {
    const Scalar p = computePressureImpl(U);
    const Scalar rho = positive(U[0]); // 近似
    Scalar gamma_eff = 1.4; // 简化，实际应计算
    return sqrt(gamma_eff * p / rho);
}