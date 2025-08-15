// dg/dg_flux/physical_flux/euler_flux.h
#pragma once
#include "dg/dg_flux/physical_flux/physical_flux_base.h"

template<uInt N>
class EulerFlux final : public PhysicalFlux<N> {
    const Thermodynamics<N>& thermo;

public:
    explicit EulerFlux(const Thermodynamics<N>& t) : thermo(t) {}

    HostDevice ForceInline
    DenseMatrix<5 + N, 3> compute(const typename Thermodynamics<N>::Primitive& prim) const override {
        DenseMatrix<5 + N, 3> F;

        Scalar rhoU = prim.rho * prim.u;
        Scalar rhoV = prim.rho * prim.v;
        Scalar rhoW = prim.rho * prim.w;
        Scalar E = prim.e + 0.5*(prim.u*prim.u + prim.v*prim.v + prim.w*prim.w);
        Scalar energy_flux_x = prim.u * (prim.rho * E + prim.p);
        Scalar energy_flux_y = prim.v * (prim.rho * E + prim.p);
        Scalar energy_flux_z = prim.w * (prim.rho * E + prim.p);

        // [rho]
        F(0,0) = rhoU; F(0,1) = rhoV; F(0,2) = rhoW;
        // [rhou]
        F(1,0) = rhoU*prim.u + prim.p; F(1,1) = rhoU*prim.v;     F(1,2) = rhoU*prim.w;
        F(2,0) = rhoV*prim.u;     F(2,1) = rhoV*prim.v + prim.p; F(2,2) = rhoV*prim.w;
        F(3,0) = rhoW*prim.u;     F(3,1) = rhoW*prim.v;     F(3,2) = rhoW*prim.w + prim.p;
        // [rhoE]
        F(4,0) = energy_flux_x; F(4,1) = energy_flux_y; F(4,2) = energy_flux_z;

        // [rhoY_i]
        if constexpr (N > 0) {
            PragmaUnroll
            for (uInt i = 0; i < N; ++i) {
                F(5+i,0) = rhoU * prim.Y[i];
                F(5+i,1) = rhoV * prim.Y[i];
                F(5+i,2) = rhoW * prim.Y[i];
            }
        }

        return F;
    }

    HostDevice ForceInline
    DenseMatrix<5 + N, 1> compute_dot(
        const typename Thermodynamics<N>::Primitive& prim,
        const Vector3& rhs_dir
    ) const override {
        Scalar un = prim.u * rhs_dir[0] + prim.v * rhs_dir[1] + prim.w * rhs_dir[2];
        Scalar rhoun = prim.rho * un;
        Scalar E = prim.e + 0.5*(prim.u*prim.u + prim.v*prim.v + prim.w*prim.w);
        Scalar energy_flux = un * (prim.rho * E + prim.p);

        DenseMatrix<5 + N, 1> F_dot;
        F_dot(0,0) = rhoun;
        F_dot(1,0) = rhoun * prim.u + prim.p * rhs_dir[0];
        F_dot(2,0) = rhoun * prim.v + prim.p * rhs_dir[1];
        F_dot(3,0) = rhoun * prim.w + prim.p * rhs_dir[2];
        F_dot(4,0) = energy_flux;

        if constexpr (N > 0) {
            PragmaUnroll
            for (uInt i = 0; i < N; ++i) {
                F_dot(5+i,0) = rhoun * prim.Y[i];
            }
        }
        return F_dot;
    }
};