#pragma once
#include "dg/physics/physics_base.h"

/// 单组分理想气体物理模型（运行时 gamma）
class IdealGasPhysics : public PhysicsBase<IdealGasPhysics, 5> {
public:
    static constexpr uInt NEQN = 5;
    Scalar gamma; // 运行时参数

    HostDevice
    IdealGasPhysics(Scalar g_ = 1.4) : gamma(g_) {}

    
    HostDevice __forceinline__ 
    DenseMatrix<5, 3> compute_flux_impl(const DenseMatrix<5, 1>& U) const {
        const Scalar rho = positive(U[0]);
        const Scalar u = U[1]/rho, v = U[2]/rho, w = U[3]/rho;
        const Scalar E = U[4]/rho;
        const Scalar p = compute_pressure_impl(U);
        return {
            rho*u, rho*v, rho*w,
            rho*u*u + p, rho*v*u, rho*w*u,
            rho*u*v, rho*v*v + p, rho*w*v,
            rho*u*w, rho*v*w, rho*w*w + p,
            u*(rho*E + p), v*(rho*E + p), w*(rho*E + p)
        };
    }

    HostDevice __forceinline__ 
    DenseMatrix<5, 1> compute_flux_1d_impl(const DenseMatrix<5, 1>& U) const {
        const Scalar rho = positive(U[0]);
        const Scalar u = U[1] / rho;
        const Scalar E = U[4] / rho;
        const Scalar p = compute_pressure_impl(U);
        return {
            rho * u,
            rho * u * u + p,
            rho * u * (U[2] / rho), // v 不变
            rho * u * (U[3] / rho), // w 不变
            u * (rho * E + p)
        };
    }

    HostDevice __forceinline__
    Scalar compute_pressure_impl(const DenseMatrix<5, 1>& U) const {
        // const Scalar rho = positive(U[0]);
        // const Scalar u = U[1]/rho, v = U[2]/rho, w = U[3]/rho;
        // const Scalar E = U[4]/rho;
        // const Scalar ke = 0.5*(u*u + v*v + w*w);
        // return (gamma - 1.0) * rho * (E - ke);
        
        const Scalar rho = positive(U[0]);
        const Scalar ke = 0.5 * (U[1]*U[1] + U[2]*U[2] + U[3]*U[3]) / rho;
        const Scalar p = (get_gamma() - 1.0) * (U[4] - ke);
        return p;
    }

    HostDevice __forceinline__ 
    Scalar compute_sound_speed_impl(const DenseMatrix<5, 1>& U) const {
        const Scalar p = compute_pressure_impl(U);
        const Scalar rho = positive(U[0]);
        return sqrt(gamma * p / rho);
    }

    HostDevice __forceinline__ 
    DenseMatrix<5, 1> compute_pressure_gradient_impl(const DenseMatrix<5, 1>& U) const {
        const Scalar rho = positive(U[0]);
        const Scalar rho_u = U[1], rho_v = U[2], rho_w = U[3];
        const Scalar inv_rho = 1.0 / rho;
        const Scalar ke = 0.5 * (rho_u*rho_u + rho_v*rho_v + rho_w*rho_w) * inv_rho * inv_rho;
        
        DenseMatrix<5, 1> grad_p;
        grad_p[0] = (gamma - 1.0) * ke;          // dp/drho
        grad_p[1] = -(gamma - 1.0) * rho_u * inv_rho; // dp/d(rho_u)
        grad_p[2] = -(gamma - 1.0) * rho_v * inv_rho; // dp/d(rho_v)
        grad_p[3] = -(gamma - 1.0) * rho_w * inv_rho; // dp/d(rho_w)
        grad_p[4] = (gamma - 1.0);                // dp/d(rho_E)
        return grad_p;
    }

    HostDevice __forceinline__
    Scalar get_gamma() const { return gamma; }
};


