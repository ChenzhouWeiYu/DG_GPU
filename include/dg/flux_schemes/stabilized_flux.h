// stabilized_flux.h
#pragma once
#include "dg/flux_schemes/flux_scheme_base.h"
#include "dg/flux_schemes/rotated_flux_scheme.h"


template<typename FluxScheme>
class StabilizedFlux : public RotatedFluxScheme<StabilizedFlux<FluxScheme>, typename FluxScheme::PhysicsType> {
public:
    using Physics = typename FluxScheme::PhysicsType;
    using Base = RotatedFluxScheme<StabilizedFlux<FluxScheme>, Physics>;
    using Base::NEQN;

    // 关键：compute_1d_impl 添加稳定化项
    HostDevice __forceinline__
    static DenseMatrix<NEQN, 1> compute_1d(
        const Physics& physics,
        const DenseMatrix<NEQN, 1>& U_L,
        const DenseMatrix<NEQN, 1>& U_R) {
        
        // 1. 计算基础一维通量
        DenseMatrix<NEQN, 1> base_flux = FluxScheme::compute_1d(physics, U_L, U_R);
        
        // 2. 计算稳定化粘性项
        DenseMatrix<NEQN, 1> viscosity = compute_stabilization_viscosity(physics, U_L, U_R);
        
        // 3. 返回增强通量
        for (uInt i = 0; i < NEQN; ++i) {
            base_flux[i] += viscosity[i];
        }
        return base_flux;
    }

private:
    HostDevice __forceinline__
    static DenseMatrix<NEQN, 1> compute_stabilization_viscosity(
        const Physics& physics,
        const DenseMatrix<NEQN, 1>& U_L,
        const DenseMatrix<NEQN, 1>& U_R) {
        
        Scalar gamma = physics.get_gamma();
        Scalar rho_L = Base::positive(U_L[0]), rho_R = Base::positive(U_R[0]);
        Scalar u_L = U_L[1] / rho_L, v_L = U_L[2] / rho_L, w_L = U_L[3] / rho_L;
        Scalar u_R = U_R[1] / rho_R, v_R = U_R[2] / rho_R, w_R = U_R[3] / rho_R;
        Scalar p_L = physics.compute_pressure(U_L), p_R = physics.compute_pressure(U_R);
        Scalar H_L = (U_L[4] + p_L) / rho_L, H_R = (U_R[4] + p_R) / rho_R;
        Scalar a_L = sqrt(fmax(0.0, (gamma - 1.0) * (H_L - 0.5*(u_L*u_L + v_L*v_L + w_L*w_L))));
        Scalar a_R = sqrt(fmax(0.0, (gamma - 1.0) * (H_R - 0.5*(u_R*u_R + v_R*v_R + w_R*w_R))));

        Scalar sqrt_rho_L = sqrt(rho_L);
        Scalar sqrt_rho_R = sqrt(rho_R);
        Scalar inv_sum_sqrt_rho = 1.0 / (sqrt_rho_L + sqrt_rho_R);

        Scalar rho_tilde = sqrt_rho_L * sqrt_rho_R;
        Scalar u_tilde = (sqrt_rho_L * u_L + sqrt_rho_R * u_R) * inv_sum_sqrt_rho;
        Scalar v_tilde = (sqrt_rho_L * v_L + sqrt_rho_R * v_R) * inv_sum_sqrt_rho;
        Scalar w_tilde = (sqrt_rho_L * w_L + sqrt_rho_R * w_R) * inv_sum_sqrt_rho;
        Scalar H_tilde = (sqrt_rho_L * H_L + sqrt_rho_R * H_R) * inv_sum_sqrt_rho;
        Scalar a_tilde = sqrt(fmax(0.0, (gamma - 1.0) * (H_tilde - 0.5*(u_tilde*u_tilde + v_tilde*v_tilde + w_tilde*w_tilde))));

        Scalar delta_rho = U_R[0] - U_L[0];
        Scalar delta_p   = p_R - p_L;
        Scalar delta_v   = v_R - v_L;
        Scalar delta_w   = w_R - w_L;

        Scalar h = fmax(fmin(p_L / p_R, p_R / p_L), 0.0);
        Scalar g = 1.0 - h * h * h;

        Scalar Mach_L = u_L / a_L;
        Scalar Mach_R = u_R / a_R;
        Scalar Mach_tilde = u_tilde / a_tilde;
        // Scalar phi = fmax(fmax(0.0, 1.0 - fabs(Mach_tilde)), fmax(1.0 - fabs(Mach_L), 1.0 - fabs(Mach_R)));
        Scalar phi = std::max({0.0, 1.0 - std::abs(Mach_tilde), 1.0 - std::abs(Mach_L), 1.0 - std::abs(Mach_R)});

        Scalar coeff_EV  = -g * phi * 0.5 * a_tilde * (delta_rho - delta_p / (a_tilde * a_tilde));
        Scalar coeff_SVy = -g * phi * 0.5 * rho_tilde * a_tilde * delta_v;
        Scalar coeff_SVz = -g * phi * 0.5 * rho_tilde * a_tilde * delta_w;

        DenseMatrix<NEQN, 1> viscosity;
        viscosity[0] = coeff_EV;
        viscosity[1] = coeff_EV * u_tilde;
        viscosity[2] = coeff_EV * v_tilde + coeff_SVy;
        viscosity[3] = coeff_EV * w_tilde + coeff_SVz;
        viscosity[4] = coeff_EV * 0.5*(u_tilde*u_tilde + v_tilde*v_tilde + w_tilde*w_tilde) + 
                      coeff_SVy * v_tilde + coeff_SVz * w_tilde;
        return 0.0*viscosity;
    }
};