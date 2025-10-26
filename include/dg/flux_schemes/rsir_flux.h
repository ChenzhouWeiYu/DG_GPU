#pragma once
#include "dg/flux_schemes/rotated_flux_scheme.h"

template<typename Physics>
class RSIRFlux : public RotatedFluxScheme<RSIRFlux<Physics>, Physics> {
public:
    using Base = RotatedFluxScheme<RSIRFlux<Physics>, Physics>;

    HostDevice __forceinline__
    static DenseMatrix<5, 1> compute_1d(
        const Physics& physics,
        const DenseMatrix<5, 1>& U_L,
        const DenseMatrix<5, 1>& U_R) {
        
        Scalar gamma = physics.get_gamma();
        Scalar rho_L = Base::positive(U_L[0]), rho_R = Base::positive(U_R[0]);
        Scalar u_L = U_L[1] / rho_L, u_R = U_R[1] / rho_R;
        Scalar p_L = physics.compute_pressure(U_L);
        Scalar p_R = physics.compute_pressure(U_R);
        Scalar H_L = (U_L[4] + p_L) / rho_L;
        Scalar H_R = (U_R[4] + p_R) / rho_R;

        // ========== 1. Roe 平均 ==========
        Scalar sqrt_rho_L = sqrt(rho_L);
        Scalar sqrt_rho_R = sqrt(rho_R);
        Scalar u_tilde = (sqrt_rho_L * u_L + sqrt_rho_R * u_R) / (sqrt_rho_L + sqrt_rho_R);
        Scalar H_tilde = (sqrt_rho_L * H_L + sqrt_rho_R * H_R) / (sqrt_rho_L + sqrt_rho_R);
        Scalar a_tilde = sqrt(fmax(0.0, (gamma - 1.0) * (H_tilde - 0.5 * u_tilde * u_tilde)));

        // ========== 2. 波速估计 ==========
        Scalar S_L = fmin(u_L - a_tilde, u_tilde - a_tilde);
        Scalar S_R = fmax(u_R + a_tilde, u_tilde + a_tilde);
        if (S_L >= 0.0) return physics.compute_flux_1d(U_L);
        if (S_R <= 0.0) return physics.compute_flux_1d(U_R);

        // ========== 3. HLL 中间状态 U* ==========
        auto F_L = physics.compute_flux_1d(U_L);
        auto F_R = physics.compute_flux_1d(U_R);
        DenseMatrix<5, 1> U_star_hll;
        Scalar denom_hll = 1.0 / (S_R - S_L);
        for (uInt i = 0; i < 5; ++i) {
            U_star_hll[i] = (S_R * U_L[i] - S_L * U_R[i] - F_R[i] + F_L[i]) * denom_hll;
        }
        Scalar S_M = U_star_hll[1] / U_star_hll[0]; // u* = (rho u)* / rho*

        // ========== 4. 一致性权重 ==========
        Scalar omega_R = (S_R - S_M) / (S_R - S_L);
        Scalar omega_L = (S_M - S_L) / (S_R - S_L);

        // ========== 5. 跳跃向量 Ψ (热力学启发式) ==========
        Scalar c2_L = gamma * p_L / rho_L;
        Scalar c2_R = gamma * p_R / rho_L;
        Scalar c2_bar = fmax(c2_L, c2_R); // 确保熵增

        // 质量跳跃
        Scalar delta_rho = rho_R - rho_L + (p_L - p_R) / c2_bar;
        // 动量跳跃
        Scalar delta_rho_u = delta_rho * S_M;
        // 能量跳跃
        Scalar v_L = 1.0 / rho_L, v_R = 1.0 / rho_R;
        Scalar e_L = p_L / ((gamma - 1.0) * rho_L);
        Scalar e_R = p_R / ((gamma - 1.0) * rho_R);
        Scalar p_star = p_L + c2_bar * (U_star_hll[0] - rho_L); // p* = p_L + c^2 (rho* - rho_L)
        Scalar e_L_star = e_L - p_star * (1.0 / U_star_hll[0] - v_L);
        Scalar e_R_star = e_R - p_star * (1.0 / U_star_hll[0] - v_R);
        Scalar delta_rho_E = U_star_hll[0] * (e_R_star + 0.5 * S_M * S_M) - U_star_hll[0] * (e_L_star + 0.5 * S_M * S_M);

        DenseMatrix<5, 1> Psi;
        Psi[0] = delta_rho;
        Psi[1] = delta_rho_u;
        Psi[2] = 0.0; // 切向动量不变（1D）
        Psi[3] = 0.0;
        Psi[4] = delta_rho_E;

        // ========== 6. 重构两个中间状态 ==========
        DenseMatrix<5, 1> U_star_L, U_star_R;
        for (uInt i = 0; i < 5; ++i) {
            U_star_L[i] = U_star_hll[i] - omega_R * Psi[i];
            U_star_R[i] = U_star_hll[i] + omega_L * Psi[i];
        }

        // ========== 7. 选择通量 ==========
        if (S_M >= 0.0) {
            // 使用左中间状态
            return physics.compute_flux_1d(U_star_L);
        } else {
            // 使用右中间状态
            return physics.compute_flux_1d(U_star_R);
        }
    }
};