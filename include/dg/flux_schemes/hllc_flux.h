#pragma once
#include "dg/flux_schemes/rotated_flux_scheme.h"

template<typename Physics>
class HLLCFlux : public RotatedFluxScheme<HLLCFlux<Physics>, Physics> {
public:
    using Base = RotatedFluxScheme<HLLCFlux<Physics>, Physics>;

    HostDevice __forceinline__
    static DenseMatrix<5, 1> compute_1d(
        const Physics& physics,
        const DenseMatrix<5, 1>& U_L,
        const DenseMatrix<5, 1>& U_R) {
        
        // 局部坐标系下，法向 = x 轴
        Scalar rho_L = Base::positive(U_L[0]), rho_R = Base::positive(U_R[0]);
        Scalar u_L = U_L[1] / rho_L, u_R = U_R[1] / rho_R;
        Scalar a_L = physics.compute_sound_speed(U_L);
        Scalar a_R = physics.compute_sound_speed(U_R);
        Scalar p_L = physics.compute_pressure(U_L);
        Scalar p_R = physics.compute_pressure(U_R);

        Scalar S_L = fmin(u_L - a_L, u_R - a_R);
        Scalar S_R = fmax(u_L + a_L, u_R + a_R);
        Scalar S_M = (p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R)) /
                     (rho_L * (S_L - u_L) - rho_R * (S_R - u_R));

        auto F_L = physics.compute_flux_1d(U_L);
        auto F_R = physics.compute_flux_1d(U_R);

        DenseMatrix<5, 1> flux;
        if (S_L >= 0.0) {
            flux = F_L;
        } else if (S_M >= 0.0) {
            flux = F_L + S_L * (compute_U_star(U_L, S_L, S_M, u_L, p_L, physics) - U_L);
        } else if (S_R >= 0.0) {
            flux = F_R + S_R * (compute_U_star(U_R, S_R, S_M, u_R, p_R, physics) - U_R);
        } else {
            flux = F_R;
        }
        return flux;
    }

private:
    HostDevice __forceinline__
    static DenseMatrix<5, 1> compute_U_star(
        const DenseMatrix<5, 1>& U,
        Scalar S,
        Scalar S_M,
        Scalar u,
        Scalar p,
        const Physics& physics) {
        Scalar rho = U[0];
        Scalar coef = rho * (S - u) / (S - S_M);
        DenseMatrix<5, 1> U_star;
        U_star[0] = coef;
        U_star[1] = coef * S_M;
        U_star[2] = coef * (U[2] / rho); // 切向速度不变
        U_star[3] = coef * (U[3] / rho);
        U_star[4] = coef * (physics.compute_pressure(U_star) / (physics.get_gamma() - 1.0) + 0.5 * S_M * S_M) +
                    (U[4] - rho * (p / (rho * (physics.get_gamma() - 1.0)) + 0.5 * u * u));
        return U_star;
    }
};