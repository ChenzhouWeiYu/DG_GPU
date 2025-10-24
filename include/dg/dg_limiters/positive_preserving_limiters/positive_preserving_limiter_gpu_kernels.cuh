// include/dg/dg_schemes/positive_preserving_limiters/positive_preserving_limiter_gpu_kernels.h
#pragma once
#include "base/type.h"
#include "matrix/dense_matrix.h"
#include "mesh/device_mesh.cuh"
#include "dg/dg_basis/dg_basis.h"

// 辅助函数：计算单个采样点的密度限制因子
template<uInt Order, uInt NEQN, uInt NumBasis>
HostDevice inline Scalar compute_density_theta(
    const DenseMatrix<NEQN*NumBasis, 1>& coef,
    Scalar x, Scalar y, Scalar z) {
    
    const auto basis = DGBasisEvaluator<Order>::eval_all(x, y, z);
    Scalar rho = 0.0;
    #pragma unroll
    for (uInt l = 0; l < NumBasis; ++l) {
        rho += basis[l] * coef[NEQN*l + 0];
    }
    return rho;
}

// 辅助函数：计算单个采样点的压强限制因子
template<typename Physics, uInt Order, uInt NEQN, uInt NumBasis>
HostDevice inline Scalar compute_pressure_theta(
    const Physics& physics,
    const DenseMatrix<NEQN*NumBasis, 1>& coef,
    const Scalar U_avg[NEQN],
    Scalar x, Scalar y, Scalar z) {
    
    const auto basis = DGBasisEvaluator<Order>::eval_all(x, y, z);
    Scalar U_gp[NEQN];
    #pragma unroll
    for (uInt k = 0; k < NEQN; ++k) {
        U_gp[k] = 0.0;
        #pragma unroll
        for (uInt l = 0; l < NumBasis; ++l) {
            U_gp[k] += basis[l] * coef[NEQN*l + k];
        }
    }

    // 内联压强计算
    Scalar rho = U_gp[0];
    Scalar u = U_gp[1] / rho, v = U_gp[2] / rho, w = U_gp[3] / rho;
    Scalar E = U_gp[4] / rho;
    Scalar ke = 0.5 * (u*u + v*v + w*w);
    Scalar p = (physics.get_gamma() - 1.0) * rho * (E - ke);

    constexpr Scalar eps = 1e-14;
    if (p >= eps) return 1.0;

    // 固定5次二分法（无break，确保warp一致性）
    Scalar t_low = 0.0, t_high = 1.0;
    for (int iter = 0; iter < 20; ++iter) {
        Scalar t_mid = 0.5 * (t_low + t_high);
        Scalar U_mid[NEQN];
        #pragma unroll
        for (uInt k = 0; k < NEQN; ++k) {
            U_mid[k] = (1.0 - t_mid) * U_avg[k] + t_mid * U_gp[k];
        }

        Scalar rho_mid = U_mid[0];
        Scalar u_mid = U_mid[1] / rho_mid, v_mid = U_mid[2] / rho_mid, w_mid = U_mid[3] / rho_mid;
        Scalar E_mid = U_mid[4] / rho_mid;
        Scalar ke_mid = 0.5 * (u_mid*u_mid + v_mid*v_mid + w_mid*w_mid);
        Scalar p_mid = (physics.get_gamma() - 1.0) * rho_mid * (E_mid - ke_mid);

        if (p_mid < eps) t_high = t_mid;
        else t_low = t_mid;
    }
    return t_low;
}

// 主核函数
template<typename Physics, uInt Order, uInt NumBasis, uInt NumSamples, typename QuadC, typename QuadF, uInt Level>
__global__ void apply_positivity_limiter_kernel(
    const MeshView mesh,
    DenseMatrix<Physics::NEQN*NumBasis,1>* U,
    const Physics physics) {
    
    uInt cellId = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellId >= mesh.num_cells) return;

    constexpr uInt NEQN = Physics::NEQN;
    DenseMatrix<NEQN*NumBasis,1>& coef = U[cellId];
    constexpr Scalar eps = 1e-14;

    // ---------------- 保正密度 ----------------
    {
        // 确保常数模密度为正
        coef[0] = fmax(coef[0], eps);
        Scalar rho_avg = coef[0];
        Scalar rho_min = rho_avg;

        // Level 0: 体积分点
        if constexpr (Level >= 0) {
            constexpr auto vol_points = QuadC::get_points();
            for (uInt i = 0; i < QuadC::num_points; ++i) {
                Scalar rho = compute_density_theta<Order, NEQN, NumBasis>(
                    coef, vol_points[i][0], vol_points[i][1], vol_points[i][2]);
                rho_min = fmin(rho_min, rho);
            }
        }

        // Level 1: 4个顶点
        if constexpr (Level >= 1) {
            const vector3f vertices[4] = {
                {0.0, 0.0, 0.0},
                {1.0, 0.0, 0.0},
                {0.0, 1.0, 0.0},
                {0.0, 0.0, 1.0}
            };
            for (uInt v = 0; v < 4; ++v) {
                Scalar rho = compute_density_theta<Order, NEQN, NumBasis>(
                    coef, vertices[v][0], vertices[v][1], vertices[v][2]);
                rho_min = fmin(rho_min, rho);
            }
        }

        // Level 2: 面积分点（4面×3轮换）
        if constexpr (Level >= 2) {
            constexpr auto face_points = QuadF::get_points();
            
            // 面0: (v1,v2,v3)
            for (uInt i = 0; i < QuadF::num_points; ++i) {
                const auto& uv = face_points[i];
                rho_min = fmin(rho_min, compute_density_theta<Order, NEQN, NumBasis>(
                    coef, uv[0], uv[1], 1.0 - uv[0] - uv[1]));
                rho_min = fmin(rho_min, compute_density_theta<Order, NEQN, NumBasis>(
                    coef, 1.0 - uv[0] - uv[1], uv[0], uv[1]));
                rho_min = fmin(rho_min, compute_density_theta<Order, NEQN, NumBasis>(
                    coef, uv[1], 1.0 - uv[0] - uv[1], uv[0]));
            }
            
            // 面1: (v0,v3,v2)
            for (uInt i = 0; i < QuadF::num_points; ++i) {
                const auto& uv = face_points[i];
                rho_min = fmin(rho_min, compute_density_theta<Order, NEQN, NumBasis>(
                    coef, 0.0, 1.0 - uv[0] - uv[1], uv[0]));
                rho_min = fmin(rho_min, compute_density_theta<Order, NEQN, NumBasis>(
                    coef, 0.0, uv[0], 1.0 - uv[0] - uv[1]));
                rho_min = fmin(rho_min, compute_density_theta<Order, NEQN, NumBasis>(
                    coef, 0.0, uv[1], uv[0]));
            }
            
            // 面2: (v0,v1,v3)
            for (uInt i = 0; i < QuadF::num_points; ++i) {
                const auto& uv = face_points[i];
                rho_min = fmin(rho_min, compute_density_theta<Order, NEQN, NumBasis>(
                    coef, uv[0], 0.0, 1.0 - uv[0] - uv[1]));
                rho_min = fmin(rho_min, compute_density_theta<Order, NEQN, NumBasis>(
                    coef, 1.0 - uv[0] - uv[1], 0.0, uv[0]));
                rho_min = fmin(rho_min, compute_density_theta<Order, NEQN, NumBasis>(
                    coef, uv[1], 0.0, uv[0]));
            }
            
            // 面3: (v0,v2,v1)
            for (uInt i = 0; i < QuadF::num_points; ++i) {
                const auto& uv = face_points[i];
                rho_min = fmin(rho_min, compute_density_theta<Order, NEQN, NumBasis>(
                    coef, uv[1], uv[0], 0.0));
                rho_min = fmin(rho_min, compute_density_theta<Order, NEQN, NumBasis>(
                    coef, 1.0 - uv[0] - uv[1], uv[0], 0.0));
                rho_min = fmin(rho_min, compute_density_theta<Order, NEQN, NumBasis>(
                    coef, uv[0], 1.0 - uv[0] - uv[1], 0.0));
            }
        }

        // 应用密度限制
        if (rho_min < eps) {
            Scalar theta = (rho_avg - eps) / fmax(rho_avg - rho_min, 1e-32);
            theta = fmin(theta, 1.0);
            #pragma unroll
            for (uInt l = 1; l < NumBasis; ++l) {
                coef[NEQN*l + 0] *= theta;
            }
        }
    }

    // ---------------- 保正压强 ----------------
    {
        Scalar theta_p = 1.0;
        Scalar U_avg[NEQN];
        #pragma unroll
        for (uInt k = 0; k < NEQN; ++k) {
            U_avg[k] = coef[NEQN*0 + k];
        }

        // Level 0: 体积分点
        if constexpr (Level >= 0) {
            constexpr auto vol_points = QuadC::get_points();
            for (uInt i = 0; i < QuadC::num_points; ++i) {
                Scalar theta = compute_pressure_theta<Physics, Order, NEQN, NumBasis>(
                    physics, coef, U_avg, 
                    vol_points[i][0], vol_points[i][1], vol_points[i][2]);
                theta_p = fmin(theta_p, theta);
            }
        }

        // // Level 1: 4个顶点
        // if constexpr (Level >= 1) {
        //     const vector3f vertices[4] = {
        //         {0.0, 0.0, 0.0},
        //         {1.0, 0.0, 0.0},
        //         {0.0, 1.0, 0.0},
        //         {0.0, 0.0, 1.0}
        //     };
        //     for (uInt v = 0; v < 4; ++v) {
        //         Scalar theta = compute_pressure_theta<Physics, Order, NEQN, NumBasis>(
        //             physics, coef, U_avg, 
        //             vertices[v][0], vertices[v][1], vertices[v][2]);
        //         theta_p = fmin(theta_p, theta);
        //     }
        // }

        // // Level 2: 面积分点（4面×3轮换）
        // if constexpr (Level >= 2) {
        //     constexpr auto face_points = QuadF::get_points();
            
        //     // 面0: (v1,v2,v3)
        //     for (uInt i = 0; i < QuadF::num_points; ++i) {
        //         const auto& uv = face_points[i];
        //         theta_p = fmin(theta_p, compute_pressure_theta<Physics, Order, NEQN, NumBasis>(
        //             physics, coef, U_avg, uv[0], uv[1], 1.0 - uv[0] - uv[1]));
        //         theta_p = fmin(theta_p, compute_pressure_theta<Physics, Order, NEQN, NumBasis>(
        //             physics, coef, U_avg, 1.0 - uv[0] - uv[1], uv[0], uv[1]));
        //         theta_p = fmin(theta_p, compute_pressure_theta<Physics, Order, NEQN, NumBasis>(
        //             physics, coef, U_avg, uv[1], 1.0 - uv[0] - uv[1], uv[0]));
        //     }
            
        //     // 面1: (v0,v3,v2)
        //     for (uInt i = 0; i < QuadF::num_points; ++i) {
        //         const auto& uv = face_points[i];
        //         theta_p = fmin(theta_p, compute_pressure_theta<Physics, Order, NEQN, NumBasis>(
        //             physics, coef, U_avg, 0.0, 1.0 - uv[0] - uv[1], uv[0]));
        //         theta_p = fmin(theta_p, compute_pressure_theta<Physics, Order, NEQN, NumBasis>(
        //             physics, coef, U_avg, 0.0, uv[0], 1.0 - uv[0] - uv[1]));
        //         theta_p = fmin(theta_p, compute_pressure_theta<Physics, Order, NEQN, NumBasis>(
        //             physics, coef, U_avg, 0.0, uv[1], uv[0]));
        //     }
            
        //     // 面2: (v0,v1,v3)
        //     for (uInt i = 0; i < QuadF::num_points; ++i) {
        //         const auto& uv = face_points[i];
        //         theta_p = fmin(theta_p, compute_pressure_theta<Physics, Order, NEQN, NumBasis>(
        //             physics, coef, U_avg, uv[0], 0.0, 1.0 - uv[0] - uv[1]));
        //         theta_p = fmin(theta_p, compute_pressure_theta<Physics, Order, NEQN, NumBasis>(
        //             physics, coef, U_avg, 1.0 - uv[0] - uv[1], 0.0, uv[0]));
        //         theta_p = fmin(theta_p, compute_pressure_theta<Physics, Order, NEQN, NumBasis>(
        //             physics, coef, U_avg, uv[1], 0.0, uv[0]));
        //     }
            
        //     // 面3: (v0,v2,v1)
        //     for (uInt i = 0; i < QuadF::num_points; ++i) {
        //         const auto& uv = face_points[i];
        //         theta_p = fmin(theta_p, compute_pressure_theta<Physics, Order, NEQN, NumBasis>(
        //             physics, coef, U_avg, uv[1], uv[0], 0.0));
        //         theta_p = fmin(theta_p, compute_pressure_theta<Physics, Order, NEQN, NumBasis>(
        //             physics, coef, U_avg, 1.0 - uv[0] - uv[1], uv[0], 0.0));
        //         theta_p = fmin(theta_p, compute_pressure_theta<Physics, Order, NEQN, NumBasis>(
        //             physics, coef, U_avg, uv[0], 1.0 - uv[0] - uv[1], 0.0));
        //     }
        // }

        // 应用压强限制
        #pragma unroll
        for (uInt k = 0; k < NEQN; ++k) {
            #pragma unroll
            for (uInt l = 1; l < NumBasis; ++l) {
                coef[NEQN*l + k] *= theta_p;
            }
        }
    }
}