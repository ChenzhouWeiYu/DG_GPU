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
    const DenseMatrix<NEQN, 1>& U_avg,
    Scalar x, Scalar y, Scalar z) {
    
    const auto basis = DGBasisEvaluator<Order>::eval_all(x, y, z);
    DenseMatrix<NEQN, 1> U_gp;
    #pragma unroll
    for (uInt k = 0; k < NEQN; ++k) {
        U_gp[k] = 0.0;
        #pragma unroll
        for (uInt l = 0; l < NumBasis; ++l) {
            U_gp[k] += basis[l] * coef[NEQN*l + k];
        }
        
        // Scalar val = 0.0;
        // #pragma unroll
        // for (uInt l = 0; l < NumBasis; ++l)
        //     val += basis[l] * coef[5*l + k];
        // U_gp[k] = val;
    }

    // 内联压强计算
    // Scalar rho = U_gp[0];
    // Scalar ke = 0.5 * (U_gp[1]*U_gp[1] + U_gp[2]*U_gp[2] + U_gp[3]*U_gp[3]) / rho;
    // Scalar p = (physics.get_gamma() - 1.0) * (U_gp[4] - ke);
    Scalar p = physics.compute_pressure(U_gp);

    constexpr Scalar eps = 1e-8;
    if (p >= eps) return 1.0;

    // Scalar t_low = 0.0, t_high = 1.0;
    // for (int iter = 0; iter < 20; ++iter) {
    //     Scalar t_mid = 0.5 * (t_low + t_high);
    //     DenseMatrix<NEQN, 1> U_mid;
    //     #pragma unroll
    //     for (uInt k = 0; k < NEQN; ++k) {
    //         U_mid[k] = (1.0 - t_mid) * U_avg[k] + t_mid * U_gp[k];
    //     }

    //     // Scalar rho_mid = U_mid[0];
    //     // Scalar ke_mid = 0.5 * (U_mid[1]*U_mid[1] + U_mid[2]*U_mid[2] + U_mid[3]*U_mid[3]) / rho_mid;
    //     // Scalar p_mid = (physics.get_gamma() - 1.0) * (U_mid[4] - ke_mid);
    //     Scalar p_mid = physics.compute_pressure(U_mid);
    //     if (p_mid < eps) t_high = t_mid;
    //     else t_low = t_mid;
    //     if ((t_high-t_low<1e-5)||(p_mid*p_mid<1e-12)) break;
    // }
    // return t_low;

    // Newton 法求解 p((1-θ)*U_avg + θ*U_gp) = eps * 2.0 ，但是 > eps 就停机
    Scalar theta = 1.0;
    DenseMatrix<NEQN, 1> delta_U;
    #pragma unroll
    for (uInt k = 0; k < NEQN; ++k) {
        delta_U[k] = U_gp[k] - U_avg[k];
    }

    for (int iter = 0; iter < 50; ++iter) { // 最大 10 次迭代
        // 计算 U(θ)
        DenseMatrix<NEQN, 1> U_theta;
        #pragma unroll
        for (uInt k = 0; k < NEQN; ++k) {
            U_theta[k] = (1.0 - theta) * U_avg[k] + theta * U_gp[k];
        }

        // 计算 p(θ)
        Scalar p_theta = physics.compute_pressure(U_theta);
        
        // 物理停机条件：p > eps
        if (p_theta > eps ) {
            return theta;
        }

        // 计算方向导数 p'(θ) = ∇p(U_θ) · ΔU
        Scalar dp_dtheta = physics.compute_pressure_directional_derivative(U_theta, delta_U);
        
        // 防止除零
        if (fabs(dp_dtheta) < 1e-16) {
            break;
        }

        // Newton 更新
        Scalar theta_new = theta - (p_theta - eps * 2.0) / dp_dtheta;
        
        // // 限制 θ ∈ [0,1]
        // theta_new = fmax(0.0, fmin(1.0, theta_new));
        
        // // 防止振荡
        // if (fabs(theta_new - theta) < 1e-8) {
        //     theta = theta_new;
        //     break;
        // }
        
        theta = theta_new;
    }
    return 0.0;
}

// 辅助函数：计算单个采样点的熵限制因子
template<typename Physics, uInt Order, uInt NEQN, uInt NumBasis>
HostDevice inline Scalar compute_entropy_theta(
    const Physics& physics,
    const DenseMatrix<NEQN*NumBasis, 1>& coef,
    const DenseMatrix<NEQN, 1>& U_avg,
    Scalar s0,
    Scalar x, Scalar y, Scalar z) {
    
    const auto basis = DGBasisEvaluator<Order>::eval_all(x, y, z);
    DenseMatrix<NEQN, 1> U_gp;
    #pragma unroll
    for (uInt k = 0; k < NEQN; ++k) {
        U_gp[k] = 0.0;
        #pragma unroll
        for (uInt l = 0; l < NumBasis; ++l) {
            U_gp[k] += basis[l] * coef[NEQN*l + k];
        }
    }

    Scalar q_gp = physics.compute_q(U_gp, s0);
    constexpr Scalar eps = 1e-14;
    if (q_gp <= eps) return 1.0; // q <= 0 已满足



    // Scalar q_avg = physics.compute_q(U_avg, s0);
    // // q_avg < 0, q_gp > 0 → 分母 < 0
    // Scalar theta = q_avg / (q_avg - q_gp);
    // return fmax(0.0, fmin(1.0, theta)); // 安全边界
    // if (q_avg > 0.0) {
    //     // 异常，决不允许出现，必须退出程序
    //     printf("Error: q_avg > 0.0\n");
    //     printf("       q_avg = %lf, q_gp = %lf, p_gp = %lf, s = %lf, s0 = %lf\n", 
    //         q_avg, q_gp, physics.compute_pressure(U_gp), physics.compute_specific_entropy(U_gp), s0);
    // }
    // if (fmax(0.0, fmin(1.0, theta)) != theta){
    //     // 异常，决不允许出现，必须退出程序
    //     printf("Error: theta = %lf is not in [0,1]\n", theta);
    //     printf("       q_avg = %lf, q_gp = %lf, p_gp = %lf, s = %lf, s0 = %lf\n", 
    //         q_avg, q_gp, physics.compute_pressure(U_gp), physics.compute_specific_entropy(U_gp), s0);
    // }
    // return theta;


    // Newton 法求解 q((1-θ)*U_avg + θ*U_gp) = 0
    Scalar theta = 1.0;
    // DenseMatrix<NEQN, 1> delta_U;
    // #pragma unroll
    // for (uInt k = 0; k < NEQN; ++k) {
    //     delta_U[k] = U_gp[k] - U_avg[k];
    // }

    // for (int iter = 0; iter < 50; ++iter) {
    //     DenseMatrix<NEQN, 1> U_theta;
    //     #pragma unroll
    //     for (uInt k = 0; k < NEQN; ++k) {
    //         U_theta[k] = (1.0 - theta) * U_avg[k] + theta * U_gp[k];
    //     }

    //     Scalar q_theta = physics.compute_q(U_theta, s0);
    //     if (q_theta <= eps) return theta;

    //     Scalar dq_dtheta = physics.compute_q_directional_derivative(U_theta, delta_U, s0);
    //     // if (fabs(dq_dtheta) < 1e-16) break;

    //     Scalar theta_new = theta - q_theta / dq_dtheta;
    //     theta = fmax(0.0, fmin(1.0, theta_new));
    //     theta = theta_new;
    // }
    Scalar q_avg = physics.compute_q(U_avg, s0);
    // q_avg < 0, q_gp > 0 → 分母 < 0
    theta = q_avg / (q_avg - q_gp);
    return fmax(0.0, fmin(1.0, theta)); // 安全边界
    // return 0.0;
}



// 主核函数
template<typename Physics, uInt Order, uInt NumBasis, uInt NumSamples, typename QuadC, typename QuadF, uInt Level, uInt WithEntropy>
__global__ void apply_positivity_limiter_kernel(
    const MeshView mesh,
    DenseMatrix<Physics::NEQN*NumBasis,1>* U,
    const Physics physics, Scalar s0) {
    
    uInt cellId = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellId >= mesh.num_cells) return;

    constexpr uInt NEQN = Physics::NEQN;
    DenseMatrix<NEQN*NumBasis,1>& coef = U[cellId];
    constexpr Scalar eps = 1e-14;
    // std::array<std::array<Scalar, NumBasis>, QuadC::num_points> basis_table;
    // constexpr auto Qpoints = QuadC::get_points();
    // #pragma unroll
    // for (uInt xgi = 0; xgi < QuadC::num_points; ++xgi)
    //     basis_table[xgi] = DGBasisEvaluator<Order>::eval_all(Qpoints[xgi][0], Qpoints[xgi][1], Qpoints[xgi][2]);

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

                // Level >= 2 就是三个顶点的轮换
                rho_min = fmin(rho_min, compute_density_theta<Order, NEQN, NumBasis>(
                    coef, uv[0], uv[1], 1.0 - uv[0] - uv[1]));
                rho_min = fmin(rho_min, compute_density_theta<Order, NEQN, NumBasis>(
                    coef, 1.0 - uv[0] - uv[1], uv[0], uv[1]));
                rho_min = fmin(rho_min, compute_density_theta<Order, NEQN, NumBasis>(
                    coef, uv[1], 1.0 - uv[0] - uv[1], uv[0]));
                
                // Level >= 3 就交换 uv[0] 和 uv[1] 再算一遍
                if constexpr (Level >= 3)
                rho_min = fmin(rho_min, compute_density_theta<Order, NEQN, NumBasis>(
                    coef, uv[1], uv[0], 1.0 - uv[0] - uv[1]));
                if constexpr (Level >= 3)
                rho_min = fmin(rho_min, compute_density_theta<Order, NEQN, NumBasis>(
                    coef, 1.0 - uv[0] - uv[1], uv[1], uv[0]));
                if constexpr (Level >= 3)
                rho_min = fmin(rho_min, compute_density_theta<Order, NEQN, NumBasis>(
                    coef, uv[0], 1.0 - uv[0] - uv[1], uv[1]));
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

                if constexpr (Level >= 3)
                rho_min = fmin(rho_min, compute_density_theta<Order, NEQN, NumBasis>(
                    coef, 0.0, 1.0 - uv[0] - uv[1], uv[1]));
                if constexpr (Level >= 3)
                rho_min = fmin(rho_min, compute_density_theta<Order, NEQN, NumBasis>(
                    coef, 0.0, uv[1], 1.0 - uv[0] - uv[1]));
                if constexpr (Level >= 3)
                rho_min = fmin(rho_min, compute_density_theta<Order, NEQN, NumBasis>(
                    coef, 0.0, uv[0], uv[1]));
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

                if constexpr (Level >= 3)
                rho_min = fmin(rho_min, compute_density_theta<Order, NEQN, NumBasis>(
                    coef, uv[1], 0.0, 1.0 - uv[0] - uv[1]));
                if constexpr (Level >= 3)
                rho_min = fmin(rho_min, compute_density_theta<Order, NEQN, NumBasis>(
                    coef, 1.0 - uv[0] - uv[1], 0.0, uv[1]));
                if constexpr (Level >= 3)
                rho_min = fmin(rho_min, compute_density_theta<Order, NEQN, NumBasis>(
                    coef, uv[0], 0.0, uv[1]));
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

                if constexpr (Level >= 3)
                rho_min = fmin(rho_min, compute_density_theta<Order, NEQN, NumBasis>(
                    coef, uv[0], uv[1], 0.0));
                if constexpr (Level >= 3)
                rho_min = fmin(rho_min, compute_density_theta<Order, NEQN, NumBasis>(
                    coef, 1.0 - uv[0] - uv[1], uv[1], 0.0));
                if constexpr (Level >= 3)
                rho_min = fmin(rho_min, compute_density_theta<Order, NEQN, NumBasis>(
                    coef, uv[1], 1.0 - uv[0] - uv[1], 0.0));
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
        DenseMatrix<NEQN, 1> U_avg;
        #pragma unroll
        for (uInt k = 0; k < NEQN; ++k) {
            U_avg[k] = coef[NEQN*0 + k];
        }
        Scalar p_avg = physics.compute_pressure(U_avg);

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

        // Level 1: 4个顶点
        if constexpr (Level >= 1) {
            const vector3f vertices[4] = {
                {0.0, 0.0, 0.0},
                {1.0, 0.0, 0.0},
                {0.0, 1.0, 0.0},
                {0.0, 0.0, 1.0}
            };
            for (uInt v = 0; v < 4; ++v) {
                Scalar theta = compute_pressure_theta<Physics, Order, NEQN, NumBasis>(
                    physics, coef, U_avg, 
                    vertices[v][0], vertices[v][1], vertices[v][2]);
                theta_p = fmin(theta_p, theta);
            }
        }

        // Level 2: 面积分点（4面×3轮换）
        if constexpr (Level >= 2) {
            constexpr auto face_points = QuadF::get_points();
            
            // 面0: (v1,v2,v3)
            for (uInt i = 0; i < QuadF::num_points; ++i) {
                const auto& uv = face_points[i];

                theta_p = fmin(theta_p, compute_pressure_theta<Physics, Order, NEQN, NumBasis>(
                    physics, coef, U_avg, uv[0], uv[1], 1.0 - uv[0] - uv[1]));
                theta_p = fmin(theta_p, compute_pressure_theta<Physics, Order, NEQN, NumBasis>(
                    physics, coef, U_avg, 1.0 - uv[0] - uv[1], uv[0], uv[1]));
                theta_p = fmin(theta_p, compute_pressure_theta<Physics, Order, NEQN, NumBasis>(
                    physics, coef, U_avg, uv[1], 1.0 - uv[0] - uv[1], uv[0]));
                
                if constexpr (Level >= 3)
                theta_p = fmin(theta_p, compute_pressure_theta<Physics, Order, NEQN, NumBasis>(
                    physics, coef, U_avg, uv[1], uv[0], 1.0 - uv[0] - uv[1]));
                if constexpr (Level >= 3)
                theta_p = fmin(theta_p, compute_pressure_theta<Physics, Order, NEQN, NumBasis>(
                    physics, coef, U_avg, 1.0 - uv[0] - uv[1], uv[1], uv[0]));
                if constexpr (Level >= 3)
                theta_p = fmin(theta_p, compute_pressure_theta<Physics, Order, NEQN, NumBasis>(
                    physics, coef, U_avg, uv[0], 1.0 - uv[0] - uv[1], uv[1]));
                
            }
            
            // 面1: (v0,v3,v2)
            for (uInt i = 0; i < QuadF::num_points; ++i) {
                const auto& uv = face_points[i];

                theta_p = fmin(theta_p, compute_pressure_theta<Physics, Order, NEQN, NumBasis>(
                    physics, coef, U_avg, 0.0, 1.0 - uv[0] - uv[1], uv[0]));
                theta_p = fmin(theta_p, compute_pressure_theta<Physics, Order, NEQN, NumBasis>(
                    physics, coef, U_avg, 0.0, uv[0], 1.0 - uv[0] - uv[1]));
                theta_p = fmin(theta_p, compute_pressure_theta<Physics, Order, NEQN, NumBasis>(
                    physics, coef, U_avg, 0.0, uv[1], uv[0]));

                if constexpr (Level >= 3)
                theta_p = fmin(theta_p, compute_pressure_theta<Physics, Order, NEQN, NumBasis>(
                    physics, coef, U_avg, 0.0, 1.0 - uv[0] - uv[1], uv[1]));
                if constexpr (Level >= 3)
                theta_p = fmin(theta_p, compute_pressure_theta<Physics, Order, NEQN, NumBasis>(
                    physics, coef, U_avg, 0.0, uv[1], 1.0 - uv[0] - uv[1]));
                if constexpr (Level >= 3)
                theta_p = fmin(theta_p, compute_pressure_theta<Physics, Order, NEQN, NumBasis>(
                    physics, coef, U_avg, 0.0, uv[0], uv[1]));
                
            }
            
            // 面2: (v0,v1,v3)
            for (uInt i = 0; i < QuadF::num_points; ++i) {
                const auto& uv = face_points[i];

                theta_p = fmin(theta_p, compute_pressure_theta<Physics, Order, NEQN, NumBasis>(
                    physics, coef, U_avg, uv[0], 0.0, 1.0 - uv[0] - uv[1]));
                theta_p = fmin(theta_p, compute_pressure_theta<Physics, Order, NEQN, NumBasis>(
                    physics, coef, U_avg, 1.0 - uv[0] - uv[1], 0.0, uv[0]));
                theta_p = fmin(theta_p, compute_pressure_theta<Physics, Order, NEQN, NumBasis>(
                    physics, coef, U_avg, uv[1], 0.0, uv[0]));

                if constexpr (Level >= 3)
                theta_p = fmin(theta_p, compute_pressure_theta<Physics, Order, NEQN, NumBasis>(
                    physics, coef, U_avg, uv[1], 0.0, 1.0 - uv[0] - uv[1]));
                if constexpr (Level >= 3)
                theta_p = fmin(theta_p, compute_pressure_theta<Physics, Order, NEQN, NumBasis>(
                    physics, coef, U_avg, 1.0 - uv[0] - uv[1], 0.0, uv[1]));
                if constexpr (Level >= 3)
                theta_p = fmin(theta_p, compute_pressure_theta<Physics, Order, NEQN, NumBasis>(
                    physics, coef, U_avg, uv[0], 0.0, uv[1]));
            }
            
            // 面3: (v0,v2,v1)
            for (uInt i = 0; i < QuadF::num_points; ++i) {
                const auto& uv = face_points[i];
                
                theta_p = fmin(theta_p, compute_pressure_theta<Physics, Order, NEQN, NumBasis>(
                    physics, coef, U_avg, uv[1], uv[0], 0.0));
                theta_p = fmin(theta_p, compute_pressure_theta<Physics, Order, NEQN, NumBasis>(
                    physics, coef, U_avg, 1.0 - uv[0] - uv[1], uv[0], 0.0));
                theta_p = fmin(theta_p, compute_pressure_theta<Physics, Order, NEQN, NumBasis>(
                    physics, coef, U_avg, uv[0], 1.0 - uv[0] - uv[1], 0.0));

                if constexpr (Level >= 3)
                theta_p = fmin(theta_p, compute_pressure_theta<Physics, Order, NEQN, NumBasis>(
                    physics, coef, U_avg, uv[0], uv[1], 0.0));
                if constexpr (Level >= 3)
                theta_p = fmin(theta_p, compute_pressure_theta<Physics, Order, NEQN, NumBasis>(
                    physics, coef, U_avg, 1.0 - uv[0] - uv[1], uv[1], 0.0));
                if constexpr (Level >= 3)
                theta_p = fmin(theta_p, compute_pressure_theta<Physics, Order, NEQN, NumBasis>(
                    physics, coef, U_avg, uv[1], 1.0 - uv[0] - uv[1], 0.0));
            }
        }

        // 应用压强限制
        #pragma unroll
        for (uInt k = 0; k < NEQN; ++k) {
            #pragma unroll
            for (uInt l = 1; l < NumBasis; ++l) {
                coef[NEQN*l + k] *= theta_p;
            }
        }
    }


    // ---------------- 熵增限制 ----------------
    if constexpr (WithEntropy)
    {
        Scalar theta_q = 1.0;
        DenseMatrix<NEQN, 1> U_avg;
        #pragma unroll
        for (uInt k = 0; k < NEQN; ++k) {
            U_avg[k] = coef[NEQN*0 + k];
        }

        // Level 0: 体积分点
        if constexpr (Level >= 0) {
            constexpr auto vol_points = QuadC::get_points();
            for (uInt i = 0; i < QuadC::num_points; ++i) {
                Scalar theta = compute_entropy_theta<Physics, Order, NEQN, NumBasis>(
                    physics, coef, U_avg, s0,
                    vol_points[i][0], vol_points[i][1], vol_points[i][2]);
                theta_q = fmin(theta_q, theta);
            }
        }

        // Level 1: 4个顶点
        if constexpr (Level >= 1) {
            const vector3f vertices[4] = {{0,0,0}, {1,0,0}, {0,1,0}, {0,0,1}};
            for (uInt v = 0; v < 4; ++v) {
                Scalar theta = compute_entropy_theta<Physics, Order, NEQN, NumBasis>(
                    physics, coef, U_avg, s0,
                    vertices[v][0], vertices[v][1], vertices[v][2]);
                theta_q = fmin(theta_q, theta);
            }
        }

        // Level 2: 面积分点（同压强部分）
        if constexpr (Level >= 2) {
            // ... 类似压强部分的 4 面 × 3 轮换 ...
        }

        // 应用熵限制
        #pragma unroll
        for (uInt k = 0; k < NEQN; ++k) {
            #pragma unroll
            for (uInt l = 1; l < NumBasis; ++l) {
                coef[NEQN*l + k] *= theta_q;
            }
        }
    }
}



// entropy_init_kernels.cuh
template<typename Physics, uInt NEQN, uInt NumBasis>
__global__ void compute_initial_s0_kernel(
    const DenseMatrix<NEQN*NumBasis, 1>* U,
    Scalar* s0_buffer,
    uInt num_cells,
    const Physics physics) {
    
    extern __shared__ Scalar shared_s0[];
    uInt tid = threadIdx.x;
    uInt gid = blockIdx.x * blockDim.x + threadIdx.x;

    Scalar local_s0 = 1e30; // 初始化为极大值
    if (gid < num_cells) {
        // 提取单元平均值（常数模）
        DenseMatrix<NEQN, 1> U_avg;
        for (uInt k = 0; k < NEQN; ++k) {
            U_avg[k] = U[gid](0,k); // 常数模 = 第0个基函数系数
        }
        local_s0 = physics.compute_specific_entropy(U_avg);
    }

    // Block 内归约
    shared_s0[tid] = local_s0;
    __syncthreads();
    for (uInt stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_s0[tid] = fmin(shared_s0[tid], shared_s0[tid + stride]);
        }
        __syncthreads();
    }

    // 第一个 thread 写入全局 buffer
    if (tid == 0) {
        s0_buffer[blockIdx.x] = shared_s0[0];
    }
}




#include "dg/dg_limiters/positive_preserving_limiters/sampling_points.h"

template<typename Physics, uInt Order, uInt NumBasis, uInt NumSamples, typename QuadC, typename QuadF, uInt Level>
__global__ void apply_positivity_limiter_kernel_table(
    const MeshView mesh,
    DenseMatrix<Physics::NEQN*NumBasis,1>* U,
    const Physics physics) {

    constexpr auto basis_table = SamplingPoints<Order, QuadC, QuadF, Level>::basis_table;

    uInt cellId = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellId >= mesh.num_cells) return;

    constexpr uInt NEQN = Physics::NEQN;
    
    DenseMatrix<NEQN*NumBasis,1>& coef = U[cellId];

    // ---------------- 保正密度 ----------------
    {
        constexpr Scalar eps = 1e-14;
        coef[0] = fmax(coef[0], eps); // 常数模
        Scalar rho_avg = coef[0];
        Scalar rho_min = rho_avg;

        // 遍历所有采样点
        for (uInt s = 0; s < NumSamples; ++s) {
            const auto& basis = basis_table[s];
            Scalar rho = 0;
            #pragma unroll
            for (uInt l = 0; l < NumBasis; ++l) rho += basis[l] * coef[NEQN*l + 0];
            rho_min = fmin(rho_min, rho);
        }

        if (rho_min < eps) {
            Scalar theta = (rho_avg - eps) / fmax(rho_avg - rho_min, 1e-32);
            theta = fmin(theta, 1.0);
            #pragma unroll
            for (uInt l = 1; l < NumBasis; ++l) coef[NEQN*l + 0] *= theta;
        }
    }

    // ---------------- 保正压强 ----------------
    {
        Scalar theta_p = 1.0;
        DenseMatrix<NEQN,1> U_avg;
        #pragma unroll
        for (uInt k = 0; k < NEQN; ++k) U_avg[k] = coef[NEQN*0 + k];

        for (uInt s = 0; s < NumSamples; ++s) {
            const auto& basis = basis_table[s];
            DenseMatrix<NEQN,1> U_gp;
            #pragma unroll
            for (uInt k = 0; k < NEQN; ++k) {
                U_gp[k] = 0.0;
                #pragma unroll
                for (uInt l = 0; l < NumBasis; ++l) {
                    U_gp[k] += basis[l] * coef[NEQN*l + k];
                }
            }

            // 内联压强计算
            // Scalar rho = U_gp[0];
            // Scalar ke = 0.5 * (U_gp[1]*U_gp[1] + U_gp[2]*U_gp[2] + U_gp[3]*U_gp[3]) / rho;
            // Scalar p = (physics.get_gamma() - 1.0) * (U_gp[4] - ke);
            Scalar p = physics.compute_pressure(U_gp);

            constexpr Scalar eps = 1e-14;
            if (p >= eps) continue;

            Scalar t_low = 0.0, t_high = 1.0;
            #pragma unroll
            for (int iter = 0; iter < 20; ++iter) {
                Scalar t_mid = 0.5 * (t_low + t_high);
                DenseMatrix<NEQN,1> U_mid;
                #pragma unroll
                for (uInt k = 0; k < NEQN; ++k) {
                    U_mid[k] = (1.0 - t_mid) * U_avg[k] + t_mid * U_gp[k];
                }

                // Scalar rho_mid = U_mid[0];
                // Scalar ke_mid = 0.5 * (U_mid[1]*U_mid[1] + U_mid[2]*U_mid[2] + U_mid[3]*U_mid[3]) / rho_mid;
                // Scalar p_mid = (physics.get_gamma() - 1.0) * (U_mid[4] - ke_mid);
                Scalar p_mid = physics.compute_pressure(U_mid);

                if (p_mid < eps) t_high = t_mid;
                else t_low = t_mid;
                if ((t_high-t_low<1e-5)||(p_mid*p_mid<1e-12)) break;
            }
            theta_p = fmin(theta_p, t_low);
        }

        #pragma unroll
        for (uInt k = 0; k < NEQN; ++k)
            #pragma unroll
            for (uInt l = 1; l < NumBasis; ++l)
                coef[NEQN*l + k] *= theta_p;
    }
}