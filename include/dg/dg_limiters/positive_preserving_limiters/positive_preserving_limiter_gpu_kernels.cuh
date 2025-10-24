// include/dg/dg_schemes/positive_preserving_limiters/positive_preserving_limiter_gpu_kernels.h
#pragma once

#include "base/type.h"
#include "matrix/dense_matrix.h"
#include "mesh/device_mesh.cuh"
#include "dg/dg_basis/dg_basis.h"
#include "dg/dg_limiters/positive_preserving_limiters/sampling_points.h"

template<typename Physics, uInt Order, uInt NumBasis, uInt NumSamples, typename QuadC, typename QuadF, uInt Level>
__global__ void apply_positivity_limiter_kernel(
    const MeshView mesh,
    DenseMatrix<Physics::NEQN*NumBasis,1>* U,
    const Physics physics) {

    // // 声明 shared memory
    // extern __shared__ Scalar shared_basis_table[];
    // auto basis_table = reinterpret_cast<std::array<std::array<Scalar, NumBasis>, NumSamples>*>(shared_basis_table);

    // // 第一个 thread 加载数据
    // if (threadIdx.x == 0) {
    //     for (uInt s = 0; s < NumSamples; ++s) {
    //         for (uInt b = 0; b < NumBasis; ++b) {
    //             (*basis_table)[s][b] = (*basis_table_)[s][b];
    //         }
    //     }
    // }
    // __syncthreads();

    // // 后续计算使用 shared memory
    
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
        Scalar U_avg[NEQN];
        #pragma unroll
        for (uInt k = 0; k < NEQN; ++k) U_avg[k] = coef[NEQN*0 + k];

        for (uInt s = 0; s < NumSamples; ++s) {
            const auto& basis = basis_table[s];
            Scalar U_gp[NEQN];
            #pragma unroll
            for (uInt k = 0; k < NEQN; ++k) {
                U_gp[k] = 0;
                #pragma unroll
                for (uInt l = 0; l < NumBasis; ++l) U_gp[k] += basis[l] * coef[NEQN*l + k];
            }

            // 使用 physics 计算压强
            // DenseMatrix<NEQN,1> U_mat;
            // #pragma unroll
            // for (uInt k = 0; k < NEQN; ++k) U_mat[k] = U_gp[k];
            // Scalar p = physics.compute_pressure(U_mat);

            // 内联压强计算（避免函数调用）
            Scalar rho = U_gp[0];
            Scalar u = U_gp[1]/rho, v = U_gp[2]/rho, w = U_gp[3]/rho;
            Scalar E = U_gp[4]/rho;
            Scalar ke = 0.5*(u*u + v*v + w*w);
            Scalar p = (1.4 - 1.0) * rho * (E - ke);


            if (p >= 1e-14) continue;

            // 二分法
            Scalar t_low = 0.0, t_high = 1.0;
            for (int iter = 0; iter < 5; ++iter) {
                Scalar t_mid = 0.5 * (t_low + t_high);
                Scalar U_mid[NEQN];
                #pragma unroll
                for (uInt k = 0; k < NEQN; ++k) 
                    U_mid[k] = (1.0 - t_mid) * U_avg[k] + t_mid * U_gp[k];
                
                // DenseMatrix<NEQN,1> U_mid_mat;
                // #pragma unroll
                // for (uInt k = 0; k < NEQN; ++k) U_mid_mat[k] = U_mid[k];
                // Scalar p_mid = physics.compute_pressure(U_mid_mat);
                // 内联压强计算（避免函数调用）
                Scalar rho = U_mid[0];
                Scalar u = U_mid[1]/rho, v = U_mid[2]/rho, w = U_mid[3]/rho;
                Scalar E = U_mid[4]/rho;
                Scalar ke = 0.5*(u*u + v*v + w*w);
                Scalar p_mid = (1.4 - 1.0) * rho * (E - ke);

                if (p_mid < 0) t_high = t_mid;
                else t_low = t_mid;
                if (t_high - t_low < 1e-5) break;
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