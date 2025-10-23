// include/DG/DG_Schemes/PositiveLimiterGPU_impl.h
#pragma once

#include "base/type.h"
#include "dg/dg_basis/dg_basis.h"
#include "matrix/matrix.h"
#include "mesh/device_mesh.h"
#include "matrix/dense_matrix.h"
#include "matrix/long_vector_device.h"
#include "PositiveLimiterGPU.h"


// forward declare host-side limiter
template <uInt Order, typename QuadType, bool OnlyNeigbAvg = false>
class PositiveLimiterGPU;

// double atomicMin
__device__ __forceinline__ double atomicMinDouble(double *address, double val)
{
    unsigned long long ret = __double_as_longlong(*address);
    while(val < __longlong_as_double(ret))
    {
        unsigned long long old = ret;
        if((ret = atomicCAS((unsigned long long *)address, old, __double_as_longlong(val))) == old)
            break;
    }
    return __longlong_as_double(ret);
}

// double atomicMax
__device__ __forceinline__ double atomicMaxDouble(double *address, double val)
{
    unsigned long long ret = __double_as_longlong(*address);
    while(val > __longlong_as_double(ret))
    {
        unsigned long long old = ret;
        if((ret = atomicCAS((unsigned long long *)address, old, __double_as_longlong(val))) == old)
            break;
    }
    return __longlong_as_double(ret);
}


template<uInt Order, uInt NumBasis, typename QuadType>
__global__ void constructMinMax_cell_kernel(
    const GPUTetrahedron* d_cells, uInt num_cells,
    const DenseMatrix<5*NumBasis,1>* U_prev,
    DenseMatrix<5,1>* per_cell_min,
    DenseMatrix<5,1>* per_cell_max)
{
    uInt cellId = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellId >= num_cells) return;

    DenseMatrix<5,1> maxU, minU;
    for (int k = 0; k < 5; ++k) {
        maxU(k,0) = -1e30;
        minU(k,0) =  1e30;
    }

    const auto& cell = d_cells[cellId];
    for (int nei_face = 0; nei_face < 4; ++nei_face) {
        uInt neighborId = cell.neighbor_cells[nei_face];
        if (neighborId == uInt(-1)) neighborId = cellId;  // 边界时取自身
        DenseMatrix<5*NumBasis,1> coef1 = U_prev[cellId];
        DenseMatrix<5*NumBasis,1> coef2 = U_prev[neighborId];

        constexpr auto Qpoints = QuadType::get_points();
        for (uInt xgi = 0; xgi < QuadType::num_points; ++xgi) {
            const auto& xg = Qpoints[xgi];
            auto basis = DGBasisEvaluator<Order>::eval_all(xg[0], xg[1], xg[2]);

            for (int k = 0; k < 5; ++k) {
                Scalar val1 = 0.0, val2 = 0.0;
                for (uInt l = 0; l < NumBasis; ++l){
                    val1 += basis[l] * coef1(5*l + k, 0);
                    val2 += basis[l] * coef2(5*l + k, 0);
                }
                maxU(k,0) = fmax(maxU(k,0), fmax(val1, val2));
                minU(k,0) = fmin(minU(k,0), fmin(val1, val2));
            }
        }
    }

    per_cell_max[cellId] = maxU;
    per_cell_min[cellId] = minU;
}

// 构建每个单元自身上的积分点极值（不聚合邻居）
template<uInt Order, uInt NumBasis, typename QuadType>
__global__ void construct_cell_extrema_kernel(
    const GPUTetrahedron* d_cells, uInt num_cells,
    const DenseMatrix<5 * NumBasis, 1>* U_prev,
    DenseMatrix<5, 1>* per_cell_min,
    DenseMatrix<5, 1>* per_cell_max)
{
    uInt cellId = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellId >= num_cells) return;

    constexpr auto Qpoints = QuadType::get_points();
    std::array<std::array<Scalar, NumBasis>, QuadType::num_points> basis_table;
    for (uInt xgi = 0; xgi < QuadType::num_points; ++xgi) {
        const auto& xg = Qpoints[xgi];
        basis_table[xgi] = DGBasisEvaluator<Order>::eval_all(xg[0], xg[1], xg[2]);
    }

    const DenseMatrix<5 * NumBasis, 1> coef = U_prev[cellId];
    DenseMatrix<5, 1> maxU, minU;
    for (int k = 0; k < 5; ++k) {
        maxU[k] = -1e30;
        minU[k] = 1e30;
    }

    for (uInt xgi = 0; xgi < QuadType::num_points; ++xgi) {
        const auto& basis = basis_table[xgi];
        for (int k = 0; k < 5; ++k) {
            Scalar val = 0.0;
            for (uInt l = 0; l < NumBasis; ++l)
                val += basis[l] * coef[5 * l + k];
            maxU[k] = fmax(maxU[k], val);
            minU[k] = fmin(minU[k], val);
        }
    }

    for (int k = 0; k < 5; ++k) {
        per_cell_max[cellId][k] = maxU[k];
        per_cell_min[cellId][k] = minU[k];
    }
}

// 构建每个单元自身上的常数模极值（不聚合邻居）
template<uInt Order, uInt NumBasis, typename QuadType>
__global__ void construct_cell_avg_extrema_kernel(
    const GPUTetrahedron* d_cells, uInt num_cells,
    const DenseMatrix<5 * NumBasis, 1>* U_prev,
    DenseMatrix<5, 1>* per_cell_min,
    DenseMatrix<5, 1>* per_cell_max)
{
    uInt cellId = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellId >= num_cells) return;

    const Scalar* coef = U_prev[cellId].data_ptr();
    for (int k = 0; k < 5; ++k) {
        Scalar val = coef[5 * 0 + k];
        per_cell_max[cellId][k] = val;
        per_cell_min[cellId][k] = val;
    }
} 


// 应用保极值限制器（聚合当前单元和邻居）
// template<uInt Order, uInt NumBasis, typename QuadType>
template<uInt Order, uInt NumBasis, typename QuadType>
__global__ void gatter_cell_extrema_kernel(
    const GPUTetrahedron* d_cells, uInt num_cells,
    DenseMatrix<5, 1>* gatter_cell_min,
    DenseMatrix<5, 1>* gatter_cell_max,
    const DenseMatrix<5, 1>* per_cell_min,
    const DenseMatrix<5, 1>* per_cell_max)
{
    uInt cellId = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellId >= num_cells) return;

    DenseMatrix<5, 1> local_min = per_cell_min[cellId];
    DenseMatrix<5, 1> local_max = per_cell_max[cellId];

    const auto& cell = d_cells[cellId];
    for (int nf = 0; nf < 4; ++nf) {
        uInt nid = cell.neighbor_cells[nf];
        if (nid == uInt(-1)) continue;
        for (int k = 0; k < 5; ++k) {
            local_min[k] = fmin(local_min[k], per_cell_min[nid][k]);
            local_max[k] = fmax(local_max[k], per_cell_max[nid][k]);
        }
    }
    gatter_cell_max[cellId] = local_max;
    gatter_cell_min[cellId] = local_min;
}

// template<uInt Order, uInt NumBasis, typename QuadType>
template<uInt Order, uInt NumBasis, typename QuadType>
__global__ void gatter_cell_extrema_by_face_kernel(
    const GPUTriangleFace* d_faces,
    uInt num_faces,
    DenseMatrix<5, 1>* gatter_cell_min,
    DenseMatrix<5, 1>* gatter_cell_max,
    const DenseMatrix<5, 1>* per_cell_min,
    const DenseMatrix<5, 1>* per_cell_max)
{
    uInt faceId = blockIdx.x * blockDim.x + threadIdx.x;
    if (faceId >= num_faces) return;

    const auto& face = d_faces[faceId];
    uInt cidL = face.neighbor_cells[0];
    uInt cidR = face.neighbor_cells[1];

    for (int k = 0; k < 5; ++k) {
        Scalar valL_min = per_cell_min[cidL][k];
        Scalar valL_max = per_cell_max[cidL][k];

        if (cidR != uInt(-1)) {
            Scalar valR_min = per_cell_min[cidR][k];
            Scalar valR_max = per_cell_max[cidR][k];

            atomicMinDouble(&gatter_cell_min[cidL][k], valR_min);
            atomicMaxDouble(&gatter_cell_max[cidL][k], valR_max);

            atomicMinDouble(&gatter_cell_min[cidR][k], valL_min);
            atomicMaxDouble(&gatter_cell_max[cidR][k], valL_max);
        } else {
            // 边界面，仅单边赋值
            atomicMinDouble(&gatter_cell_min[cidL][k], valL_min);
            atomicMaxDouble(&gatter_cell_max[cidL][k], valL_max);
        }
    }
}



// 应用保极值限制器（聚合当前单元和邻居）
template<uInt Order, uInt NumBasis, typename QuadType>
__global__ void apply_extrema_limiter_kernel(
    const GPUTetrahedron* d_cells, uInt num_cells,
    DenseMatrix<5 * NumBasis, 1>* U_current,
    const DenseMatrix<5, 1>* per_cell_min,
    const DenseMatrix<5, 1>* per_cell_max)
{
    uInt cellId = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellId >= num_cells) return;

    constexpr auto Qpoints = QuadType::get_points();
    DenseMatrix<5 * NumBasis, 1>& coef = U_current[cellId];

    DenseMatrix<5, 1> local_min = per_cell_min[cellId];
    DenseMatrix<5, 1> local_max = per_cell_max[cellId];

    // const auto& cell = d_cells[cellId];
    // for (int nf = 0; nf < 4; ++nf) {
    //     uInt nid = cell.neighbor_cells[nf];
    //     if (nid == uInt(-1)) continue;
    //     for (int k = 0; k < 5; ++k) {
    //         local_min[k] = fmin(local_min[k], per_cell_min[nid][k]);
    //         local_max[k] = fmax(local_max[k], per_cell_max[nid][k]);
    //     }
    // }

    for (int k = 0; k < 5; ++k) {
        // !!!!!!!! 
        // This step violates the conservation law.
        // coef[5 * 0 + k] = fmin(fmax(coef[5 * 0 + k],local_min[k]),local_max[k]);
        Scalar avg = coef[5 * 0 + k];
        Scalar val_min = avg, val_max = avg;

        for (uInt xgi = 0; xgi < QuadType::num_points; ++xgi) {
            const auto& xg = Qpoints[xgi];
            auto basis = DGBasisEvaluator<Order>::eval_all(xg[0], xg[1], xg[2]);
            Scalar val = 0.0;
            for (uInt l = 0; l < NumBasis; ++l)
                val += basis[l] * coef[5 * l + k];
            val_min = fmin(val_min, val);
            val_max = fmax(val_max, val);
        }

        if (val_min < local_min[k] || val_max > local_max[k]) {
            Scalar theta_min = (avg != val_min) ? (avg - local_min[k]) / (avg - val_min + 1e-14) : 0.0;
            Scalar theta_max = (avg != val_max) ? (local_max[k] - avg) / (val_max - avg + 1e-14) : 0.0;
            Scalar theta = fmin(fmin(theta_min, theta_max), 1.0);
            for (uInt l = 1; l < NumBasis; ++l) coef[5 * l + k] *= theta;
            
            Scalar new_val_min = avg, new_val_max = avg;
            for (uInt xgi = 0; xgi < QuadType::num_points; ++xgi) {
                const auto& xg = Qpoints[xgi];
                auto basis = DGBasisEvaluator<Order>::eval_all(xg[0], xg[1], xg[2]);
                Scalar val = 0.0;
                for (uInt l = 0; l < NumBasis; ++l)
                    val += basis[l] * coef[5 * l + k];
                new_val_min = fmin(new_val_min, val);
                new_val_max = fmax(new_val_max, val);
            }
            // if(k==0){
            // if(local_max[0]>8.00 || val_max > 8.00)
            //     printf("rho (%8.4lf,%8.4lf,%8.4lf,%8.4lf,%8.4lf)  theta(%8.4lf,%8.4lf,%8.4lf)  new rho (%8.4lf,%8.4lf)\n",local_max[0],val_max,avg,val_min,local_min[0],theta_min,theta_max,theta,new_val_min,new_val_max);
            // }
            // if(k==1){
            //     if(local_max[1]>57.1578 || val_max > 57.1578)
            //     printf("rhou(%8.4lf,%8.4lf,%8.4lf,%8.4lf,%8.4lf)  theta(%8.4lf,%8.4lf,%8.4lf)  new rhou(%8.4lf,%8.4lf)\n",local_max[1],val_max,avg,val_min,local_min[1],theta_min,theta_max,theta,new_val_min,new_val_max);
            // }
            // if(k==2){
            //     if(local_min[2]<-33.00 || val_min <-33.00)
            //     printf("rhov(%8.4lf,%8.4lf,%8.4lf,%8.4lf,%8.4lf)  theta(%8.4lf,%8.4lf,%8.4lf)  new rhov(%8.4lf,%8.4lf)\n",local_max[2],val_max,avg,val_min,local_min[2],theta_min,theta_max,theta,new_val_min,new_val_max);
            // }
        }

        
    }
}

__device__ __forceinline__ Scalar compute_ke(Scalar* U, Scalar eps = 1e-16, Scalar gamma = 1.4){
    Scalar rho = fmax(eps,U[0]);
    Scalar rhou = U[1], rhov = U[2], rhow = U[3];
    Scalar ke = (rhou*rhou + rhov*rhov + rhow*rhow) / fmax(2.0*rho, eps);
    // Scalar p = (gamma - 1.0) * (rhoE - ke);
    return ke;
}

__device__ __forceinline__ Scalar compute_pressure(Scalar* U, Scalar eps = 1e-16, Scalar gamma = 1.4){
    Scalar rho = fmax(eps,U[0]);
    Scalar rhou = U[1], rhov = U[2], rhow = U[3], rhoE = U[4];
    Scalar ke = (rhou*rhou + rhov*rhov + rhow*rhow) / fmax(2.0*rho, eps);
    Scalar p = (gamma - 1.0) * (rhoE - ke);
    return p;
}

template<uInt Order, uInt NumBasis, typename QuadType>
__global__ void apply_2_kernel(
    const GPUTetrahedron* d_cells, uInt num_cells,
    DenseMatrix<5*NumBasis,1>* U_current,
    Scalar gamma)
{
    uInt cellId = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellId >= num_cells) return;
    Scalar eps = 1e-14;

    constexpr auto Qpoints = QuadType::get_points();
    DenseMatrix<5*NumBasis,1>& coef = U_current[cellId];

    std::array<std::array<Scalar, NumBasis>, QuadType::num_points> basis_table;
    #pragma unroll
    for (uInt xgi = 0; xgi < QuadType::num_points; ++xgi)
        basis_table[xgi] = DGBasisEvaluator<Order>::eval_all(Qpoints[xgi][0], Qpoints[xgi][1], Qpoints[xgi][2]);

    // ---------------- 保正密度 ----------------
    {
        // !!!!!!!! 
        // This step violates the conservation law.
        coef[0] = fmax(coef[0],eps);
        Scalar rho_avg = coef[0]; // constant mode
        Scalar rho_min = 1e30;

        #pragma unroll
        for (uInt xgi = 0; xgi < QuadType::num_points; ++xgi) {
            const auto& basis = basis_table[xgi];
            Scalar val = 0;
            #pragma unroll
            for (uInt l = 0; l < NumBasis; ++l)
                val += basis[l] * coef[5*l + 0];
            rho_min = fmin(rho_min, val);
        }
        if (rho_min < eps) {
            Scalar numerator = rho_avg - eps;
            Scalar denominator = rho_avg - rho_min;
            Scalar theta = (denominator < 1e-32) ? 0.0 : fmin(1.0, numerator / denominator);
            #pragma unroll
            for (uInt l = 1; l < NumBasis; ++l)
                coef[5*l + 0] *= theta;
        }
    }


    // {
    //     // Scalar ke = compute_ke(U,eps,gamma);
    //     // !!!!!!!! 
    //     // This step violates the conservation law.
    //     coef[4] = fmax(coef[4],eps);
    //     Scalar rhoE_avg = coef[4]; // constant mode
    //     Scalar rhoE_min = 1e30;

    //     #pragma unroll
    //     for (uInt xgi = 0; xgi < QuadType::num_points; ++xgi) {
    //         const auto& basis = basis_table[xgi];
    //         Scalar val = 0;
    //         #pragma unroll
    //         for (uInt l = 0; l < NumBasis; ++l)
    //             val += basis[l] * coef[5*l + 4];
    //         rhoE_min = fmin(rhoE_min, val);
    //     }
    //     if (rhoE_min < eps) {
    //         Scalar numerator = rhoE_avg - eps;
    //         Scalar denominator = rhoE_avg - rhoE_min;
    //         Scalar theta = (denominator < 1e-32) ? 0.0 : fmin(1.0, numerator / denominator);
    //         #pragma unroll
    //         for (uInt l = 1; l < NumBasis; ++l)
    //             coef[5*l + 4] *= theta;
    //     }
    // }

    // ---------------- 保正压强 ----------------
    {
        Scalar theta_p = 1.0;

        // 提取 constant 模式
        Scalar U_avg[5];
        // coef[4] = fmax(coef[4],1.0);
        #pragma unroll
        for (int k = 0; k < 5; ++k)
            U_avg[k] = coef(5*0 + k, 0);
        // Scalar ke = compute_ke(U_avg,eps,gamma);
        // coef[4] = fmax(coef[4],4*eps+ke);
        // U_avg[4] = coef[4];
        // if(compute_pressure(U_avg,eps,gamma)<eps){
        //     // const auto& cell = d_cells[cellId];
        //     printf("U_avg(%8.4lf,%8.4lf,%8.4lf,%8.4lf,%8.4lf)   p_avg %8.4lf\n", U_avg[0],U_avg[1],U_avg[2],U_avg[3],U_avg[4], compute_pressure(U_avg,eps,gamma));
        // }

        
        #pragma unroll
        for (uInt xgi = 0; xgi < QuadType::num_points; ++xgi) {
            const auto& basis = basis_table[xgi];
            Scalar U_gp[5];

            #pragma unroll
            for (int k = 0; k < 5; ++k) {
                Scalar val = 0.0;
                #pragma unroll
                for (uInt l = 0; l < NumBasis; ++l)
                    val += basis[l] * coef[5*l + k];
                U_gp[k] = val;
            }

            // 计算压力
            // Scalar rho = fmax(eps,U_gp[0]), rhou = U_gp[1], rhov = U_gp[2], rhow = U_gp[3], rhoE = U_gp[4];
            // Scalar ke = (rhou*rhou + rhov*rhov + rhow*rhow) / fmax(2.0*rho, eps);
            // Scalar p = (gamma - 1.0) * (rhoE - ke);
            Scalar p = compute_pressure(U_gp,eps,gamma);
            if (p >= eps) continue;

            // 二分修正
            Scalar t_low = 0.0, t_high = 1.0;
            #pragma unroll
            for (int iter = 0; iter < 50; ++iter) {
                Scalar t_mid = 0.5 * (t_low + t_high);
                Scalar U_mid[5];
                #pragma unroll
                for (int k = 0; k < 5; ++k)
                    U_mid[k] = (1.0 - t_mid) * U_avg[k] + t_mid * U_gp[k];

                // Scalar rho_m = fmax(eps,U_mid[0]), rhou_m = U_mid[1], rhov_m = U_mid[2], rhow_m = U_mid[3], rhoE_m = U_mid[4];
                // Scalar ke_m = (rhou_m*rhou_m + rhov_m*rhov_m + rhow_m*rhow_m) / fmax(2.0*rho_m, eps);
                // Scalar p_m = (gamma - 1.0) * (rhoE_m - ke_m);
                Scalar p_m = compute_pressure(U_mid,eps,gamma);
                // if(compute_pressure(U_avg,eps,gamma)<eps){
                //     printf("U_mid(%8.4lf,%8.4lf,%8.4lf,%8.4lf,%8.4lf)   p_mid %8.4lf   t_low %8.4lf   t_high %8.4lf\n", U_mid[0],U_mid[1],U_mid[2],U_mid[3],U_mid[4], p_m,t_low,t_high);
                // }
                if (p_m < 0.0) t_high = t_mid;
                else t_low = t_mid;
                if ((t_high-t_low<1e-5)||(p_m*p_m<1e-12)) break;
            }

            theta_p = fmin(theta_p, t_low);
        }

        // 限制非 constant 模式
        #pragma unroll
        for (int k = 0; k < 5; ++k)
            #pragma unroll
            for (uInt l = 1; l < NumBasis; ++l)
                coef[5*l + k] *= theta_p;
    }
}

