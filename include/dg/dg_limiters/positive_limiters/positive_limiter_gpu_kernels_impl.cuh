// include/DG/DG_Schemes/PositiveLimiterGPU_kernels_impl.h
#pragma once

#include "dg/dg_limiters/positive_limiters/positive_limiter_gpu_kernels.cuh"

// forward declare host-side limiter
template <uInt Order, typename QuadC, typename QuadF, bool OnlyNeigbAvg = false>
class PositiveLimiterGPU;

template<uInt Order, uInt NumBasis, typename QuadC, typename QuadF>
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

        constexpr auto Qpoints = QuadC::get_points();
        for (uInt xgi = 0; xgi < QuadC::num_points; ++xgi) {
            const auto& xg = Qpoints[xgi];
            const auto& basis = DGBasisEvaluator<Order>::eval_all(xg[0], xg[1], xg[2]);

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
template<uInt Order, uInt NumBasis, typename QuadC, typename QuadF>
__global__ void construct_cell_extrema_kernel(
    const GPUTetrahedron* d_cells, uInt num_cells,
    const DenseMatrix<5 * NumBasis, 1>* U_prev,
    DenseMatrix<5, 1>* per_cell_min,
    DenseMatrix<5, 1>* per_cell_max)
{
    uInt cellId = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellId >= num_cells) return;

    constexpr auto Qpoints = QuadC::get_points();
    std::array<std::array<Scalar, NumBasis>, QuadC::num_points> basis_table;
    for (uInt xgi = 0; xgi < QuadC::num_points; ++xgi) {
        const auto& xg = Qpoints[xgi];
        basis_table[xgi] = DGBasisEvaluator<Order>::eval_all(xg[0], xg[1], xg[2]);
    }

    const DenseMatrix<5 * NumBasis, 1>& coef = U_prev[cellId];
    DenseMatrix<5, 1> maxU, minU;
    for (int k = 0; k < 5; ++k) {
        maxU[k] = -1e30;
        minU[k] = 1e30;
    }

    for (uInt xgi = 0; xgi < QuadC::num_points; ++xgi) {
        const auto& basis = basis_table[xgi];
        for (int k = 0; k < 5; ++k) {
            Scalar val = 0.0;
            for (uInt l = 0; l < NumBasis; ++l)
                val += basis[l] * coef[5 * l + k];
            maxU[k] = fmax(maxU[k], val);
            minU[k] = fmin(minU[k], val);
        }
    }

    // constexpr auto Fpoints = QuadF::get_points();
    // {
    //     for (uInt xgi = 0; xgi < QuadF::num_points; ++xgi) {
    //         const auto& xg = Fpoints[xgi];
    //         Scalar p0 = 0.0 * (1-xg[0]-xg[1]) + 0.0 * xg[0] + 0.0 * xg[1];
    //         Scalar p1 = 1.0 * (1-xg[0]-xg[1]) + 0.0 * xg[0] + 0.0 * xg[1];
    //         Scalar p2 = 0.0 * (1-xg[0]-xg[1]) + 1.0 * xg[0] + 0.0 * xg[1];
    //     //     basis_table[xgi] = DGBasisEvaluator<Order>::eval_all(p0,p1,p2);
    //     // }
    //     // for (uInt xgi = 0; xgi < QuadF::num_points; ++xgi) {
    //     //     const auto& basis = basis_table[xgi];
    //         const auto& basis = DGBasisEvaluator<Order>::eval_all(p0,p1,p2);
    //         #pragma unroll
    //         for (int k = 0; k < 5; ++k) {
    //             Scalar val = 0;
    //             #pragma unroll
    //             for (uInt l = 0; l < NumBasis; ++l)
    //                 val += basis[l] * coef[5*l + 0];
    //             maxU[k] = fmax(maxU[k], val);
    //             minU[k] = fmin(minU[k], val);
    //         }
    //     }
    // }
    // {
    //     for (uInt xgi = 0; xgi < QuadF::num_points; ++xgi) {
    //         const auto& xg = Fpoints[xgi];
    //         Scalar p0 = 0.0 * (1-xg[0]-xg[1]) + 0.0 * xg[0] + 0.0 * xg[1];
    //         Scalar p1 = 1.0 * (1-xg[0]-xg[1]) + 0.0 * xg[0] + 0.0 * xg[1];
    //         Scalar p2 = 0.0 * (1-xg[0]-xg[1]) + 0.0 * xg[0] + 1.0 * xg[1];
    //     //     basis_table[xgi] = DGBasisEvaluator<Order>::eval_all(p0,p1,p2);
    //     // }
    //     // for (uInt xgi = 0; xgi < QuadF::num_points; ++xgi) {
    //     //     const auto& basis = basis_table[xgi];
    //         const auto& basis = DGBasisEvaluator<Order>::eval_all(p0,p1,p2);
    //         #pragma unroll
    //         for (int k = 0; k < 5; ++k) {
    //             Scalar val = 0;
    //             #pragma unroll
    //             for (uInt l = 0; l < NumBasis; ++l)
    //                 val += basis[l] * coef[5*l + 0];
    //             maxU[k] = fmax(maxU[k], val);
    //             minU[k] = fmin(minU[k], val);
    //         }
    //     }
    // }
    // {
    //     for (uInt xgi = 0; xgi < QuadF::num_points; ++xgi) {
    //         const auto& xg = Fpoints[xgi];
    //         Scalar p0 = 0.0 * (1-xg[0]-xg[1]) + 0.0 * xg[0] + 0.0 * xg[1];
    //         Scalar p1 = 0.0 * (1-xg[0]-xg[1]) + 1.0 * xg[0] + 0.0 * xg[1];
    //         Scalar p2 = 0.0 * (1-xg[0]-xg[1]) + 0.0 * xg[0] + 1.0 * xg[1];
    //     //     basis_table[xgi] = DGBasisEvaluator<Order>::eval_all(p0,p1,p2);
    //     // }
    //     // for (uInt xgi = 0; xgi < QuadF::num_points; ++xgi) {
    //     //     const auto& basis = basis_table[xgi];
    //         const auto& basis = DGBasisEvaluator<Order>::eval_all(p0,p1,p2);
    //         #pragma unroll
    //         for (int k = 0; k < 5; ++k) {
    //             Scalar val = 0;
    //             #pragma unroll
    //             for (uInt l = 0; l < NumBasis; ++l)
    //                 val += basis[l] * coef[5*l + 0];
    //             maxU[k] = fmax(maxU[k], val);
    //             minU[k] = fmin(minU[k], val);
    //         }
    //     }
    // }
    // {
    //     for (uInt xgi = 0; xgi < QuadF::num_points; ++xgi) {
    //         const auto& xg = Fpoints[xgi];
    //         Scalar p0 = 1.0 * (1-xg[0]-xg[1]) + 0.0 * xg[0] + 0.0 * xg[1];
    //         Scalar p1 = 0.0 * (1-xg[0]-xg[1]) + 1.0 * xg[0] + 0.0 * xg[1];
    //         Scalar p2 = 0.0 * (1-xg[0]-xg[1]) + 0.0 * xg[0] + 1.0 * xg[1];
    //     //     basis_table[xgi] = DGBasisEvaluator<Order>::eval_all(p0,p1,p2);
    //     // }
    //     // for (uInt xgi = 0; xgi < QuadF::num_points; ++xgi) {
    //     //     const auto& basis = basis_table[xgi];
    //         const auto& basis = DGBasisEvaluator<Order>::eval_all(p0,p1,p2);
    //         #pragma unroll
    //         for (int k = 0; k < 5; ++k) {
    //             Scalar val = 0;
    //             #pragma unroll
    //             for (uInt l = 0; l < NumBasis; ++l)
    //                 val += basis[l] * coef[5*l + 0];
    //             maxU[k] = fmax(maxU[k], val);
    //             minU[k] = fmin(minU[k], val);
    //         }
    //     }
    // }


    for (int k = 0; k < 5; ++k) {
        per_cell_max[cellId][k] = maxU[k];
        per_cell_min[cellId][k] = minU[k];
    }
}

// 构建每个单元自身上的常数模极值（不聚合邻居）
template<uInt Order, uInt NumBasis, typename QuadC, typename QuadF>
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
template<uInt Order, uInt NumBasis, typename QuadC, typename QuadF>
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

// 应用保极值限制器（聚合当前单元和邻居）
template<uInt Order, uInt NumBasis, typename QuadC, typename QuadF>
__global__ void apply_extrema_limiter_kernel(
    const GPUTetrahedron* d_cells, uInt num_cells,
    DenseMatrix<5 * NumBasis, 1>* U_current,
    const DenseMatrix<5, 1>* per_cell_min,
    const DenseMatrix<5, 1>* per_cell_max)
{
    uInt cellId = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellId >= num_cells) return;

    constexpr auto Qpoints = QuadC::get_points();
    DenseMatrix<5 * NumBasis, 1>& coef = U_current[cellId];

    const DenseMatrix<5, 1>& local_min = per_cell_min[cellId];
    const DenseMatrix<5, 1>& local_max = per_cell_max[cellId];

    for (int k = 0; k < 5; ++k) {
        // 这个均值也有可能无法落在极值的范围内
        Scalar avg = coef[5 * 0 + k];
        Scalar val_min = avg, val_max = avg;

        for (uInt xgi = 0; xgi < QuadC::num_points; ++xgi) {
            const auto& xg = Qpoints[xgi];
            const auto& basis = DGBasisEvaluator<Order>::eval_all(xg[0], xg[1], xg[2]);
            Scalar val = 0.0;
            #pragma unroll
            for (uInt l = 0; l < NumBasis; ++l)
                val += basis[l] * coef[5 * l + k];
            val_min = fmin(val_min, val);
            val_max = fmax(val_max, val);
        }


        // constexpr auto Fpoints = QuadF::get_points();
        // {
        //     for (uInt xgi = 0; xgi < QuadF::num_points; ++xgi) {
        //         const auto& xg = Fpoints[xgi];
        //         Scalar p0 = 0.0 * (1-xg[0]-xg[1]) + 0.0 * xg[0] + 0.0 * xg[1];
        //         Scalar p1 = 1.0 * (1-xg[0]-xg[1]) + 0.0 * xg[0] + 0.0 * xg[1];
        //         Scalar p2 = 0.0 * (1-xg[0]-xg[1]) + 1.0 * xg[0] + 0.0 * xg[1];
        //         const auto& basis = DGBasisEvaluator<Order>::eval_all(p0,p1,p2);
        //         Scalar val = 0;
        //         #pragma unroll
        //         for (uInt l = 0; l < NumBasis; ++l)
        //             val += basis[l] * coef[5*l + 0];
        //         val_min = fmin(val_min, val);
        //         val_max = fmax(val_max, val);
        //     }
        // }
        // {
        //     for (uInt xgi = 0; xgi < QuadF::num_points; ++xgi) {
        //         const auto& xg = Fpoints[xgi];
        //         Scalar p0 = 0.0 * (1-xg[0]-xg[1]) + 0.0 * xg[0] + 0.0 * xg[1];
        //         Scalar p1 = 1.0 * (1-xg[0]-xg[1]) + 0.0 * xg[0] + 0.0 * xg[1];
        //         Scalar p2 = 0.0 * (1-xg[0]-xg[1]) + 0.0 * xg[0] + 1.0 * xg[1];
        //         const auto& basis = DGBasisEvaluator<Order>::eval_all(p0,p1,p2);
        //         Scalar val = 0;
        //         #pragma unroll
        //         for (uInt l = 0; l < NumBasis; ++l)
        //             val += basis[l] * coef[5*l + 0];
        //         val_min = fmin(val_min, val);
        //         val_max = fmax(val_max, val);
        //     }
        // }
        // {
        //     for (uInt xgi = 0; xgi < QuadF::num_points; ++xgi) {
        //         const auto& xg = Fpoints[xgi];
        //         Scalar p0 = 0.0 * (1-xg[0]-xg[1]) + 0.0 * xg[0] + 0.0 * xg[1];
        //         Scalar p1 = 0.0 * (1-xg[0]-xg[1]) + 1.0 * xg[0] + 0.0 * xg[1];
        //         Scalar p2 = 0.0 * (1-xg[0]-xg[1]) + 0.0 * xg[0] + 1.0 * xg[1];
        //         const auto& basis = DGBasisEvaluator<Order>::eval_all(p0,p1,p2);
        //         Scalar val = 0;
        //         #pragma unroll
        //         for (uInt l = 0; l < NumBasis; ++l)
        //             val += basis[l] * coef[5*l + 0];
        //         val_min = fmin(val_min, val);
        //         val_max = fmax(val_max, val);
        //     }
        // }
        // {
        //     for (uInt xgi = 0; xgi < QuadF::num_points; ++xgi) {
        //         const auto& xg = Fpoints[xgi];
        //         Scalar p0 = 1.0 * (1-xg[0]-xg[1]) + 0.0 * xg[0] + 0.0 * xg[1];
        //         Scalar p1 = 0.0 * (1-xg[0]-xg[1]) + 1.0 * xg[0] + 0.0 * xg[1];
        //         Scalar p2 = 0.0 * (1-xg[0]-xg[1]) + 0.0 * xg[0] + 1.0 * xg[1];
        //         const auto& basis = DGBasisEvaluator<Order>::eval_all(p0,p1,p2);
        //         Scalar val = 0;
        //         #pragma unroll
        //         for (uInt l = 0; l < NumBasis; ++l)
        //             val += basis[l] * coef[5*l + 0];
        //         val_min = fmin(val_min, val);
        //         val_max = fmax(val_max, val);
        //     }
        // }



        if (val_min < local_min[k] || val_max > local_max[k]) {
            Scalar theta_min = (avg != val_min) ? (avg - local_min[k]) / (avg - val_min + 1e-14) : 0.0;
            Scalar theta_max = (avg != val_max) ? (local_max[k] - avg) / (val_max - avg + 1e-14) : 0.0;
            Scalar theta = fmin(fmin(theta_min, theta_max), 1.0);
            for (uInt l = 1; l < NumBasis; ++l) coef[5 * l + k] *= theta;
            
            // Scalar new_val_min = avg, new_val_max = avg;
            // for (uInt xgi = 0; xgi < QuadC::num_points; ++xgi) {
            //     const auto& xg = Qpoints[xgi];
            //     const auto& basis = DGBasisEvaluator<Order>::eval_all(xg[0], xg[1], xg[2]);
            //     Scalar val = 0.0;
            //     for (uInt l = 0; l < NumBasis; ++l)
            //         val += basis[l] * coef[5 * l + k];
            //     new_val_min = fmin(new_val_min, val);
            //     new_val_max = fmax(new_val_max, val);
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

template<uInt Order, uInt NumBasis, typename QuadC, typename QuadF>
__global__ void apply_2_kernel(
    const GPUTetrahedron* d_cells, uInt num_cells,
    DenseMatrix<5*NumBasis,1>* U_current,
    Scalar gamma)
{
    uInt cellId = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellId >= num_cells) return;
    Scalar eps = 1e-14;

    constexpr auto Qpoints = QuadC::get_points();
    DenseMatrix<5*NumBasis,1>& coef = U_current[cellId];

    std::array<std::array<Scalar, NumBasis>, QuadC::num_points> basis_table;
    #pragma unroll
    for (uInt xgi = 0; xgi < QuadC::num_points; ++xgi)
        basis_table[xgi] = DGBasisEvaluator<Order>::eval_all(Qpoints[xgi][0], Qpoints[xgi][1], Qpoints[xgi][2]);

    // ---------------- 保正密度 ----------------
    {
        // !!!!!!!! 
        // This step violates the conservation law.
        coef[0] = fmax(coef[0],eps);
        Scalar rho_avg = coef[0]; // constant mode
        Scalar rho_min = 1e30;

        // #pragma unroll
        for (uInt xgi = 0; xgi < QuadC::num_points; ++xgi) {
            const auto& basis = basis_table[xgi];
            Scalar val = 0;
            #pragma unroll
            for (uInt l = 0; l < NumBasis; ++l)
                val += basis[l] * coef[5*l + 0];
            rho_min = fmin(rho_min, val);
        }



        
        // constexpr auto Fpoints = QuadF::get_points();
        // {
        //     for (uInt xgi = 0; xgi < QuadF::num_points; ++xgi) {
        //         const auto& xg = Fpoints[xgi];
        //         Scalar p0 = 0.0 * (1-xg[0]-xg[1]) + 0.0 * xg[0] + 0.0 * xg[1];
        //         Scalar p1 = 1.0 * (1-xg[0]-xg[1]) + 0.0 * xg[0] + 0.0 * xg[1];
        //         Scalar p2 = 0.0 * (1-xg[0]-xg[1]) + 1.0 * xg[0] + 0.0 * xg[1];
        //         {
        //             const auto& basis = DGBasisEvaluator<Order>::eval_all(p0,p1,p2);
        //             Scalar val = 0;
        //             #pragma unroll
        //             for (uInt l = 0; l < NumBasis; ++l)
        //                 val += basis[l] * coef[5*l + 0];
        //             rho_min = fmin(rho_min, val);
        //         }
        //         {
        //             const auto& basis = DGBasisEvaluator<Order>::eval_all(p1,p2,p0);
        //             Scalar val = 0;
        //             #pragma unroll
        //             for (uInt l = 0; l < NumBasis; ++l)
        //                 val += basis[l] * coef[5*l + 0];
        //             rho_min = fmin(rho_min, val);
        //         }
        //         {
        //             const auto& basis = DGBasisEvaluator<Order>::eval_all(p2,p0,p1);
        //             Scalar val = 0;
        //             #pragma unroll
        //             for (uInt l = 0; l < NumBasis; ++l)
        //                 val += basis[l] * coef[5*l + 0];
        //             rho_min = fmin(rho_min, val);
        //         }
        //     }
        // }
        // {
        //     for (uInt xgi = 0; xgi < QuadF::num_points; ++xgi) {
        //         const auto& xg = Fpoints[xgi];
        //         Scalar p0 = 0.0 * (1-xg[0]-xg[1]) + 0.0 * xg[0] + 0.0 * xg[1];
        //         Scalar p1 = 1.0 * (1-xg[0]-xg[1]) + 0.0 * xg[0] + 0.0 * xg[1];
        //         Scalar p2 = 0.0 * (1-xg[0]-xg[1]) + 0.0 * xg[0] + 1.0 * xg[1];
        //         {
        //             const auto& basis = DGBasisEvaluator<Order>::eval_all(p0,p1,p2);
        //             Scalar val = 0;
        //             #pragma unroll
        //             for (uInt l = 0; l < NumBasis; ++l)
        //                 val += basis[l] * coef[5*l + 0];
        //             rho_min = fmin(rho_min, val);
        //         }
        //         {
        //             const auto& basis = DGBasisEvaluator<Order>::eval_all(p1,p2,p0);
        //             Scalar val = 0;
        //             #pragma unroll
        //             for (uInt l = 0; l < NumBasis; ++l)
        //                 val += basis[l] * coef[5*l + 0];
        //             rho_min = fmin(rho_min, val);
        //         }
        //         {
        //             const auto& basis = DGBasisEvaluator<Order>::eval_all(p2,p0,p1);
        //             Scalar val = 0;
        //             #pragma unroll
        //             for (uInt l = 0; l < NumBasis; ++l)
        //                 val += basis[l] * coef[5*l + 0];
        //             rho_min = fmin(rho_min, val);
        //         }
        //     }
        // }
        // {
        //     for (uInt xgi = 0; xgi < QuadF::num_points; ++xgi) {
        //         const auto& xg = Fpoints[xgi];
        //         Scalar p0 = 0.0 * (1-xg[0]-xg[1]) + 0.0 * xg[0] + 0.0 * xg[1];
        //         Scalar p1 = 0.0 * (1-xg[0]-xg[1]) + 1.0 * xg[0] + 0.0 * xg[1];
        //         Scalar p2 = 0.0 * (1-xg[0]-xg[1]) + 0.0 * xg[0] + 1.0 * xg[1];
        //         {
        //             const auto& basis = DGBasisEvaluator<Order>::eval_all(p0,p1,p2);
        //             Scalar val = 0;
        //             #pragma unroll
        //             for (uInt l = 0; l < NumBasis; ++l)
        //                 val += basis[l] * coef[5*l + 0];
        //             rho_min = fmin(rho_min, val);
        //         }
        //         {
        //             const auto& basis = DGBasisEvaluator<Order>::eval_all(p1,p2,p0);
        //             Scalar val = 0;
        //             #pragma unroll
        //             for (uInt l = 0; l < NumBasis; ++l)
        //                 val += basis[l] * coef[5*l + 0];
        //             rho_min = fmin(rho_min, val);
        //         }
        //         {
        //             const auto& basis = DGBasisEvaluator<Order>::eval_all(p2,p0,p1);
        //             Scalar val = 0;
        //             #pragma unroll
        //             for (uInt l = 0; l < NumBasis; ++l)
        //                 val += basis[l] * coef[5*l + 0];
        //             rho_min = fmin(rho_min, val);
        //         }
        //     }
        // }
        // {
        //     for (uInt xgi = 0; xgi < QuadF::num_points; ++xgi) {
        //         const auto& xg = Fpoints[xgi];
        //         Scalar p0 = 1.0 * (1-xg[0]-xg[1]) + 0.0 * xg[0] + 0.0 * xg[1];
        //         Scalar p1 = 0.0 * (1-xg[0]-xg[1]) + 1.0 * xg[0] + 0.0 * xg[1];
        //         Scalar p2 = 0.0 * (1-xg[0]-xg[1]) + 0.0 * xg[0] + 1.0 * xg[1];
        //         {
        //             const auto& basis = DGBasisEvaluator<Order>::eval_all(p0,p1,p2);
        //             Scalar val = 0;
        //             #pragma unroll
        //             for (uInt l = 0; l < NumBasis; ++l)
        //                 val += basis[l] * coef[5*l + 0];
        //             rho_min = fmin(rho_min, val);
        //         }
        //         {
        //             const auto& basis = DGBasisEvaluator<Order>::eval_all(p1,p2,p0);
        //             Scalar val = 0;
        //             #pragma unroll
        //             for (uInt l = 0; l < NumBasis; ++l)
        //                 val += basis[l] * coef[5*l + 0];
        //             rho_min = fmin(rho_min, val);
        //         }
        //         {
        //             const auto& basis = DGBasisEvaluator<Order>::eval_all(p2,p0,p1);
        //             Scalar val = 0;
        //             #pragma unroll
        //             for (uInt l = 0; l < NumBasis; ++l)
        //                 val += basis[l] * coef[5*l + 0];
        //             rho_min = fmin(rho_min, val);
        //         }
        //     }
        // }
        {
            {
                const auto& basis = DGBasisEvaluator<Order>::eval_all(0.0,0.0,0.0);
                Scalar val = 0;
                #pragma unroll
                for (uInt l = 0; l < NumBasis; ++l)
                    val += basis[l] * coef[5*l + 0];
                rho_min = fmin(rho_min, val);
            }
            {
                const auto& basis = DGBasisEvaluator<Order>::eval_all(1.0,0.0,0.0);
                Scalar val = 0;
                #pragma unroll
                for (uInt l = 0; l < NumBasis; ++l)
                    val += basis[l] * coef[5*l + 0];
                rho_min = fmin(rho_min, val);
            }
            {
                const auto& basis = DGBasisEvaluator<Order>::eval_all(0.0,1.0,0.0);
                Scalar val = 0;
                #pragma unroll
                for (uInt l = 0; l < NumBasis; ++l)
                    val += basis[l] * coef[5*l + 0];
                rho_min = fmin(rho_min, val);
            }
            {
                const auto& basis = DGBasisEvaluator<Order>::eval_all(0.0,0.0,1.0);
                Scalar val = 0;
                #pragma unroll
                for (uInt l = 0; l < NumBasis; ++l)
                    val += basis[l] * coef[5*l + 0];
                rho_min = fmin(rho_min, val);
            }
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

    // ---------------- 保正压强 ----------------
    {
        Scalar theta_p = 1.0;

        // 提取 constant 模式
        Scalar U_avg[5];
        // coef[4] = fmax(coef[4],1.0);
        #pragma unroll
        for (int k = 0; k < 5; ++k)
            U_avg[k] = coef(5*0 + k, 0);
        // Scalar p_avg = compute_pressure(U_avg,eps,gamma);
        // if (p_avg < eps) printf("cell id: %ld,    U_avg(%le,%le,%le,%le,%le),    p_avg: %le", cellId, U_avg[0], U_avg[1], U_avg[2], U_avg[3], U_avg[4], p_avg);
        // #pragma unroll
        for (uInt xgi = 0; xgi < QuadC::num_points; ++xgi) {
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

                // 更新后的压强
                Scalar p_m = compute_pressure(U_mid,eps,gamma);

                // 左边大于 0，右边小于 0，中间小于 0 就替换右边，否则替换左边
                if (p_m < 0.0) t_high = t_mid;
                else t_low = t_mid;
                
                // 停机，事实上步长缩减每步减少一半，完全不需要 50 步，
                // 10 步就能达到 1e-3，至多 20 步达到 1e-6
                if ((t_high-t_low<1e-5)||(p_m*p_m<1e-12)) break;
            }
            theta_p = fmin(theta_p, t_low);
        }



        
        
        // constexpr auto Fpoints = QuadF::get_points();
        // {
        //     for (uInt xgi = 0; xgi < QuadF::num_points; ++xgi) {
        //         const auto& xg = Fpoints[xgi];
        //         Scalar p0 = 0.0 * (1-xg[0]-xg[1]) + 0.0 * xg[0] + 0.0 * xg[1];
        //         Scalar p1 = 1.0 * (1-xg[0]-xg[1]) + 0.0 * xg[0] + 0.0 * xg[1];
        //         Scalar p2 = 0.0 * (1-xg[0]-xg[1]) + 1.0 * xg[0] + 0.0 * xg[1];
        //         {
        //             const auto& basis = DGBasisEvaluator<Order>::eval_all(p0,p1,p2);

        //             Scalar U_gp[5];
        //             #pragma unroll
        //             for (int k = 0; k < 5; ++k) {
        //                 Scalar val = 0.0;
        //                 #pragma unroll
        //                 for (uInt l = 0; l < NumBasis; ++l)
        //                     val += basis[l] * coef[5*l + k];
        //                 U_gp[k] = val;
        //             }
        //             Scalar p = compute_pressure(U_gp,eps,gamma);
        //             if (p >= eps) continue;
        //             Scalar t_low = 0.0, t_high = 1.0;
        //             #pragma unroll
        //             for (int iter = 0; iter < 50; ++iter) {
        //                 Scalar t_mid = 0.5 * (t_low + t_high);
        //                 Scalar U_mid[5];
        //                 #pragma unroll
        //                 for (int k = 0; k < 5; ++k)
        //                     U_mid[k] = (1.0 - t_mid) * U_avg[k] + t_mid * U_gp[k];
        //                 Scalar p_m = compute_pressure(U_mid,eps,gamma);
        //                 if (p_m < 0.0) t_high = t_mid;
        //                 else t_low = t_mid;
        //                 if ((t_high-t_low<1e-5)||(p_m*p_m<1e-12)) break;
        //             }
        //             theta_p = fmin(theta_p, t_low);
        //         }
        //         {
        //             const auto& basis = DGBasisEvaluator<Order>::eval_all(p1,p2,p0);

        //             Scalar U_gp[5];
        //             #pragma unroll
        //             for (int k = 0; k < 5; ++k) {
        //                 Scalar val = 0.0;
        //                 #pragma unroll
        //                 for (uInt l = 0; l < NumBasis; ++l)
        //                     val += basis[l] * coef[5*l + k];
        //                 U_gp[k] = val;
        //             }
        //             Scalar p = compute_pressure(U_gp,eps,gamma);
        //             if (p >= eps) continue;
        //             Scalar t_low = 0.0, t_high = 1.0;
        //             #pragma unroll
        //             for (int iter = 0; iter < 50; ++iter) {
        //                 Scalar t_mid = 0.5 * (t_low + t_high);
        //                 Scalar U_mid[5];
        //                 #pragma unroll
        //                 for (int k = 0; k < 5; ++k)
        //                     U_mid[k] = (1.0 - t_mid) * U_avg[k] + t_mid * U_gp[k];
        //                 Scalar p_m = compute_pressure(U_mid,eps,gamma);
        //                 if (p_m < 0.0) t_high = t_mid;
        //                 else t_low = t_mid;
        //                 if ((t_high-t_low<1e-5)||(p_m*p_m<1e-12)) break;
        //             }
        //             theta_p = fmin(theta_p, t_low);
        //         }
        //         {
        //             const auto& basis = DGBasisEvaluator<Order>::eval_all(p2,p0,p1);

        //             Scalar U_gp[5];
        //             #pragma unroll
        //             for (int k = 0; k < 5; ++k) {
        //                 Scalar val = 0.0;
        //                 #pragma unroll
        //                 for (uInt l = 0; l < NumBasis; ++l)
        //                     val += basis[l] * coef[5*l + k];
        //                 U_gp[k] = val;
        //             }
        //             Scalar p = compute_pressure(U_gp,eps,gamma);
        //             if (p >= eps) continue;
        //             Scalar t_low = 0.0, t_high = 1.0;
        //             #pragma unroll
        //             for (int iter = 0; iter < 50; ++iter) {
        //                 Scalar t_mid = 0.5 * (t_low + t_high);
        //                 Scalar U_mid[5];
        //                 #pragma unroll
        //                 for (int k = 0; k < 5; ++k)
        //                     U_mid[k] = (1.0 - t_mid) * U_avg[k] + t_mid * U_gp[k];
        //                 Scalar p_m = compute_pressure(U_mid,eps,gamma);
        //                 if (p_m < 0.0) t_high = t_mid;
        //                 else t_low = t_mid;
        //                 if ((t_high-t_low<1e-5)||(p_m*p_m<1e-12)) break;
        //             }
        //             theta_p = fmin(theta_p, t_low);
        //         }
        //     }
        // }
        // {
        //     for (uInt xgi = 0; xgi < QuadF::num_points; ++xgi) {
        //         const auto& xg = Fpoints[xgi];
        //         Scalar p0 = 0.0 * (1-xg[0]-xg[1]) + 0.0 * xg[0] + 0.0 * xg[1];
        //         Scalar p1 = 1.0 * (1-xg[0]-xg[1]) + 0.0 * xg[0] + 0.0 * xg[1];
        //         Scalar p2 = 0.0 * (1-xg[0]-xg[1]) + 0.0 * xg[0] + 1.0 * xg[1];
        //         {
        //             const auto& basis = DGBasisEvaluator<Order>::eval_all(p0,p1,p2);

        //             Scalar U_gp[5];
        //             #pragma unroll
        //             for (int k = 0; k < 5; ++k) {
        //                 Scalar val = 0.0;
        //                 #pragma unroll
        //                 for (uInt l = 0; l < NumBasis; ++l)
        //                     val += basis[l] * coef[5*l + k];
        //                 U_gp[k] = val;
        //             }
        //             Scalar p = compute_pressure(U_gp,eps,gamma);
        //             if (p >= eps) continue;
        //             Scalar t_low = 0.0, t_high = 1.0;
        //             #pragma unroll
        //             for (int iter = 0; iter < 50; ++iter) {
        //                 Scalar t_mid = 0.5 * (t_low + t_high);
        //                 Scalar U_mid[5];
        //                 #pragma unroll
        //                 for (int k = 0; k < 5; ++k)
        //                     U_mid[k] = (1.0 - t_mid) * U_avg[k] + t_mid * U_gp[k];
        //                 Scalar p_m = compute_pressure(U_mid,eps,gamma);
        //                 if (p_m < 0.0) t_high = t_mid;
        //                 else t_low = t_mid;
        //                 if ((t_high-t_low<1e-5)||(p_m*p_m<1e-12)) break;
        //             }
        //             theta_p = fmin(theta_p, t_low);
        //         }
        //         {
        //             const auto& basis = DGBasisEvaluator<Order>::eval_all(p1,p2,p0);

        //             Scalar U_gp[5];
        //             #pragma unroll
        //             for (int k = 0; k < 5; ++k) {
        //                 Scalar val = 0.0;
        //                 #pragma unroll
        //                 for (uInt l = 0; l < NumBasis; ++l)
        //                     val += basis[l] * coef[5*l + k];
        //                 U_gp[k] = val;
        //             }
        //             Scalar p = compute_pressure(U_gp,eps,gamma);
        //             if (p >= eps) continue;
        //             Scalar t_low = 0.0, t_high = 1.0;
        //             #pragma unroll
        //             for (int iter = 0; iter < 50; ++iter) {
        //                 Scalar t_mid = 0.5 * (t_low + t_high);
        //                 Scalar U_mid[5];
        //                 #pragma unroll
        //                 for (int k = 0; k < 5; ++k)
        //                     U_mid[k] = (1.0 - t_mid) * U_avg[k] + t_mid * U_gp[k];
        //                 Scalar p_m = compute_pressure(U_mid,eps,gamma);
        //                 if (p_m < 0.0) t_high = t_mid;
        //                 else t_low = t_mid;
        //                 if ((t_high-t_low<1e-5)||(p_m*p_m<1e-12)) break;
        //             }
        //             theta_p = fmin(theta_p, t_low);
        //         }
        //         {
        //             const auto& basis = DGBasisEvaluator<Order>::eval_all(p2,p0,p1);

        //             Scalar U_gp[5];
        //             #pragma unroll
        //             for (int k = 0; k < 5; ++k) {
        //                 Scalar val = 0.0;
        //                 #pragma unroll
        //                 for (uInt l = 0; l < NumBasis; ++l)
        //                     val += basis[l] * coef[5*l + k];
        //                 U_gp[k] = val;
        //             }
        //             Scalar p = compute_pressure(U_gp,eps,gamma);
        //             if (p >= eps) continue;
        //             Scalar t_low = 0.0, t_high = 1.0;
        //             #pragma unroll
        //             for (int iter = 0; iter < 50; ++iter) {
        //                 Scalar t_mid = 0.5 * (t_low + t_high);
        //                 Scalar U_mid[5];
        //                 #pragma unroll
        //                 for (int k = 0; k < 5; ++k)
        //                     U_mid[k] = (1.0 - t_mid) * U_avg[k] + t_mid * U_gp[k];
        //                 Scalar p_m = compute_pressure(U_mid,eps,gamma);
        //                 if (p_m < 0.0) t_high = t_mid;
        //                 else t_low = t_mid;
        //                 if ((t_high-t_low<1e-5)||(p_m*p_m<1e-12)) break;
        //             }
        //             theta_p = fmin(theta_p, t_low);
        //         }
        //     }
        // }
        // {
        //     for (uInt xgi = 0; xgi < QuadF::num_points; ++xgi) {
        //         const auto& xg = Fpoints[xgi];
        //         Scalar p0 = 0.0 * (1-xg[0]-xg[1]) + 0.0 * xg[0] + 0.0 * xg[1];
        //         Scalar p1 = 0.0 * (1-xg[0]-xg[1]) + 1.0 * xg[0] + 0.0 * xg[1];
        //         Scalar p2 = 0.0 * (1-xg[0]-xg[1]) + 0.0 * xg[0] + 1.0 * xg[1];
        //         {
        //             const auto& basis = DGBasisEvaluator<Order>::eval_all(p0,p1,p2);

        //             Scalar U_gp[5];
        //             #pragma unroll
        //             for (int k = 0; k < 5; ++k) {
        //                 Scalar val = 0.0;
        //                 #pragma unroll
        //                 for (uInt l = 0; l < NumBasis; ++l)
        //                     val += basis[l] * coef[5*l + k];
        //                 U_gp[k] = val;
        //             }
        //             Scalar p = compute_pressure(U_gp,eps,gamma);
        //             if (p >= eps) continue;
        //             Scalar t_low = 0.0, t_high = 1.0;
        //             #pragma unroll
        //             for (int iter = 0; iter < 50; ++iter) {
        //                 Scalar t_mid = 0.5 * (t_low + t_high);
        //                 Scalar U_mid[5];
        //                 #pragma unroll
        //                 for (int k = 0; k < 5; ++k)
        //                     U_mid[k] = (1.0 - t_mid) * U_avg[k] + t_mid * U_gp[k];
        //                 Scalar p_m = compute_pressure(U_mid,eps,gamma);
        //                 if (p_m < 0.0) t_high = t_mid;
        //                 else t_low = t_mid;
        //                 if ((t_high-t_low<1e-5)||(p_m*p_m<1e-12)) break;
        //             }
        //             theta_p = fmin(theta_p, t_low);
        //         }
        //         {
        //             const auto& basis = DGBasisEvaluator<Order>::eval_all(p1,p2,p0);

        //             Scalar U_gp[5];
        //             #pragma unroll
        //             for (int k = 0; k < 5; ++k) {
        //                 Scalar val = 0.0;
        //                 #pragma unroll
        //                 for (uInt l = 0; l < NumBasis; ++l)
        //                     val += basis[l] * coef[5*l + k];
        //                 U_gp[k] = val;
        //             }
        //             Scalar p = compute_pressure(U_gp,eps,gamma);
        //             if (p >= eps) continue;
        //             Scalar t_low = 0.0, t_high = 1.0;
        //             #pragma unroll
        //             for (int iter = 0; iter < 50; ++iter) {
        //                 Scalar t_mid = 0.5 * (t_low + t_high);
        //                 Scalar U_mid[5];
        //                 #pragma unroll
        //                 for (int k = 0; k < 5; ++k)
        //                     U_mid[k] = (1.0 - t_mid) * U_avg[k] + t_mid * U_gp[k];
        //                 Scalar p_m = compute_pressure(U_mid,eps,gamma);
        //                 if (p_m < 0.0) t_high = t_mid;
        //                 else t_low = t_mid;
        //                 if ((t_high-t_low<1e-5)||(p_m*p_m<1e-12)) break;
        //             }
        //             theta_p = fmin(theta_p, t_low);
        //         }
        //         {
        //             const auto& basis = DGBasisEvaluator<Order>::eval_all(p2,p0,p1);

        //             Scalar U_gp[5];
        //             #pragma unroll
        //             for (int k = 0; k < 5; ++k) {
        //                 Scalar val = 0.0;
        //                 #pragma unroll
        //                 for (uInt l = 0; l < NumBasis; ++l)
        //                     val += basis[l] * coef[5*l + k];
        //                 U_gp[k] = val;
        //             }
        //             Scalar p = compute_pressure(U_gp,eps,gamma);
        //             if (p >= eps) continue;
        //             Scalar t_low = 0.0, t_high = 1.0;
        //             #pragma unroll
        //             for (int iter = 0; iter < 50; ++iter) {
        //                 Scalar t_mid = 0.5 * (t_low + t_high);
        //                 Scalar U_mid[5];
        //                 #pragma unroll
        //                 for (int k = 0; k < 5; ++k)
        //                     U_mid[k] = (1.0 - t_mid) * U_avg[k] + t_mid * U_gp[k];
        //                 Scalar p_m = compute_pressure(U_mid,eps,gamma);
        //                 if (p_m < 0.0) t_high = t_mid;
        //                 else t_low = t_mid;
        //                 if ((t_high-t_low<1e-5)||(p_m*p_m<1e-12)) break;
        //             }
        //             theta_p = fmin(theta_p, t_low);
        //         }
        //     }
        // }
        // {
        //     for (uInt xgi = 0; xgi < QuadF::num_points; ++xgi) {
        //         const auto& xg = Fpoints[xgi];
        //         Scalar p0 = 1.0 * (1-xg[0]-xg[1]) + 0.0 * xg[0] + 0.0 * xg[1];
        //         Scalar p1 = 0.0 * (1-xg[0]-xg[1]) + 1.0 * xg[0] + 0.0 * xg[1];
        //         Scalar p2 = 0.0 * (1-xg[0]-xg[1]) + 0.0 * xg[0] + 1.0 * xg[1];
        //         {
        //             const auto& basis = DGBasisEvaluator<Order>::eval_all(p0,p1,p2);

        //             Scalar U_gp[5];
        //             #pragma unroll
        //             for (int k = 0; k < 5; ++k) {
        //                 Scalar val = 0.0;
        //                 #pragma unroll
        //                 for (uInt l = 0; l < NumBasis; ++l)
        //                     val += basis[l] * coef[5*l + k];
        //                 U_gp[k] = val;
        //             }
        //             Scalar p = compute_pressure(U_gp,eps,gamma);
        //             if (p >= eps) continue;
        //             Scalar t_low = 0.0, t_high = 1.0;
        //             #pragma unroll
        //             for (int iter = 0; iter < 50; ++iter) {
        //                 Scalar t_mid = 0.5 * (t_low + t_high);
        //                 Scalar U_mid[5];
        //                 #pragma unroll
        //                 for (int k = 0; k < 5; ++k)
        //                     U_mid[k] = (1.0 - t_mid) * U_avg[k] + t_mid * U_gp[k];
        //                 Scalar p_m = compute_pressure(U_mid,eps,gamma);
        //                 if (p_m < 0.0) t_high = t_mid;
        //                 else t_low = t_mid;
        //                 if ((t_high-t_low<1e-5)||(p_m*p_m<1e-12)) break;
        //             }
        //             theta_p = fmin(theta_p, t_low);
        //         }
        //         {
        //             const auto& basis = DGBasisEvaluator<Order>::eval_all(p1,p2,p0);

        //             Scalar U_gp[5];
        //             #pragma unroll
        //             for (int k = 0; k < 5; ++k) {
        //                 Scalar val = 0.0;
        //                 #pragma unroll
        //                 for (uInt l = 0; l < NumBasis; ++l)
        //                     val += basis[l] * coef[5*l + k];
        //                 U_gp[k] = val;
        //             }
        //             Scalar p = compute_pressure(U_gp,eps,gamma);
        //             if (p >= eps) continue;
        //             Scalar t_low = 0.0, t_high = 1.0;
        //             #pragma unroll
        //             for (int iter = 0; iter < 50; ++iter) {
        //                 Scalar t_mid = 0.5 * (t_low + t_high);
        //                 Scalar U_mid[5];
        //                 #pragma unroll
        //                 for (int k = 0; k < 5; ++k)
        //                     U_mid[k] = (1.0 - t_mid) * U_avg[k] + t_mid * U_gp[k];
        //                 Scalar p_m = compute_pressure(U_mid,eps,gamma);
        //                 if (p_m < 0.0) t_high = t_mid;
        //                 else t_low = t_mid;
        //                 if ((t_high-t_low<1e-5)||(p_m*p_m<1e-12)) break;
        //             }
        //             theta_p = fmin(theta_p, t_low);
        //         }
        //         {
        //             const auto& basis = DGBasisEvaluator<Order>::eval_all(p2,p0,p1);

        //             Scalar U_gp[5];
        //             #pragma unroll
        //             for (int k = 0; k < 5; ++k) {
        //                 Scalar val = 0.0;
        //                 #pragma unroll
        //                 for (uInt l = 0; l < NumBasis; ++l)
        //                     val += basis[l] * coef[5*l + k];
        //                 U_gp[k] = val;
        //             }
        //             Scalar p = compute_pressure(U_gp,eps,gamma);
        //             if (p >= eps) continue;
        //             Scalar t_low = 0.0, t_high = 1.0;
        //             #pragma unroll
        //             for (int iter = 0; iter < 50; ++iter) {
        //                 Scalar t_mid = 0.5 * (t_low + t_high);
        //                 Scalar U_mid[5];
        //                 #pragma unroll
        //                 for (int k = 0; k < 5; ++k)
        //                     U_mid[k] = (1.0 - t_mid) * U_avg[k] + t_mid * U_gp[k];
        //                 Scalar p_m = compute_pressure(U_mid,eps,gamma);
        //                 if (p_m < 0.0) t_high = t_mid;
        //                 else t_low = t_mid;
        //                 if ((t_high-t_low<1e-5)||(p_m*p_m<1e-12)) break;
        //             }
        //             theta_p = fmin(theta_p, t_low);
        //         }
        //     }
        // }
        {
            {
                const auto& basis = DGBasisEvaluator<Order>::eval_all(0.0,0.0,0.0);

                Scalar U_gp[5];
                #pragma unroll
                for (int k = 0; k < 5; ++k) {
                    Scalar val = 0.0;
                    #pragma unroll
                    for (uInt l = 0; l < NumBasis; ++l)
                        val += basis[l] * coef[5*l + k];
                    U_gp[k] = val;
                }
                Scalar p = compute_pressure(U_gp,eps,gamma);
                if (p < eps) {    
                    Scalar t_low = 0.0, t_high = 1.0;
                    #pragma unroll
                    for (int iter = 0; iter < 50; ++iter) {
                        Scalar t_mid = 0.5 * (t_low + t_high);
                        Scalar U_mid[5];
                        #pragma unroll
                        for (int k = 0; k < 5; ++k)
                            U_mid[k] = (1.0 - t_mid) * U_avg[k] + t_mid * U_gp[k];
                        Scalar p_m = compute_pressure(U_mid,eps,gamma);
                        if (p_m < 0.0) t_high = t_mid;
                        else t_low = t_mid;
                        if ((t_high-t_low<1e-5)||(p_m*p_m<1e-12)) break;
                    }
                    theta_p = fmin(theta_p, t_low);
                }
            }
            {
                const auto& basis = DGBasisEvaluator<Order>::eval_all(1.0,0.0,0.0);

                Scalar U_gp[5];
                #pragma unroll
                for (int k = 0; k < 5; ++k) {
                    Scalar val = 0.0;
                    #pragma unroll
                    for (uInt l = 0; l < NumBasis; ++l)
                        val += basis[l] * coef[5*l + k];
                    U_gp[k] = val;
                }
                Scalar p = compute_pressure(U_gp,eps,gamma);
                if (p < eps) {    
                    Scalar t_low = 0.0, t_high = 1.0;
                    #pragma unroll
                    for (int iter = 0; iter < 50; ++iter) {
                        Scalar t_mid = 0.5 * (t_low + t_high);
                        Scalar U_mid[5];
                        #pragma unroll
                        for (int k = 0; k < 5; ++k)
                            U_mid[k] = (1.0 - t_mid) * U_avg[k] + t_mid * U_gp[k];
                        Scalar p_m = compute_pressure(U_mid,eps,gamma);
                        if (p_m < 0.0) t_high = t_mid;
                        else t_low = t_mid;
                        if ((t_high-t_low<1e-5)||(p_m*p_m<1e-12)) break;
                    }
                    theta_p = fmin(theta_p, t_low);
                }
            }
            {
                const auto& basis = DGBasisEvaluator<Order>::eval_all(0.0,1.0,0.0);

                Scalar U_gp[5];
                #pragma unroll
                for (int k = 0; k < 5; ++k) {
                    Scalar val = 0.0;
                    #pragma unroll
                    for (uInt l = 0; l < NumBasis; ++l)
                        val += basis[l] * coef[5*l + k];
                    U_gp[k] = val;
                }
                Scalar p = compute_pressure(U_gp,eps,gamma);
                if (p < eps) {    
                    Scalar t_low = 0.0, t_high = 1.0;
                    #pragma unroll
                    for (int iter = 0; iter < 50; ++iter) {
                        Scalar t_mid = 0.5 * (t_low + t_high);
                        Scalar U_mid[5];
                        #pragma unroll
                        for (int k = 0; k < 5; ++k)
                            U_mid[k] = (1.0 - t_mid) * U_avg[k] + t_mid * U_gp[k];
                        Scalar p_m = compute_pressure(U_mid,eps,gamma);
                        if (p_m < 0.0) t_high = t_mid;
                        else t_low = t_mid;
                        if ((t_high-t_low<1e-5)||(p_m*p_m<1e-12)) break;
                    }
                    theta_p = fmin(theta_p, t_low);
                }
            }
            {
                const auto& basis = DGBasisEvaluator<Order>::eval_all(0.0,0.0,1.0);

                Scalar U_gp[5];
                #pragma unroll
                for (int k = 0; k < 5; ++k) {
                    Scalar val = 0.0;
                    #pragma unroll
                    for (uInt l = 0; l < NumBasis; ++l)
                        val += basis[l] * coef[5*l + k];
                    U_gp[k] = val;
                }
                Scalar p = compute_pressure(U_gp,eps,gamma);
                if (p < eps) {    
                    Scalar t_low = 0.0, t_high = 1.0;
                    #pragma unroll
                    for (int iter = 0; iter < 50; ++iter) {
                        Scalar t_mid = 0.5 * (t_low + t_high);
                        Scalar U_mid[5];
                        #pragma unroll
                        for (int k = 0; k < 5; ++k)
                            U_mid[k] = (1.0 - t_mid) * U_avg[k] + t_mid * U_gp[k];
                        Scalar p_m = compute_pressure(U_mid,eps,gamma);
                        if (p_m < 0.0) t_high = t_mid;
                        else t_low = t_mid;
                        if ((t_high-t_low<1e-5)||(p_m*p_m<1e-12)) break;
                    }
                    theta_p = fmin(theta_p, t_low);
                }
            }
        }







        // 限制非 constant 模式
        #pragma unroll
        for (int k = 0; k < 5; ++k)
            #pragma unroll
            for (uInt l = 1; l < NumBasis; ++l)
                coef[5*l + k] *= theta_p;
    }
}

