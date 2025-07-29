// include/DG/DG_Schemes/PositiveLimiterGPU.h
#pragma once

#include "base/type.h"
#include "dg/dg_basis/dg_basis.h"
#include "matrix/matrix.h"
#include "mesh/device_mesh.h"
#include "matrix/dense_matrix.h"
#include "matrix/long_vector_device.h"
#include "PositiveLimiterGPU_impl.h"

// ---------------------- GPU 限制器类 (HOST 端接口) -----------------------

template<uInt Order, typename QuadType, bool OnlyNeigbAvg>
class PositiveLimiterGPU {
public:
    using Basis = DGBasisEvaluator<Order>;
    static constexpr uInt NumBasis = Basis::NumBasis;

    PositiveLimiterGPU(const DeviceMesh& device_mesh, Scalar gamma = 1.4)
        : mesh_(device_mesh), gamma_(gamma)
    {
        d_per_cell_max.resize(mesh_.num_cells());
        d_per_cell_min.resize(mesh_.num_cells());
        d_per_cell_min.fill_zeros();
        d_per_cell_max.fill_zeros();
            d_cell_max.resize(mesh_.num_cells());
            d_cell_min.resize(mesh_.num_cells());
            d_cell_min.fill_zeros();
            d_cell_max.fill_zeros();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        }
        cudaDeviceSynchronize();
    }

    void constructMinMax(const LongVectorDevice<5*NumBasis>& previous_coeffs)
    {
        dim3 block(32);
        dim3 grid_face((mesh_.num_faces() + block.x - 1) / block.x);
        dim3 grid_cell((mesh_.num_cells() + block.x - 1) / block.x);
        
        d_per_cell_min.fill_zeros();
        d_per_cell_max.fill_zeros();
            d_cell_min.fill_zeros();
            d_cell_max.fill_zeros();
        if (1){
            if (OnlyNeigbAvg) {
                // 如果只需要邻居平均，使用 cell_avg 的 kernel
                // 暂时忽略这里
                construct_cell_avg_extrema_kernel<Order, NumBasis, QuadType>
                    <<<grid_cell, block>>>(mesh_.device_cells(), mesh_.num_cells(),
                                    previous_coeffs.d_blocks,
                                    d_cell_min.d_blocks,
                                    d_cell_max.d_blocks);
            } else {
                construct_cell_extrema_kernel<Order, NumBasis, QuadType>
                    <<<grid_cell, block>>>(mesh_.device_cells(), mesh_.num_cells(),
                                    previous_coeffs.d_blocks,
                                    d_cell_min.d_blocks,
                                    d_cell_max.d_blocks);
            }
            gatter_cell_extrema_kernel<Order, NumBasis, QuadType>
                <<<grid_cell, block>>>(mesh_.device_cells(), mesh_.num_cells(),
                                    d_per_cell_min.d_blocks,
                                    d_per_cell_max.d_blocks,
                                    d_cell_min.d_blocks,
                                    d_cell_max.d_blocks);
        }
        else{
            constructMinMax_cell_kernel<Order, NumBasis, QuadType>
                    <<<grid_cell, block>>>(mesh_.device_cells(), mesh_.num_cells(),
                                    previous_coeffs.d_blocks,
                                    d_per_cell_min.d_blocks,
                                    d_per_cell_max.d_blocks);
        }
        
        
        
    }
    void apply(LongVectorDevice<5*NumBasis>& current_coeffs)
    {
        dim3 block(32);
        dim3 grid_face((mesh_.num_faces() + block.x - 1) / block.x);
        dim3 grid_cell((mesh_.num_cells() + block.x - 1) / block.x);

        apply_extrema_limiter_kernel<Order, NumBasis, QuadType><<<grid_cell, block>>>(
            mesh_.device_cells(), mesh_.num_cells(),
            current_coeffs.d_blocks,
            d_per_cell_min.d_blocks, d_per_cell_max.d_blocks);

        apply_2_kernel<Order, NumBasis, QuadType><<<grid_cell, block>>>(
            mesh_.device_cells(), mesh_.num_cells(),
            current_coeffs.d_blocks,
            gamma_);
    }

    void apply_1(LongVectorDevice<5*NumBasis>& current_coeffs)
    {
        dim3 block(32);
        dim3 grid_face((mesh_.num_faces() + block.x - 1) / block.x);
        dim3 grid_cell((mesh_.num_cells() + block.x - 1) / block.x);

        apply_extrema_limiter_kernel<Order, NumBasis, QuadType><<<grid_cell, block>>>(
            mesh_.device_cells(), mesh_.num_cells(),
            current_coeffs.d_blocks,
            d_per_cell_min.d_blocks, d_per_cell_max.d_blocks);
    }

    void apply_2(LongVectorDevice<5*NumBasis>& current_coeffs)
    {
        dim3 block(32);
        dim3 grid_face((mesh_.num_faces() + block.x - 1) / block.x);
        dim3 grid_cell((mesh_.num_cells() + block.x - 1) / block.x);

        apply_2_kernel<Order, NumBasis, QuadType><<<grid_cell, block>>>(
            mesh_.device_cells(), mesh_.num_cells(),
            current_coeffs.d_blocks,
            gamma_);
    }

public:
    const DeviceMesh& mesh_;
    Scalar gamma_;
    LongVectorDevice<5> d_per_cell_min, d_per_cell_max;
            LongVectorDevice<5> d_cell_min, d_cell_max;
};
