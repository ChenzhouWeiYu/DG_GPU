// include/DG/DG_Schemes/PositiveLimiterGPU_impl.h
#pragma once

#include "dg/dg_limiters/positive_limiters/positive_limiter_gpu.cuh"
#include "dg/dg_limiters/positive_limiters/positive_limiter_gpu_kernels.cuh"

template<uInt Order, typename QuadC, typename QuadF, bool OnlyNeigbAvg>
PositiveLimiterGPU<Order, QuadC, QuadF, OnlyNeigbAvg>::PositiveLimiterGPU(
    const DeviceMesh& device_mesh, Scalar gamma)
    : mesh_(device_mesh), gamma_(gamma)
{
    d_per_cell_max.resize(mesh_.num_cells());
    d_per_cell_min.resize(mesh_.num_cells());
    d_cell_max.resize(mesh_.num_cells());
    d_cell_min.resize(mesh_.num_cells());

    d_per_cell_min.fill_zeros();
    d_per_cell_max.fill_zeros();
    d_cell_min.fill_zeros();
    d_cell_max.fill_zeros();

    cudaDeviceSynchronize();
}

template<uInt Order, typename QuadC, typename QuadF, bool OnlyNeigbAvg>
void PositiveLimiterGPU<Order, QuadC, QuadF, OnlyNeigbAvg>::constructMinMax(
    const LongVectorDevice<5*NumBasis>& previous_coeffs)
{
    dim3 block(32);
    dim3 grid_cell((mesh_.num_cells() + block.x - 1) / block.x);

    d_per_cell_min.fill_zeros();
    d_per_cell_max.fill_zeros();
    d_cell_min.fill_zeros();
    d_cell_max.fill_zeros();

    if constexpr (OnlyNeigbAvg) {
        construct_cell_avg_extrema_kernel<Order, NumBasis, QuadC, QuadF><<<grid_cell, block>>>(
            mesh_.device_cells(), mesh_.num_cells(),
            previous_coeffs.d_blocks,
            d_cell_min.d_blocks,
            d_cell_max.d_blocks);
    } else {
        construct_cell_extrema_kernel<Order, NumBasis, QuadC, QuadF><<<grid_cell, block>>>(
            mesh_.device_cells(), mesh_.num_cells(),
            previous_coeffs.d_blocks,
            d_cell_min.d_blocks,
            d_cell_max.d_blocks);
    }

    gatter_cell_extrema_kernel<Order, NumBasis, QuadC, QuadF><<<grid_cell, block>>>(
        mesh_.device_cells(), mesh_.num_cells(),
        d_per_cell_min.d_blocks,
        d_per_cell_max.d_blocks,
        d_cell_min.d_blocks,
        d_cell_max.d_blocks);
}

template<uInt Order, typename QuadC, typename QuadF, bool OnlyNeigbAvg>
void PositiveLimiterGPU<Order, QuadC, QuadF, OnlyNeigbAvg>::apply(
    LongVectorDevice<5*NumBasis>& current_coeffs)
{
    apply_1(current_coeffs);
    apply_2(current_coeffs);
}

template<uInt Order, typename QuadC, typename QuadF, bool OnlyNeigbAvg>
void PositiveLimiterGPU<Order, QuadC, QuadF, OnlyNeigbAvg>::apply_1(
    LongVectorDevice<5*NumBasis>& current_coeffs)
{
    dim3 block(32);
    dim3 grid((mesh_.num_cells() + block.x - 1) / block.x);

    apply_extrema_limiter_kernel<Order, NumBasis, QuadC, QuadF><<<grid, block>>>(
        mesh_.device_cells(), mesh_.num_cells(),
        current_coeffs.d_blocks,
        d_per_cell_min.d_blocks,
        d_per_cell_max.d_blocks);
}

template<uInt Order, typename QuadC, typename QuadF, bool OnlyNeigbAvg>
void PositiveLimiterGPU<Order, QuadC, QuadF, OnlyNeigbAvg>::apply_2(
    LongVectorDevice<5*NumBasis>& current_coeffs)
{
    dim3 block(32);
    dim3 grid((mesh_.num_cells() + block.x - 1) / block.x);

    apply_2_kernel<Order, NumBasis, QuadC, QuadF><<<grid, block>>>(
        mesh_.device_cells(), mesh_.num_cells(),
        current_coeffs.d_blocks,
        gamma_);
}

