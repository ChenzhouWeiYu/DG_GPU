// include/dg/dg_schemes/positive_preserving_limiter_gpu_impl.h
#pragma once
#include "dg/dg_limiters/positive_preserving_limiters/sampling_points.h"
#include "dg/dg_limiters/positive_preserving_limiters/positive_preserving_limiter_gpu.cuh"
#include "dg/dg_limiters/positive_preserving_limiters/positive_preserving_limiter_gpu_kernels.cuh"


// entropy_init.h
template<typename Physics, uInt Order>
Scalar compute_initial_s0(
    const LongVectorDevice<Physics::NEQN * DGBasisEvaluator<Order>::NumBasis>& U,
    const Physics& physics,
    uInt num_cells) {
    
    constexpr uInt NEQN = Physics::NEQN;
    constexpr uInt NumBasis = DGBasisEvaluator<Order>::NumBasis;

    // 1. 分配临时 GPU buffer
    uInt block_size = 256;
    uInt grid_size = (num_cells + block_size - 1) / block_size;
    Scalar* d_s0_buffer;
    cudaMalloc(&d_s0_buffer, grid_size * sizeof(Scalar));

    // 2. 启动 kernel
    size_t shared_mem = block_size * sizeof(Scalar);
    compute_initial_s0_kernel<Physics, NEQN, NumBasis><<<grid_size, block_size, shared_mem>>>(
        U.d_blocks, d_s0_buffer, num_cells, physics);

    // 3. 归约到 host
    std::vector<Scalar> h_s0_buffer(grid_size);
    cudaMemcpy(h_s0_buffer.data(), d_s0_buffer, grid_size * sizeof(Scalar), cudaMemcpyDeviceToHost);

    // for( auto& s0 : h_s0_buffer) {
    //     std::cout << s0 << std::endl;
    // }
    Scalar global_s0 = *std::min_element(h_s0_buffer.begin(), h_s0_buffer.end());
    // std::cout << "global_s0: " << global_s0 << std::endl;
    // getchar();

    // 4. 释放 GPU 内存
    cudaFree(d_s0_buffer);

    return global_s0;
}


template<typename Physics, uInt Order, typename QuadC, typename QuadF, uInt Level, uInt WithEntropy>
PositivityPreservingLimiterGPU<Physics, Order, QuadC, QuadF, Level, WithEntropy>::PositivityPreservingLimiterGPU(
    const DeviceMesh& mesh, const Physics& physics, Scalar s0)
    : mesh_(mesh), physics_(physics), s0_(s0) {
    
    // // 主机端生成表
    // static constexpr auto host_table = SamplingPoints<Order, QuadC, QuadF, Level>::basis_table;
    
    // // 上传到 GPU
    // cudaMalloc(&d_basis_table_, sizeof(host_table));
    // cudaMemcpy(d_basis_table_, &host_table, sizeof(host_table), cudaMemcpyHostToDevice);
}

// template<typename Physics, uInt Order, typename QuadC, typename QuadF, uInt Level>
// PositivityPreservingLimiterGPU<Physics, Order, QuadC, QuadF, Level>::~PositivityPreservingLimiterGPU() {
//     if (d_basis_table_) cudaFree(d_basis_table_);
// }


template<typename Physics, uInt Order, typename QuadC, typename QuadF, uInt Level, uInt WithEntropy>
void PositivityPreservingLimiterGPU<Physics, Order, QuadC, QuadF, Level, WithEntropy>::apply(
    LongVectorDevice<NEQN*NumBasis>& U) {
    
    dim3 block(256);
    dim3 grid((mesh_.num_cells() + block.x - 1) / block.x);

    apply_positivity_limiter_kernel<Physics, Order, NumBasis, NumSamples, QuadC, QuadF, Level, WithEntropy>
        <<<grid, block>>> (mesh_.view(), U.d_blocks, physics_, s0_);
    
    // apply_positivity_limiter_kernel_table<Physics, Order, NumBasis, NumSamples, QuadC, QuadF, Level>
    //     <<<grid, block>>> (mesh_.view(), U.d_blocks, physics_);
}
