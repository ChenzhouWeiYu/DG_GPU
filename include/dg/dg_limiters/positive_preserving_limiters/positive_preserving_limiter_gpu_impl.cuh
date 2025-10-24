// include/dg/dg_schemes/positive_preserving_limiter_gpu_impl.h
#pragma once
#include "dg/dg_limiters/positive_preserving_limiters/sampling_points.h"
#include "dg/dg_limiters/positive_preserving_limiters/positive_preserving_limiter_gpu.cuh"
#include "dg/dg_limiters/positive_preserving_limiters/positive_preserving_limiter_gpu_kernels.cuh"

template<typename Physics, uInt Order, typename QuadC, typename QuadF, uInt Level>
PositivityPreservingLimiterGPU<Physics, Order, QuadC, QuadF, Level>::PositivityPreservingLimiterGPU(
    const DeviceMesh& mesh, const Physics& physics)
    : mesh_(mesh), physics_(physics) {
    
    // 主机端生成表
    static constexpr auto host_table = SamplingPoints<Order, QuadC, QuadF, Level>::basis_table;
    
    // 上传到 GPU
    cudaMalloc(&d_basis_table_, sizeof(host_table));
    cudaMemcpy(d_basis_table_, &host_table, sizeof(host_table), cudaMemcpyHostToDevice);
}

template<typename Physics, uInt Order, typename QuadC, typename QuadF, uInt Level>
PositivityPreservingLimiterGPU<Physics, Order, QuadC, QuadF, Level>::~PositivityPreservingLimiterGPU() {
    if (d_basis_table_) cudaFree(d_basis_table_);
}


template<typename Physics, uInt Order, typename QuadC, typename QuadF, uInt Level>
void PositivityPreservingLimiterGPU<Physics, Order, QuadC, QuadF, Level>::apply(
    LongVectorDevice<NEQN*NumBasis>& U) {
    
    dim3 block(256);
    dim3 grid((mesh_.num_cells() + block.x - 1) / block.x);
    // size_t shared_mem_size = sizeof(std::array<std::array<Scalar, NumBasis>, NumSamples>);
    apply_positivity_limiter_kernel<Physics, Order, NumBasis, NumSamples, QuadC, QuadF, Level>
        <<<grid, block>>> (mesh_.view(), U.d_blocks, physics_);
}
