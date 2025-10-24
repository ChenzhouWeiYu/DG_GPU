// include/dg/dg_limiters/positive_preserving_limiters/positive_preserving_limiter_gpu.h
#pragma once
#include "base/type.h"
#include "matrix/dense_matrix.h"
#include "matrix/long_vector_device.cuh"
#include "mesh/device_mesh.cuh"
#include "dg/dg_basis/dg_basis.h"
#include "dg/dg_limiters/positive_preserving_limiters/positive_preserving_limiter_gpu_kernels.cuh"
#include "dg/dg_limiters/positive_preserving_limiters/sampling_points.h"

template<typename Physics, uInt Order, typename QuadC, typename QuadF, uInt Level = 2>
class PositivityPreservingLimiterGPU {
private:
    using Basis = DGBasisEvaluator<Order>;
    static constexpr uInt NumBasis = Basis::NumBasis;
    static constexpr uInt NEQN = Physics::NEQN;
    static constexpr uInt NumSamples = SamplingPoints<Order, QuadC, QuadF, Level>::num_samples;

public:
    PositivityPreservingLimiterGPU(const DeviceMesh& mesh, const Physics& physics);
    // ~PositivityPreservingLimiterGPU();

    void apply(LongVectorDevice<NEQN*NumBasis>& U);

private:
    const DeviceMesh& mesh_;
    Physics physics_;
    // 预计算的基函数表（POD）
    // std::array<std::array<Scalar, NumBasis>, NumSamples>* d_basis_table_ = nullptr;
};
