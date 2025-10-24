// include/dg/dg_schemes/positive_limiter_gpu.h
#pragma once

#include "base/type.h"
#include "matrix/dense_matrix.h"
#include "matrix/long_vector_device.cuh"
#include "mesh/device_mesh.cuh"
#include "dg/dg_basis/dg_basis.h"
#include "dg/dg_limiters/positive_limiters_backup/positive_limiter_gpu_kernels.cuh"

template<uInt Order, typename QuadC, typename QuadF, bool OnlyNeigbAvg>
class PositiveLimiterGPU {
public:
    using Basis = DGBasisEvaluator<Order>;
    static constexpr uInt NumBasis = Basis::NumBasis;

    PositiveLimiterGPU(const DeviceMesh& device_mesh, Scalar gamma = 1.4);

    void constructMinMax(const LongVectorDevice<5*NumBasis>& previous_coeffs);
    void apply(LongVectorDevice<5*NumBasis>& current_coeffs);
    void apply_1(LongVectorDevice<5*NumBasis>& current_coeffs);
    void apply_2(LongVectorDevice<5*NumBasis>& current_coeffs);

private:
    const DeviceMesh& mesh_;
    Scalar gamma_;
    LongVectorDevice<5> d_per_cell_min, d_per_cell_max;
    LongVectorDevice<5> d_cell_min, d_cell_max;
};