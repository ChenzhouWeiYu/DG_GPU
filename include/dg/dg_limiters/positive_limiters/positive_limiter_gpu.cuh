// include/DG/DG_Schemes/PositiveLimiterGPU.h
#pragma once

#include "base/type.h"
#include "matrix/dense_matrix.h"
#include "matrix/long_vector_device.h"
#include "mesh/device_mesh.h"
#include "dg/dg_basis/dg_basis.h"
#include "dg/dg_limiters/positive_limiters/positive_limiter_gpu_kernels.cuh"

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

// 显式实例化声明（可补充）
#define explict_template_instantiation(Order) \
extern template class PositiveLimiterGPU<Order, typename AutoQuadSelector<Order, GaussLegendreTet::Auto>::type, typename AutoQuadSelector<Order, GaussLegendreTri::Auto>::type, false>;\
extern template class PositiveLimiterGPU<Order, typename AutoQuadSelector<Order, GaussLegendreTet::Auto>::type, typename AutoQuadSelector<Order, GaussLegendreTri::Auto>::type, true>;

explict_template_instantiation(0)
explict_template_instantiation(1)
explict_template_instantiation(2)
explict_template_instantiation(3)
explict_template_instantiation(4)
explict_template_instantiation(5)
#undef explict_template_instantiation