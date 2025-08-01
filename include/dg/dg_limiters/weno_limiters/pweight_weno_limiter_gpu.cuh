#pragma once

#include "base/type.h"
#include "dg/dg_basis/dg_basis.h"
#include "matrix/matrix.h"
#include "mesh/device_mesh.cuh"
#include "matrix/dense_matrix.h"
#include "matrix/long_vector_device.cuh"

// ---------------------- GPU WENO 限制器类 (HOST 端接口) -----------------------

template<uInt Order, typename QuadC, typename QuadF>
class PWeightWENOLimiterGPU {
public:
    using Basis = DGBasisEvaluator<Order>;
    static constexpr uInt NumBasis = Basis::NumBasis;

    PWeightWENOLimiterGPU(const DeviceMesh& device_mesh)
        : mesh_(device_mesh)
    {}

    void apply(LongVectorDevice<5 * NumBasis>& current_coeffs);

private:
    const DeviceMesh& mesh_;
};


// 显式实例化
#define explict_template_instantiation(Order) \
extern template class PWeightWENOLimiterGPU<Order, typename AutoQuadSelector<Order, GaussLegendreTet::Auto>::type, typename AutoQuadSelector<Order, GaussLegendreTri::Auto>::type>;

explict_template_instantiation(0)
explict_template_instantiation(1)
explict_template_instantiation(2)
explict_template_instantiation(3)
explict_template_instantiation(4)
explict_template_instantiation(5)
#undef explict_template_instantiation