#pragma once
#include "mesh/device_mesh.h"          // DeviceMesh 定义
#include "matrix/long_vector_device.h"  // LongVectorDevice<DoFs>
#include "dg/dg_schemes/explicit_convection_gpu.h"    // ExplicitConvectionGPU<Order>

enum class TimeIntegrationScheme {
    EULER,
    SSP_RK3
};

template<uInt DoFs, uInt Order>
class TimeIntegrator {
public:
    using Scalar = double;

    explicit TimeIntegrator(const DeviceMesh& mesh,
                            LongVectorDevice<DoFs>& U_n,
                            const LongVectorDevice<DoFs>& r_mass,
                            Scalar CFL = 0.5);

    void set_scheme(TimeIntegrationScheme scheme);
    void advance(ExplicitConvectionGPU<Order>& convection, Scalar curr_time, Scalar dt);

private:
    const DeviceMesh& mesh_;
    LongVectorDevice<DoFs>& U_n_;
    const LongVectorDevice<DoFs>& r_mass_;

    // 内部缓冲区（不暴露给外部）
    LongVectorDevice<DoFs> U_1_;
    LongVectorDevice<DoFs> U_2_;
    LongVectorDevice<DoFs> U_temp_;

    TimeIntegrationScheme scheme_;
    Scalar CFL_;


};

// 必须显式实例化
extern template class TimeIntegrator<5*DGBasisEvaluator<1>::NumBasis, 1>;
extern template class TimeIntegrator<5*DGBasisEvaluator<2>::NumBasis, 2>;
extern template class TimeIntegrator<5*DGBasisEvaluator<3>::NumBasis, 3>;
extern template class TimeIntegrator<5*DGBasisEvaluator<4>::NumBasis, 4>;
extern template class TimeIntegrator<5*DGBasisEvaluator<5>::NumBasis, 5>;
