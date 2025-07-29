#pragma once
#include "mesh/device_mesh.h"          // DeviceMesh 定义
#include "matrix/long_vector_device.h"  // LongVectorDevice<DoFs>
#include "dg/dg_schemes/explicit_convection_gpu/explicit_convection_gpu.h"    // ExplicitConvectionGPU<Order>
#include "dg/dg_limiters/positive_limiters/positive_limiter_gpu.cuh"
#include "dg/dg_limiters/weno_limiters/weno_limiter_gpu.cuh"
#include "dg/dg_limiters/weno_limiters/pweight_weno_limiter_gpu.cuh"

enum class TimeIntegrationScheme {
    EULER,
    SSP_RK3
};

template<uInt DoFs, uInt Order = 1, bool OnlyNeigbAvg = false>
class TimeIntegrator {
public:
    using Scalar = double;

    explicit TimeIntegrator(const DeviceMesh& mesh,
                            LongVectorDevice<DoFs>& U_n,
                            const LongVectorDevice<DoFs>& r_mass,
                            PositiveLimiterGPU<Order, typename AutoQuadSelector<Order, GaussLegendreTet::Auto>::type, typename AutoQuadSelector<Order, GaussLegendreTri::Auto>::type, OnlyNeigbAvg>& positivelimiter,
                            WENOLimiterGPU<Order, typename AutoQuadSelector<Order, GaussLegendreTet::Auto>::type, typename AutoQuadSelector<Order, GaussLegendreTri::Auto>::type>& wenolimiter,
                            PWeightWENOLimiterGPU<Order, typename AutoQuadSelector<Order, GaussLegendreTet::Auto>::type, typename AutoQuadSelector<Order, GaussLegendreTri::Auto>::type>& pweightwenolimiter
                        );

    void set_scheme(TimeIntegrationScheme scheme);
    template<typename FluxType>
    void advance(ExplicitConvectionGPU<Order,FluxType>& convection, Scalar curr_time, Scalar dt, uInt limiter_flag = uInt(-1));

private:
    const DeviceMesh& mesh_;
    LongVectorDevice<DoFs>& U_n_;
    const LongVectorDevice<DoFs>& r_mass_;

    // 内部缓冲区（不暴露给外部）
    LongVectorDevice<DoFs> U_1_;
    LongVectorDevice<DoFs> U_2_;
    LongVectorDevice<DoFs> U_temp_;

    TimeIntegrationScheme scheme_;
    PositiveLimiterGPU<Order, typename AutoQuadSelector<Order, GaussLegendreTet::Auto>::type, typename AutoQuadSelector<Order, GaussLegendreTri::Auto>::type, OnlyNeigbAvg>& positivelimiter;
    WENOLimiterGPU<Order, typename AutoQuadSelector<Order, GaussLegendreTet::Auto>::type, typename AutoQuadSelector<Order, GaussLegendreTri::Auto>::type>& wenolimiter;
    PWeightWENOLimiterGPU<Order, typename AutoQuadSelector<Order, GaussLegendreTet::Auto>::type, typename AutoQuadSelector<Order, GaussLegendreTri::Auto>::type>& pweightwenolimiter;


};

// 必须显式实例化
// extern template class TimeIntegrator<5*DGBasisEvaluator<1>::NumBasis, 1>;
// extern template class TimeIntegrator<5*DGBasisEvaluator<2>::NumBasis, 2>;
// extern template class TimeIntegrator<5*DGBasisEvaluator<3>::NumBasis, 3>;
// extern template class TimeIntegrator<5*DGBasisEvaluator<4>::NumBasis, 4>;
// extern template class TimeIntegrator<5*DGBasisEvaluator<5>::NumBasis, 5>;


#define Explicit_For_Flux(NAME,Order) \
extern template void TimeIntegrator<5*DGBasisEvaluator<Order>::NumBasis, Order, true>::advance(ExplicitConvectionGPU<Order,NAME##75C>&,Scalar,Scalar,uInt);\
extern template void TimeIntegrator<5*DGBasisEvaluator<Order>::NumBasis, Order, true>::advance(ExplicitConvectionGPU<Order,NAME##53C>&,Scalar,Scalar,uInt);\
extern template void TimeIntegrator<5*DGBasisEvaluator<Order>::NumBasis, Order, false>::advance(ExplicitConvectionGPU<Order,NAME##75C>&,Scalar,Scalar,uInt);\
extern template void TimeIntegrator<5*DGBasisEvaluator<Order>::NumBasis, Order, false>::advance(ExplicitConvectionGPU<Order,NAME##53C>&,Scalar,Scalar,uInt);\


#define explict_template_instantiation(Order)\
extern template class TimeIntegrator<5*DGBasisEvaluator<Order>::NumBasis, Order, true>;\
extern template class TimeIntegrator<5*DGBasisEvaluator<Order>::NumBasis, Order, false>;\
FOREACH_FLUX_TYPE(Explicit_For_Flux,Order)\


explict_template_instantiation(0)
explict_template_instantiation(1)
explict_template_instantiation(2)
explict_template_instantiation(3)
explict_template_instantiation(4)
explict_template_instantiation(5)
#undef explict_template_instantiation

#undef Explicit_For_Flux

