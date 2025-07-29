#include "dg/dg_limiters/positive_limiters/positive_limiter_gpu.cuh"
#include "dg/dg_limiters/positive_limiters/positive_limiter_gpu_impl.cuh"
#include "dg/dg_limiters/positive_limiters/positive_limiter_gpu_kernels_impl.cuh"


// 显式实例化
#define explict_template_instantiation(Order) \
template class PositiveLimiterGPU<Order, typename AutoQuadSelector<Order, GaussLegendreTet::Auto>::type, typename AutoQuadSelector<Order, GaussLegendreTri::Auto>::type, false>;\
template class PositiveLimiterGPU<Order, typename AutoQuadSelector<Order, GaussLegendreTet::Auto>::type, typename AutoQuadSelector<Order, GaussLegendreTri::Auto>::type, true>;

// explict_template_instantiation(0)
// explict_template_instantiation(1)
explict_template_instantiation(2)
// explict_template_instantiation(3)
// explict_template_instantiation(4)
// explict_template_instantiation(5)
#undef explict_template_instantiation