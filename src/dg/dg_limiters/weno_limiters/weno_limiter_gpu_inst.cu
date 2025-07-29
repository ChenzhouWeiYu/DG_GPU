// include/DG/DG_Limiters/WENOLimiterGPU_inst.cu
#include "dg/dg_limiters/weno_limiters/weno_limiter_gpu.cuh"
#include "dg/dg_limiters/weno_limiters/weno_limiter_gpu_impl.cuh"


// 显式实例化
#define explict_template_instantiation(Order) \
template class WENOLimiterGPU<Order, typename AutoQuadSelector<Order, GaussLegendreTet::Auto>::type, typename AutoQuadSelector<Order, GaussLegendreTri::Auto>::type>;

explict_template_instantiation(0)
explict_template_instantiation(1)
explict_template_instantiation(2)
explict_template_instantiation(3)
explict_template_instantiation(4)
explict_template_instantiation(5)
#undef explict_template_instantiation