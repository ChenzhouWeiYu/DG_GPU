#include "dg/dg_schemes/explicit_convection_gpu/explicit_convection_gpu.cuh"
#include "dg/dg_schemes/explicit_convection_gpu/explicit_convection_gpu_kernels.cuh"
#include "dg/dg_schemes/explicit_convection_gpu/explicit_convection_gpu_kernels_impl.cuh"
#include "dg/dg_schemes/explicit_convection_gpu/explicit_convection_gpu_impl.cuh"


#define Explicit_For_Flux(NAME,Order) \
template class ExplicitConvectionGPU<Order,5>;

// FOREACH_FLUX_TYPE(Explicit_For_Flux,0)
// FOREACH_FLUX_TYPE(Explicit_For_Flux,1)
// FOREACH_FLUX_TYPE(Explicit_For_Flux,2)
// FOREACH_FLUX_TYPE(Explicit_For_Flux,3)
// FOREACH_FLUX_TYPE(Explicit_For_Flux,4)
// FOREACH_FLUX_TYPE(Explicit_For_Flux,5)

#undef Explicit_For_Flux


template class ExplicitConvectionGPU<3,5>;
template class ExplicitConvectionGPU<4,5>;
template class ExplicitConvectionGPU<5,5>;