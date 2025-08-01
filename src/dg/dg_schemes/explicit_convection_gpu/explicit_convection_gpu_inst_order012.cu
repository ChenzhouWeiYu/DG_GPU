#include "dg/dg_schemes/explicit_convection_gpu/explicit_convection_gpu.cuh"
#include "dg/dg_schemes/explicit_convection_gpu/explicit_convection_gpu_cells_impl.cuh"
#include "dg/dg_schemes/explicit_convection_gpu/explicit_convection_gpu_boundarys_impl.cuh"
#include "dg/dg_schemes/explicit_convection_gpu/explicit_convection_gpu_internals_impl.cuh"
#include "dg/dg_schemes/explicit_convection_gpu/explicit_convection_gpu_impl.cuh"



// #define Explicit_For_Flux(Order)\
// template class ExplicitConvectionGPU<Order,LF75C>;\
// template class ExplicitConvectionGPU<Order,LF53C>;\
// template class ExplicitConvectionGPU<Order,HLL75C>;\
// template class ExplicitConvectionGPU<Order,HLL53C>;\
// template class ExplicitConvectionGPU<Order,HLLC75C>;\
// template class ExplicitConvectionGPU<Order,HLLC53C>;\
// template class ExplicitConvectionGPU<Order,LaxFriedrichs75C>;\
// template class ExplicitConvectionGPU<Order,LaxFriedrichs53C>;

#define Explicit_For_Flux(NAME,Order) \
template class ExplicitConvectionGPU<Order,NAME##75C>;\
template class ExplicitConvectionGPU<Order,NAME##53C>;

FOREACH_FLUX_TYPE(Explicit_For_Flux,0)
FOREACH_FLUX_TYPE(Explicit_For_Flux,1)
FOREACH_FLUX_TYPE(Explicit_For_Flux,2)
// FOREACH_FLUX_TYPE(Explicit_For_Flux,3)
// FOREACH_FLUX_TYPE(Explicit_For_Flux,4)
// FOREACH_FLUX_TYPE(Explicit_For_Flux,5)

#undef Explicit_For_Flux

// Explicit_For_Flux(0)
// Explicit_For_Flux(1)
// Explicit_For_Flux(2)
// Explicit_For_Flux(3)
// Explicit_For_Flux(4)
// Explicit_For_Flux(5)

// #undef Explicit_For_Flux