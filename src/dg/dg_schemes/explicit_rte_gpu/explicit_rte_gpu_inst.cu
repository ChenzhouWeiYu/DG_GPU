#include "dg/dg_schemes/explicit_rte_gpu/explicit_rte_gpu.h"
#include "dg/dg_schemes/explicit_rte_gpu/explicit_rte_gpu_cells_impl.h"
#include "dg/dg_schemes/explicit_rte_gpu/explicit_rte_gpu_boundarys_impl.h"
#include "dg/dg_schemes/explicit_rte_gpu/explicit_rte_gpu_internals_impl.h"
#include "dg/dg_schemes/explicit_rte_gpu/explicit_rte_gpu_impl.h"



#define Explicit_For_Flux(Order) \
template class ExplicitRTEGPU<Order, Order,\
    typename AutoQuadSelector<Order, GaussLegendreTet::Auto>::type,\
    typename AutoQuadSelector<Order, GaussLegendreTri::Auto>::type>;

Explicit_For_Flux(1)
Explicit_For_Flux(2)

#undef Explicit_For_Flux
