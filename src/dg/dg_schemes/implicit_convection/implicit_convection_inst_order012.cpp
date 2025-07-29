#include "dg/dg_schemes/implicit_convection/implicit_convection.h"
#include "dg/dg_schemes/implicit_convection/implicit_convection_impl.h"
#include "dg/dg_flux/euler_physical_flux.h"


#define Explicit_For_Flux(Order) \
template class ImplicitConvection<Order,AirFluxC>;\
template class ImplicitConvection<Order,MonatomicFluxC>;\
template class ImplicitConvection<Order+1,AirFluxC>;\
template class ImplicitConvection<Order+1,MonatomicFluxC>;\
template class ImplicitConvection<Order+2,AirFluxC>;\
template class ImplicitConvection<Order+2,MonatomicFluxC>;

Explicit_For_Flux(0)
// Explicit_For_Flux(3)

#undef Explicit_For_Flux