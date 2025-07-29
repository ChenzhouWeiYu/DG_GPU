#include "dg/dg_schemes/explicit_convection/explicit_convection.h"
#include "dg/dg_schemes/explicit_convection/explicit_convection_impl.h"




#define Explicit_For_Flux(Order) \
template class ExplicitConvection<Order,AirFluxC>;\
template class ExplicitConvection<Order,MonatomicFluxC>;\
template class ExplicitConvection<Order+1,AirFluxC>;\
template class ExplicitConvection<Order+1,MonatomicFluxC>;\
template class ExplicitConvection<Order+2,AirFluxC>;\
template class ExplicitConvection<Order+2,MonatomicFluxC>;

// Explicit_For_Flux(0)
Explicit_For_Flux(3)

#undef Explicit_For_Flux