#include "dg/dg_schemes/implicit_diffusion/implicit_diffusion.h"
#include "dg/dg_schemes/implicit_diffusion/implicit_diffusion_impl.h"
#include "dg/dg_flux/diffusion_physical_flux.h"





#define Explicit_For_Flux(Order) \
template class ImplicitDiffusion<Order,AirFluxD>;\
template class ImplicitDiffusion<Order,MonatomicFluxD>;\
template class ImplicitDiffusion<Order+1,AirFluxD>;\
template class ImplicitDiffusion<Order+1,MonatomicFluxD>;\
template class ImplicitDiffusion<Order+2,AirFluxD>;\
template class ImplicitDiffusion<Order+2,MonatomicFluxD>;

Explicit_For_Flux(0)
Explicit_For_Flux(3)

#undef Explicit_For_Flux