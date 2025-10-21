#pragma once

#include "base/type.h"
#include "dg/dg_basis/dg_basis.h"
#include "mesh/device_mesh.cuh"
#include "matrix/matrix.h"
#include "base/exact.h"
#include "dg/dg_flux/euler_physical_flux.h"
#include "matrix/long_vector_device.cuh"
#include "dg/physics/ideal_gas_physics.h"
#include "dg/flux_schemes/hllc_flux.h"

// GPU 显式对流核
template<typename Physics, typename FluxScheme, uInt Order=3, 
         typename GaussQuadCell = GaussLegendreTet::Auto, 
         typename GaussQuadFace = GaussLegendreTri::Auto>
class ExplicitConvectionGPU {
private:
    using Basis = DGBasisEvaluator<Order>;
    
    using QuadC = typename std::conditional_t<
        std::is_same_v<GaussQuadCell, GaussLegendreTet::Auto>,
        typename AutoQuadSelector<Order, GaussLegendreTet::Auto>::type,
        GaussQuadCell
    >;
    using QuadF = typename std::conditional_t<
        std::is_same_v<GaussQuadFace, GaussLegendreTri::Auto>,
        typename AutoQuadSelector<Order, GaussLegendreTri::Auto>::type,
        GaussQuadFace
    >;

    static constexpr uInt NEQN = Physics::NEQN;
    static constexpr uInt NBIS = Basis::NumBasis;
    
    Physics physics_; // 物理模型
public:
    ExplicitConvectionGPU(const Physics& physics) : physics_(physics) {}
    // 3个 kernel launcher
    void eval_cells(const DeviceMesh& mesh, 
                    const LongVectorDevice<NEQN*NBIS>& U,
                    LongVectorDevice<NEQN*NBIS>& rhs);

    void eval_faces(const DeviceMesh& mesh, 
                        const LongVectorDevice<NEQN*NBIS>& U,
                        LongVectorDevice<NEQN*NBIS>& rhs, Scalar time = 0.0);
                    
    void eval_internals(const DeviceMesh& mesh, 
                        const LongVectorDevice<NEQN*NBIS>& U,
                        LongVectorDevice<NEQN*NBIS>& rhs);
                        
    void eval_boundarys(const DeviceMesh& mesh, 
                        const LongVectorDevice<NEQN*NBIS>& U,
                        LongVectorDevice<NEQN*NBIS>& rhs, Scalar time = 0.0);
                        

    void eval(const DeviceMesh& mesh, 
                        const LongVectorDevice<NEQN*NBIS>& U,
                        LongVectorDevice<NEQN*NBIS>& rhs, Scalar time = 0.0);
};









// #define Explicit_For_Flux(Order)\
// extern template class ExplicitConvectionGPU<Order,LF75C>;\
// extern template class ExplicitConvectionGPU<Order,LF53C>;\
// extern template class ExplicitConvectionGPU<Order,Roe75C>;\
// extern template class ExplicitConvectionGPU<Order,Roe53C>;\
// extern template class ExplicitConvectionGPU<Order,HLL75C>;\
// extern template class ExplicitConvectionGPU<Order,HLL53C>;\
// extern template class ExplicitConvectionGPU<Order,HLLC75C>;\
// extern template class ExplicitConvectionGPU<Order,HLLC53C>;\
// extern template class ExplicitConvectionGPU<Order,LaxFriedrichs75C>;\
// extern template class ExplicitConvectionGPU<Order,LaxFriedrichs53C>;

// #define Explicit_For_Flux(NAME,Order) \
// extern template class ExplicitConvectionGPU<Order,NAME##75C>;\
// extern template class ExplicitConvectionGPU<Order,NAME##53C>;

// FOREACH_FLUX_TYPE(Explicit_For_Flux,0)
// FOREACH_FLUX_TYPE(Explicit_For_Flux,1)
// FOREACH_FLUX_TYPE(Explicit_For_Flux,2)
// FOREACH_FLUX_TYPE(Explicit_For_Flux,3)
// FOREACH_FLUX_TYPE(Explicit_For_Flux,4)
// FOREACH_FLUX_TYPE(Explicit_For_Flux,5)

// #undef Explicit_For_Flux
// Explicit_For_Flux(0)
// Explicit_For_Flux(1)
// Explicit_For_Flux(2)
// Explicit_For_Flux(3)
// Explicit_For_Flux(4)
// Explicit_For_Flux(5)

// #undef Explicit_For_Flux