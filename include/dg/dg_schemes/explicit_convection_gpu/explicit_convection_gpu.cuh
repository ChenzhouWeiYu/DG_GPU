// dg/dg_schemes/explicit_convection_gpu/explicit_convection_gpu.cuh
#pragma once
#include "base/type.h"
#include "dg/dg_basis/dg_basis.h"
#include "mesh/device_mesh.cuh"
#include "matrix/matrix.h"
#include "matrix/long_vector_device.cuh"
#include "dg/dg_flux/physical_flux/physical_flux_base.h"
#include "dg/dg_flux/numerical_flux/numerical_flux_base.h"

template<uInt Order = 3,
         uInt N_state = 5,
         typename GaussQuadCell = GaussLegendreTet::Auto,
         typename GaussQuadFace = GaussLegendreTri::Auto>
class ExplicitConvectionGPU {
private:
    using Basis = DGBasisEvaluator<Order>;
    static constexpr uInt N_basis = Basis::NumBasis;

    using QuadC = typename AutoQuadSelector<Order, GaussQuadCell>::type;
    using QuadF = typename AutoQuadSelector<Order, GaussQuadFace>::type;

    // 绑定物理通量和数值通量
    const PhysicalFlux<N_state - 5>& physical_flux;
    const NumericalFlux<N_state - 5>& numerical_flux;

public:
    // 构造函数：传入通量引用
    ExplicitConvectionGPU(
        const PhysicalFlux<N_state - 5>& pflux,
        const NumericalFlux<N_state - 5>& nflux
    ) : physical_flux(pflux), numerical_flux(nflux) {}

    // 三个积分项
    void eval_cells(
        const DeviceMesh& mesh,
        const LongVectorDevice<N_state * N_basis>& U,
        LongVectorDevice<N_state * N_basis>& rhs
    );

    void eval_internals(
        const DeviceMesh& mesh,
        const LongVectorDevice<N_state * N_basis>& U,
        LongVectorDevice<N_state * N_basis>& rhs
    );

    void eval_boundarys(
        const DeviceMesh& mesh,
        const LongVectorDevice<N_state * N_basis>& U,
        LongVectorDevice<N_state * N_basis>& rhs,
        Scalar time = 0.0
    );

    // 总入口
    void eval(
        const DeviceMesh& mesh,
        const LongVectorDevice<N_state * N_basis>& U,
        LongVectorDevice<N_state * N_basis>& rhs,
        Scalar time = 0.0
    );
};





#define Explicit_For_Flux(Order) \
extern template class ExplicitConvectionGPU<Order, 5,\
    typename AutoQuadSelector<Order, GaussLegendreTet::Auto>::type,\
    typename AutoQuadSelector<Order, GaussLegendreTri::Auto>::type>;

Explicit_For_Flux(1)
Explicit_For_Flux(2)

#undef Explicit_For_Flux