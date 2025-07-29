#pragma once
#include "mesh/device_mesh.h"          // DeviceMesh 定义
#include "matrix/long_vector_device.h"  // LongVectorDevice<DoFs>
#include "dg/dg_schemes/explicit_rte_gpu_impl/explicit_rte_gpu.h"
#include "dg/time_integrator.h"


template<uInt X3Order, uInt S2Order,
         typename GaussQuadCell = GaussLegendreTet::Auto,
         typename GaussQuadTri  = GaussLegendreTri::Auto,
         typename S2Mesh = S2MeshIcosahedral>
class RTE_TimeIntegrator {
public:
    
    using X3Basis = DGBasisEvaluator<X3Order>;

    using S2Basis = DGBasisEvaluator2D<S2Order>;

    static constexpr uInt X3DoFs = X3Basis::NumBasis;
    static constexpr uInt S2DoFs = S2Basis::NumBasis;
    static constexpr uInt S2Cells = S2Mesh::num_cells;
    static constexpr auto s2_cells = S2Mesh::s2_cells();
    static constexpr uInt DoFs = X3DoFs * S2DoFs * S2Cells;

    RTE_TimeIntegrator(const DeviceMesh& mesh,
                            LongVectorDevice<DoFs>& U_n,
                            const LongVectorDevice<DoFs>& r_mass);

    void set_scheme(TimeIntegrationScheme scheme);
    void advance(ExplicitRTEGPU<X3Order,S2Order,GaussQuadCell,GaussQuadTri,S2Mesh>& convection, Scalar curr_time, Scalar dt);

private:
    const DeviceMesh& mesh_;
    LongVectorDevice<DoFs>& U_n_;
    const LongVectorDevice<DoFs>& r_mass_;

    // 内部缓冲区（不暴露给外部）
    LongVectorDevice<DoFs> U_1_;
    LongVectorDevice<DoFs> U_2_;
    LongVectorDevice<DoFs> U_temp_;

    TimeIntegrationScheme scheme_;
};


#define explict_template_instantiation(Order)\
extern template class RTE_TimeIntegrator<Order, Order,\
    typename AutoQuadSelector<Order, GaussLegendreTet::Auto>::type,\
    typename AutoQuadSelector<Order, GaussLegendreTri::Auto>::type>;


explict_template_instantiation(1)
explict_template_instantiation(2)
#undef explict_template_instantiation



