#pragma once

#include "base/type.h"
#include "dg/dg_basis/dg_basis.h"
#include "mesh/device_mesh.h"
#include "matrix/matrix.h"
#include "matrix/long_vector_device.h"
#include "base/exact.h"
#include "dg/dg_schemes/explicit_rte_gpu/s2_mesh_icosahedral.h"

// S² × 𝔻 GPU 显式离散求解器
template<uInt OrderXYZ, uInt OrderOmega,
         typename GaussQuadCell = GaussLegendreTet::Auto,
         typename GaussQuadTri  = GaussLegendreTri::Auto,
         typename S2Mesh = S2MeshIcosahedral>
class ExplicitRTEGPU {
private:
    using BasisXYZ = DGBasisEvaluator<OrderXYZ>;
    using BasisOmega = DGBasisEvaluator2D<OrderOmega>;

    using QuadC = typename std::conditional_t<
        std::is_same_v<GaussQuadCell, GaussLegendreTet::Auto>,
        typename AutoQuadSelector<OrderXYZ, GaussLegendreTet::Auto>::type,
        GaussQuadCell>;

    using QuadA = typename std::conditional_t<
        std::is_same_v<GaussQuadTri, GaussLegendreTri::Auto>,
        typename AutoQuadSelector<OrderOmega, GaussLegendreTri::Auto>::type,
        GaussQuadTri>;

    static constexpr uInt Nx = BasisXYZ::NumBasis;
    static constexpr uInt Na = BasisOmega::NumBasis;

    
    __device__ inline vector3f transform_to_cell(const GPUTriangleFace& face, const vector2f& uv, uInt side) {
        const vector3f& nc0 = face.natural_coords[side][0];
        const vector3f& nc1 = face.natural_coords[side][1];
        const vector3f& nc2 = face.natural_coords[side][2];
        
        Scalar uv0 = 1 - uv[0] - uv[1];
        Scalar uv1 = uv[0];
        Scalar uv2 = uv[1];

        // 手动展开每个分量的乘法和加法
        Scalar x = nc0[0] * uv0 + nc1[0] * uv1 + nc2[0] * uv2;
        Scalar y = nc0[1] * uv0 + nc1[1] * uv1 + nc2[1] * uv2;
        Scalar z = nc0[2] * uv0 + nc1[2] * uv1 + nc2[2] * uv2;

        vector3f result{x, y, z};
        return result;
    }

public:
    // kernel launcher
    void eval_cells(const DeviceMesh& mesh,
                    const LongVectorDevice<Nx*Na*S2Mesh::num_cells>& U,
                    LongVectorDevice<Nx*Na*S2Mesh::num_cells>& rhs);

    void eval_internals(const DeviceMesh& mesh,
                        const LongVectorDevice<Nx*Na*S2Mesh::num_cells>& U,
                        LongVectorDevice<Nx*Na*S2Mesh::num_cells>& rhs);

    void eval_boundarys(const DeviceMesh& mesh,
                        const LongVectorDevice<Nx*Na*S2Mesh::num_cells>& U,
                        LongVectorDevice<Nx*Na*S2Mesh::num_cells>& rhs,
                        Scalar time = 0.0);

    void eval(const DeviceMesh& mesh,
              const LongVectorDevice<Nx*Na*S2Mesh::num_cells>& U,
              LongVectorDevice<Nx*Na*S2Mesh::num_cells>& rhs,
              Scalar time = 0.0);
};




#define Explicit_For_Flux(Order) \
extern template class ExplicitRTEGPU<Order, Order,\
    typename AutoQuadSelector<Order, GaussLegendreTet::Auto>::type,\
    typename AutoQuadSelector<Order, GaussLegendreTri::Auto>::type>;

Explicit_For_Flux(1)
Explicit_For_Flux(2)

#undef Explicit_For_Flux

// #include "dg/dg_schemes/explicit_rte_gpu_impl/explicit_rte_gpu_impl.h"