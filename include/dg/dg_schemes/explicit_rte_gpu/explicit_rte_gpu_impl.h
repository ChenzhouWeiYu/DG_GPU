#pragma once

#include "dg/dg_schemes/explicit_rte_gpu/explicit_rte_gpu.h"


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


template<uInt OrderXYZ, uInt OrderOmega, typename GaussQuadCell,
         typename GaussQuadTri, typename S2Mesh>
void ExplicitRTEGPU<OrderXYZ, OrderOmega, GaussQuadCell, GaussQuadTri, S2Mesh>::eval(
    const DeviceMesh& mesh,
    const LongVectorDevice<Nx * Na * S2Mesh::num_cells>& U,
    LongVectorDevice<Nx * Na * S2Mesh::num_cells>& rhs,
    Scalar time)
{
    eval_cells(mesh, U, rhs);
    eval_internals(mesh, U, rhs);
    eval_boundarys(mesh, U, rhs, time);
}
