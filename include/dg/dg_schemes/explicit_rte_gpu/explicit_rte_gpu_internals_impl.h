#pragma once

#include "dg/dg_schemes/explicit_rte_gpu/explicit_rte_gpu.h"
#include "dg/dg_schemes/explicit_rte_gpu/explicit_rte_gpu_impl.h"

template<uInt OrderXYZ, uInt OrderOmega, uInt Nx, uInt Na,
         typename GaussQuadCell, typename GaussQuadTri, typename S2Mesh>
__global__ void eval_rte_internals_kernel(
    const GPUTriangleFace* mesh_faces, uInt num_faces,
    const DenseMatrix<Nx * Na * S2Mesh::num_cells, 1>* U,
    DenseMatrix<Nx * Na * S2Mesh::num_cells, 1>* rhs)
{
    using BasisXYZ = DGBasisEvaluator<OrderXYZ>;
    using BasisOmega = DGBasisEvaluator2D<OrderOmega>;

    constexpr auto Qx = GaussQuadTri::get_points();
    constexpr auto Wx = GaussQuadTri::get_weights();

    constexpr auto Qo = GaussQuadTri::get_points();
    constexpr auto Wo = GaussQuadTri::get_weights();

    uInt fid = blockIdx.x * blockDim.x + threadIdx.x;
    if (fid >= num_faces) return;

    const GPUTriangleFace& face = mesh_faces[fid];
    const uInt cell_L = face.neighbor_cells[0];
    const uInt cell_R = face.neighbor_cells[1];
    if (cell_R == uInt(-1)) return;

    constexpr uInt Ns = S2Mesh::num_cells;
    constexpr auto s2_cells = S2Mesh::s2_cells();

    for (uInt aid = 0; aid < Ns; ++aid) {
        const auto& angle_cell = s2_cells[aid];

        for (uInt ga = 0; ga < Qo.size(); ++ga) {
            const vector2f& xi_ang = Qo[ga];
            Scalar w_ang = Wo[ga] * 2.0 * angle_cell.area;

            vector2f phi_mu = angle_cell.map_to_physical(xi_ang);
            vector3f Omega = S2Mesh::spherical_to_cartesian(phi_mu);

            Scalar lambda = fabs(Omega[0]*face.normal[0] +
                                 Omega[1]*face.normal[1] +
                                 Omega[2]*face.normal[2]);

            auto basis_ang = BasisOmega::eval_all(xi_ang[0], xi_ang[1]);

            for (uInt gx = 0; gx < Qx.size(); ++gx) {
                const vector2f& uv = Qx[gx];
                Scalar w_surf = Wx[gx] * face.area * 2.0;
                Scalar weight = w_ang * w_surf;

                auto xi_L = transform_to_cell(face, uv, 0);
                auto xi_R = transform_to_cell(face, uv, 1);

                auto basis_L = BasisXYZ::eval_all(xi_L[0], xi_L[1], xi_L[2]);
                auto basis_R = BasisXYZ::eval_all(xi_R[0], xi_R[1], xi_R[2]);

                for (uInt ja = 0; ja < Na; ++ja) {
                    Scalar psi_L = 0.0, psi_R = 0.0;
                    for (uInt i = 0; i < Nx; ++i) {
                        uInt idx_L = aid * (Nx * Na) + i * Na + ja;
                        uInt idx_R = idx_L;
                        psi_L += U[cell_L](idx_L, 0) * basis_L[i];
                        psi_R += U[cell_R](idx_R, 0) * basis_R[i];
                    }

                    Scalar dot = Omega[0]*face.normal[0] + Omega[1]*face.normal[1] + Omega[2]*face.normal[2];
                    Scalar flux = 0.5 * (psi_L + psi_R) * dot - 0.5 * lambda * (psi_R - psi_L);

                    for (uInt jx = 0; jx < Nx; ++jx) {
                        Scalar phiL = basis_L[jx], phiR = basis_R[jx];
                        uInt idx = aid * (Nx * Na) + jx * Na + ja;

                        atomicAdd(&rhs[cell_L](idx, 0),  flux * phiL * weight);
                        atomicAdd(&rhs[cell_R](idx, 0), -flux * phiR * weight);
                    }
                }
            }
        }
    }
}


template<uInt OrderXYZ, uInt OrderOmega,
         typename GaussQuadCell, typename GaussQuadTri, typename S2Mesh>
void ExplicitRTEGPU<OrderXYZ, OrderOmega, GaussQuadCell, GaussQuadTri, S2Mesh>::eval_internals(
    const DeviceMesh& mesh,
    const LongVectorDevice<Nx * Na * S2Mesh::num_cells>& U,
    LongVectorDevice<Nx * Na * S2Mesh::num_cells>& rhs)
{
    dim3 block(32);
    dim3 grid((mesh.num_faces() + block.x - 1) / block.x);
    eval_rte_internals_kernel<OrderXYZ, OrderOmega, Nx, Na, QuadC, QuadA, S2Mesh>
        <<<grid, block>>>(mesh.device_faces(), mesh.num_faces(), U.d_blocks, rhs.d_blocks);
}
