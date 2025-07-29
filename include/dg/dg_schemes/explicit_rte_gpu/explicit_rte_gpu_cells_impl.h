#pragma once

#include "dg/dg_schemes/explicit_rte_gpu/explicit_rte_gpu.h"
#include "dg/dg_schemes/explicit_rte_gpu/explicit_rte_gpu_impl.h"

template<uInt OrderXYZ, uInt OrderOmega, uInt Nx, uInt Na,
         typename GaussQuadCell,
         typename GaussQuadTri, typename S2Mesh>
__global__ void eval_cells_rte_kernel(
    const GPUTetrahedron* mesh_cells, uInt num_cells,
    const DenseMatrix<Nx * Na * S2Mesh::num_cells, 1>* U,
    DenseMatrix<Nx * Na * S2Mesh::num_cells, 1>* rhs)
{
    using BasisXYZ = DGBasisEvaluator<OrderXYZ>;
    using BasisOmega = DGBasisEvaluator2D<OrderOmega>;

    constexpr auto Qx = GaussQuadCell::get_points();
    constexpr auto Wx = GaussQuadCell::get_weights();
    constexpr auto Qo = GaussQuadTri::get_points();
    constexpr auto Wo = GaussQuadTri::get_weights();

    uInt cid = blockIdx.x * blockDim.x + threadIdx.x;
    if (cid >= num_cells) return;

    const GPUTetrahedron& cell = mesh_cells[cid];
    const auto& coef = U[cid];

    constexpr uInt Ns = S2Mesh::num_cells;

    constexpr auto s2_cells = S2Mesh::s2_cells();
    for (uInt aid = 0; aid < Ns; ++aid) {
        const auto& s2_cell = s2_cells[aid];

        for (uInt ga = 0; ga < Qo.size(); ++ga) {
            const auto& xi_phi = Qo[ga];
            Scalar w_ang = Wo[ga] * 2.0 * s2_cell.area;

            auto phi_mu = s2_cell.map_to_physical(xi_phi);
            vector3f Omega = S2Mesh::spherical_to_cartesian(phi_mu);
            const auto& theta_vals = BasisOmega::eval_all(xi_phi[0], xi_phi[1]);

            for (uInt gx = 0; gx < Qx.size(); ++gx) {
                const auto& xi_x = Qx[gx];
                Scalar w_x = Wx[gx] * cell.volume * 6.0;

                const auto& phi_vals = BasisXYZ::eval_all(xi_x[0], xi_x[1], xi_x[2]);
                const auto& grad_phi = BasisXYZ::grad_all(xi_x[0], xi_x[1], xi_x[2]);

                Scalar psi_val = 0.0;
                for (uInt i = 0; i < Nx; ++i)
                    for (uInt j = 0; j < Na; ++j)
                        psi_val += coef(aid * (Nx * Na) + i * Na + j, 0) * phi_vals[i] * theta_vals[j];

                for (uInt jx = 0; jx < Nx; ++jx) {
                    auto grad = cell.invJac.multiply(grad_phi[jx]);
                    Scalar O_dot_grad = Omega[0]*grad[0] + Omega[1]*grad[1] + Omega[2]*grad[2];

                    for (uInt ja = 0; ja < Na; ++ja) {
                        Scalar integrand = psi_val * O_dot_grad * theta_vals[ja];
                        uInt idx = aid * (Nx * Na) + jx * Na + ja;
                        atomicAdd(&rhs[cid](idx, 0), -integrand * w_x * w_ang);
                    }
                }
            }
        }
    }
}


template<uInt OrderXYZ, uInt OrderOmega,
         typename GaussQuadCell, typename GaussQuadTri, typename S2Mesh>
void ExplicitRTEGPU<OrderXYZ, OrderOmega, GaussQuadCell, GaussQuadTri, S2Mesh>::eval_cells(
    const DeviceMesh& mesh,
    const LongVectorDevice<Nx * Na * S2Mesh::num_cells>& U,
    LongVectorDevice<Nx * Na * S2Mesh::num_cells>& rhs)
{
    dim3 block(32);
    dim3 grid((mesh.num_cells() + block.x - 1) / block.x);
    eval_cells_rte_kernel<OrderXYZ, OrderOmega, Nx, Na, QuadC, QuadA, S2Mesh><<<grid, block>>>(
        mesh.device_cells(), mesh.num_cells(), U.d_blocks, rhs.d_blocks);
}



























// #pragma once

// #include "base/type.h"
// #include "dg/dg_basis/dg_basis.h"
// #include "mesh/device_mesh.h"
// #include "matrix/matrix.h"
// #include "base/exact.h"
// // #include "dg/dg_flux/euler_physical_flux.h"
// #include "matrix/long_vector_device.h"


// class S2Mesh {
//     static constexpr uInt num_angle_cells = 20; // 黄金五角化十二面体太难写了，换成20面体

//     struct S2Cell {
//         std::array<vector2f, 3> vertices;  // phi, mu
//         Scalar area;                       // 球面三角形面积，等价于仿射变换Jacobian的一半
//         // DenseMatrix<2,2> J;                // 仿射变换Jacobian（参考→物理）
//         // DenseMatrix<2,2> JinvT;            // Jacobian逆转置

//         // 仅参考单元 (ξ, η) → 物理球面三角形 (φ, μ)
//         HostDevice vector2f map_to_physical(const vector2f& xi) const {
//             // const auto& J = get_J();
//             Scalar J00 = vertices[1][0] - vertices[0][0];
//             Scalar J01 = vertices[2][0] - vertices[0][0];
//             Scalar J10 = vertices[1][1] - vertices[0][1];
//             Scalar J11 = vertices[2][1] - vertices[0][1];
//             return {
//                 J00*xi[0] + J01*xi[1] + vertices[0][0],
//                 J10*xi[0] + J11*xi[1] + vertices[0][1]
//             };
//         }
//         HostDevice DenseMatrix<2,2> get_J() const {
//             return {
//                 vertices[1][0] - vertices[0][0],
//                 vertices[2][0] - vertices[0][0],
//                 vertices[1][1] - vertices[0][1],
//                 vertices[2][1] - vertices[0][1]
//             };
//         }
//         HostDevice DenseMatrix<2,2> get_JinvT() const {
//             const auto& J = get_J();
//             Scalar det = J(0,0)*J(1,1) - J(0,1)*J(1,0);
//             return {
//                 J(1,1)/det, -J(0,1)/det,
//                 -J(1,0)/det, J(0,0)/det
//             };
//         }
//     };

//     static constexpr std::array<S2Cell, 20> s2_cells = {
//         S2Cell{ vector2f{ 2.1243706856919418, 0.0000000000000000 }, vector2f{ 3.1415926535897931, 0.5257311121191336 }, vector2f{ 1.5707963267948966, 0.8506508083520400 }, 0.4787270691636970 },
//         S2Cell{ vector2f{ 2.1243706856919418, 0.0000000000000000 }, vector2f{ 1.5707963267948966, 0.8506508083520400 }, vector2f{ 1.0172219678978514, 0.0000000000000000 }, 0.4787270691636970 },
//         S2Cell{ vector2f{ 2.1243706856919418, 0.0000000000000000 }, vector2f{ 1.0172219678978514, 0.0000000000000000 }, vector2f{ 1.5707963267948966, -0.8506508083520400 }, 0.4787270691636970 },
//         S2Cell{ vector2f{ 2.1243706856919418, 0.0000000000000000 }, vector2f{ 1.5707963267948966, -0.8506508083520400 }, vector2f{ 3.1415926535897931, -0.5257311121191336 }, 0.4787270691636970 },
//         S2Cell{ vector2f{ 2.1243706856919418, 0.0000000000000000 }, vector2f{ 3.1415926535897931, -0.5257311121191336 }, vector2f{ 3.1415926535897931, 0.5257311121191336 }, 0.4787270691636970 },
//         S2Cell{ vector2f{ 1.0172219678978514, 0.0000000000000000 }, vector2f{ 1.5707963267948966, 0.8506508083520400 }, vector2f{ 0.0000000000000000, 0.5257311121191336 }, 0.4787270691636970 },
//         S2Cell{ vector2f{ 1.5707963267948966, 0.8506508083520400 }, vector2f{ 3.1415926535897931, 0.5257311121191336 }, vector2f{ -1.5707963267948966, 0.8506508083520400 }, 0.4787270691636970 },
//         S2Cell{ vector2f{ 3.1415926535897931, 0.5257311121191336 }, vector2f{ 3.1415926535897931, -0.5257311121191336 }, vector2f{ -2.1243706856919418, 0.0000000000000000 }, 0.4787270691636970 },
//         S2Cell{ vector2f{ 3.1415926535897931, -0.5257311121191336 }, vector2f{ 1.5707963267948966, -0.8506508083520400 }, vector2f{ -1.5707963267948966, -0.8506508083520400 }, 0.4787270691636970 },
//         S2Cell{ vector2f{ 1.5707963267948966, -0.8506508083520400 }, vector2f{ 1.0172219678978514, 0.0000000000000000 }, vector2f{ 0.0000000000000000, -0.5257311121191336 }, 0.4787270691636970 },
//         S2Cell{ vector2f{ -1.0172219678978514, 0.0000000000000000 }, vector2f{ 0.0000000000000000, 0.5257311121191336 }, vector2f{ -1.5707963267948966, 0.8506508083520400 }, 0.4787270691636970 },
//         S2Cell{ vector2f{ -1.0172219678978514, 0.0000000000000000 }, vector2f{ -1.5707963267948966, 0.8506508083520400 }, vector2f{ -2.1243706856919418, 0.0000000000000000 }, 0.4787270691636970 },
//         S2Cell{ vector2f{ -1.0172219678978514, 0.0000000000000000 }, vector2f{ -2.1243706856919418, 0.0000000000000000 }, vector2f{ -1.5707963267948966, -0.8506508083520400 }, 0.4787270691636970 },
//         S2Cell{ vector2f{ -1.0172219678978514, 0.0000000000000000 }, vector2f{ -1.5707963267948966, -0.8506508083520400 }, vector2f{ 0.0000000000000000, -0.5257311121191336 }, 0.4787270691636970 },
//         S2Cell{ vector2f{ -1.0172219678978514, 0.0000000000000000 }, vector2f{ 0.0000000000000000, -0.5257311121191336 }, vector2f{ 0.0000000000000000, 0.5257311121191336 }, 0.4787270691636970 },
//         S2Cell{ vector2f{ -1.5707963267948966, 0.8506508083520400 }, vector2f{ 0.0000000000000000, 0.5257311121191336 }, vector2f{ 1.5707963267948966, 0.8506508083520400 }, 0.4787270691636970 },
//         S2Cell{ vector2f{ -2.1243706856919418, 0.0000000000000000 }, vector2f{ -1.5707963267948966, 0.8506508083520400 }, vector2f{ 3.1415926535897931, 0.5257311121191336 }, 0.4787270691636970 },
//         S2Cell{ vector2f{ -1.5707963267948966, -0.8506508083520400 }, vector2f{ -2.1243706856919418, 0.0000000000000000 }, vector2f{ 3.1415926535897931, -0.5257311121191336 }, 0.4787270691636970 },
//         S2Cell{ vector2f{ 0.0000000000000000, -0.5257311121191336 }, vector2f{ -1.5707963267948966, -0.8506508083520400 }, vector2f{ 1.5707963267948966, -0.8506508083520400 }, 0.4787270691636970 },
//         S2Cell{ vector2f{ 0.0000000000000000, 0.5257311121191336 }, vector2f{ 0.0000000000000000, -0.5257311121191336 }, vector2f{ 1.0172219678978514, 0.0000000000000000 }, 0.4787270691636970 },
//     };
//     static vector3f spherical_to_cartesian(vector2f phi_mu) {
//         Scalar phi = phi_mu[0];
//         Scalar sin_theta = sqrt(1.0 - phi_mu[1]*phi_mu[1]);
//         Scalar cos_theta = phi_mu[1];
//         return {
//             sin_theta * cos(phi),
//             sin_theta * sin(phi),
//             cos_theta
//         };
//     }
// };




// template<uInt OrderXYZ, uInt OrderOmega, uInt Nx, uInt Na, typename S2Mesh, typename GaussQuadCell, typename GaussQuadTri>
// __global__ void eval_cells_rte_kernel(
//     const GPUTetrahedron* mesh_cells, uInt num_cells,
//     const DenseMatrix<Nx*Na,1>* U,
//     DenseMatrix<Nx*Na,1>* rhs)
// {
//     using BasisXYZ = DGBasisEvaluator<OrderXYZ>;
//     using BasisOmega = DGBasisEvaluator2D<OrderOmega>;
//     constexpr auto Qx = GaussQuadCell::get_points();      // 标准四面体积分点
//     constexpr auto Wx = GaussQuadCell::get_weights();
//     constexpr auto Qo = GaussQuadTri::get_points();       // 标准三角形积分点
//     constexpr auto Wo = GaussQuadTri::get_weights();

//     uInt cid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (cid >= num_cells) return;

//     const GPUTetrahedron& cell = mesh_cells[cid];
//     const DenseMatrix<Nx*Na,1>& coef = U[cid];

//     for (uInt aid = 0; aid < S2Mesh::cells.size(); ++aid) {
//         const auto& s2_cell = S2Mesh::s2_cells[aid]; 

//         for (uInt ga = 0; ga < Qo.size(); ++ga) {
//             const vector2f& xi = Qo[ga];
//             Scalar w_ang = Wo[ga] * 2.0 * s2_cell.area;
//             vector2f phi_mu = s2_cell.map_to_physical(xi);
//             vector3f Omega = S2Mesh::spherical_to_cartesian(phi_mu);

//             const auto& theta_vals = BasisOmega::eval_all(xi[0], xi[1]); 

//             for (uInt gx = 0; gx < Qx.size(); ++gx) {
//                 const auto& xi_x = Qx[gx];
//                 Scalar w_x = Wx[gx] * cell.volume * 6.0;

//                 const auto& phi_vals = BasisXYZ::eval_all(xi_x[0], xi_x[1], xi_x[2]);
//                 const auto& grad_phi = BasisXYZ::grad_all(xi_x[0], xi_x[1], xi_x[2]);

//                 // 计算 ψ(x,Ω)
//                 Scalar psi_val = 0.0;
//                 for (uInt i = 0; i < Nx; ++i)
//                     for (uInt j = 0; j < Na; ++j)
//                         psi_val += coef(i*Na + j, 0) * phi_vals[i] * theta_vals[j];

//                 for (uInt jx = 0; jx < Nx; ++jx) {
//                     const auto& grad_x = cell.invJac.multiply(grad_phi[jx]);
//                     Scalar OdotG = Omega[0] * grad_x[0] + Omega[1] * grad_x[1] + Omega[2] * grad_x[2];
//                     for (uInt ja = 0; ja < Na; ++ja) {
//                         Scalar integrand = psi_val * OdotG * theta_vals[ja];
//                         rhs[cid](jx*Na + ja, 0) -= integrand * w_x * w_ang;
//                     }
//                 }
//             }
//         }
//     }
// }
