// dg/dg_schemes/explicit_convection_gpu/explicit_convection_gpu_kernels_impl.cuh
#pragma once
#include "dg/dg_schemes/explicit_convection_gpu/explicit_convection_gpu_kernels.cuh"
#include "base/exact.h"

template<uInt Order, uInt N_state, typename GaussQuadCell, typename GaussQuadFace>
__global__ void eval_cells_kernel(
    const GPUTetrahedron* cells,
    uInt num_cells,
    const DenseMatrix<N_state * DGBasisEvaluator<Order>::NumBasis, 1>* U,
    DenseMatrix<N_state * DGBasisEvaluator<Order>::NumBasis, 1>* rhs,
    const PhysicalFlux<N_state - 5>* physical_flux
) {
    using Basis = DGBasisEvaluator<Order>;
    static constexpr uInt N_basis = Basis::NumBasis;
    constexpr uInt num_points = GaussQuadCell::num_points;
    constexpr auto Qpoints = GaussQuadCell::get_points();
    constexpr auto Qweights = GaussQuadCell::get_weights();

    uInt cid = blockIdx.x * blockDim.x + threadIdx.x;
    if (cid >= num_cells) return;

    const auto& cell = cells[cid];
    const auto& coef = U[cid];
    DenseMatrix<N_state * N_basis, 1> result = DenseMatrix<N_state * N_basis, 1>::Zeros();

    for (uInt g = 0; g < num_points; ++g) {
        const vector3f& xi = Qpoints[g];
        Scalar w = Qweights[g] * cell.volume * 6;

        auto basis = Basis::eval_all(xi[0], xi[1], xi[2]);
        auto grads = Basis::grad_all(xi[0], xi[1], xi[2]);

        auto U_val = dg_utils::reconstruct_state<Order, N_state, N_basis>(coef, basis);
        // auto G_val = dg_utils::reconstruct_gradient<Order, N_state, N_basis>(coef, grads, cell.invJac);

        auto prim = physical_flux->thermo.reconstruct(U_val);
        
        for (uInt j = 0; j < N_basis; ++j) {
            auto grad_phys = cell.invJac.multiply(grads[j]);
            auto F_dot = physical_flux->compute_dot(prim, grad_phys);
            PragmaUnroll
            for (uInt k = 0; k < N_state; ++k) {
                result(N_state * j + k, 0) -= F_dot(k, 0) * w;
            }
        }
    }
    rhs[cid] = result;
}






template<uInt Order, uInt N_state, typename GaussQuadFace>
__global__ void eval_internals_kernel(
    const GPUTriangleFace* faces,
    uInt num_faces,
    const DenseMatrix<N_state * DGBasisEvaluator<Order>::NumBasis, 1>* U,
    DenseMatrix<N_state * DGBasisEvaluator<Order>::NumBasis, 1>* rhs,
    const NumericalFlux<N_state - 5>* numerical_flux
) {
    using Basis = DGBasisEvaluator<Order>;
    static constexpr uInt N_basis = Basis::NumBasis;
    constexpr uInt num_points = GaussQuadFace::num_points;
    constexpr auto Qpoints = GaussQuadFace::get_points();
    constexpr auto Qweights = GaussQuadFace::get_weights();

    uInt fid = blockIdx.x * blockDim.x + threadIdx.x;
    if (fid >= num_faces) return;

    const auto& face = faces[fid];
    if (face.neighbor_cells[1] == uInt(-1)) return;

    const auto& coef_L = U[face.neighbor_cells[0]];
    const auto& coef_R = U[face.neighbor_cells[1]];
    DenseMatrix<N_state*N_basis,1> result_L = DenseMatrix<N_state * N_basis, 1>::Zeros();
    DenseMatrix<N_state*N_basis,1> result_R = DenseMatrix<N_state * N_basis, 1>::Zeros();

    for (uInt g = 0; g < num_points; ++g) {
        const vector2f& uv = Qpoints[g];
        Scalar w = Qweights[g] * face.area * 2;

        auto xi_L = dg_utils::transform_to_cell(face, uv, 0);
        auto xi_R = dg_utils::transform_to_cell(face, uv, 1);

        auto basis_L = Basis::eval_all(xi_L[0], xi_L[1], xi_L[2]);
        auto basis_R = Basis::eval_all(xi_R[0], xi_R[1], xi_R[2]);

        auto U_L = dg_utils::reconstruct_state<Order, N_state, N_basis>(coef_L, basis_L);
        auto U_R = dg_utils::reconstruct_state<Order, N_state, N_basis>(coef_R, basis_R);

        auto F_num = numerical_flux->compute(U_L, U_R, face.normal);

        for (uInt j = 0; j < N_basis; ++j) {
            PragmaUnroll
            for (uInt k = 0; k < N_state; ++k) {
                result_L(N_state*j + k, 0) += F_num(k, 0) * basis_L[j] * w;
                result_R(N_state*j + k, 0) -= F_num(k, 0) * basis_R[j] * w;
            }
        }
    }

    // atomicAdd 汇总
    
    for (uInt j = 0; j < N_basis; ++j) {
        PragmaUnroll
        for (uInt k = 0; k < N_state; ++k) {
            atomicAdd(&rhs[face.neighbor_cells[0]](N_state*j + k, 0), result_L(N_state*j + k, 0));
            atomicAdd(&rhs[face.neighbor_cells[1]](N_state*j + k, 0), result_R(N_state*j + k, 0));
        }
    }
}







template<uInt Order, uInt N_state, typename GaussQuadFace>
__global__ void eval_boundarys_kernel(
    const GPUTriangleFace* faces,
    uInt num_faces,
    const GPUTetrahedron* cells,
    uInt num_cells,
    const vector3f* points,
    Scalar time,
    const DenseMatrix<N_state * DGBasisEvaluator<Order>::NumBasis, 1>* U,
    DenseMatrix<N_state * DGBasisEvaluator<Order>::NumBasis, 1>* rhs,
    const NumericalFlux<N_state - 5>* numerical_flux
) {
    using Basis = DGBasisEvaluator<Order>;
    static constexpr uInt N_basis = Basis::NumBasis;
    constexpr uInt num_points = GaussQuadFace::num_points;
    constexpr auto Qpoints = GaussQuadFace::get_points();
    constexpr auto Qweights = GaussQuadFace::get_weights();

    uInt fid = blockIdx.x * blockDim.x + threadIdx.x;
    if (fid >= num_faces) return;

    const auto& face = faces[fid];
    const uInt cell_L = face.neighbor_cells[0];
    const uInt cell_R = face.neighbor_cells[1];

    if (cell_R != uInt(-1)) return; // 只处理边界

    const auto& cell = cells[cell_L];
    const auto& coef_L = U[cell_L];
    DenseMatrix<N_state * N_basis, 1> result_L = DenseMatrix<N_state * N_basis, 1>::Zeros();

    for (uInt g = 0; g < num_points; ++g) {
        const vector2f& uv = Qpoints[g];
        Scalar w = Qweights[g] * face.area * 2;

        const auto& xi_L = dg_utils::transform_to_cell(face, uv, 0);
        const auto& basis_L = Basis::eval_all(xi_L[0], xi_L[1], xi_L[2]);

        const auto& U_L = dg_utils::reconstruct_state<Order, N_state, N_basis>(coef_L, basis_L);

        // 计算 U_R（边界状态）
        vector3f xyz = {
            face.nodes[0] < 3 ? points[face.nodes[0]][0] * (1 - uv[0] - uv[1]) +
                              points[face.nodes[1]][0] * uv[0] +
                              points[face.nodes[2]][0] * uv[1] : 0,
            face.nodes[0] < 3 ? points[face.nodes[0]][1] * (1 - uv[0] - uv[1]) +
                              points[face.nodes[1]][1] * uv[0] +
                              points[face.nodes[2]][1] * uv[1] : 0,
            face.nodes[0] < 3 ? points[face.nodes[0]][2] * (1 - uv[0] - uv[1]) +
                              points[face.nodes[1]][2] * uv[0] +
                              points[face.nodes[2]][2] * uv[1] : 0
        };

        DenseMatrix<N_state, 1> U_R = U_L; // 默认镜像

        if (face.boundaryType == BoundaryType::Dirichlet) {
            U_R = DenseMatrix<N_state, 1>({
                rho_xyz(xyz, time),
                rhou_xyz(xyz, time),
                rhov_xyz(xyz, time),
                rhow_xyz(xyz, time),
                rhoe_xyz(xyz, time)
            });
        }
        else if (face.boundaryType == BoundaryType::Pseudo3DZ) {
            U_R[3] = -U_L[3];
        }
        else if (face.boundaryType == BoundaryType::Pseudo3DY) {
            U_R[2] = -U_L[2];
        }
        else if (face.boundaryType == BoundaryType::Pseudo3DX) {
            U_R[1] = -U_L[1];
        }
        else if (face.boundaryType == BoundaryType::Symmetry) {
            Scalar dot = U_L[1]*face.normal[0] + U_L[2]*face.normal[1] + U_L[3]*face.normal[2];
            U_R[1] = U_L[1] - 2.0 * dot * face.normal[0];
            U_R[2] = U_L[2] - 2.0 * dot * face.normal[1];
            U_R[3] = U_L[3] - 2.0 * dot * face.normal[2];
        }
        else if (face.boundaryType == BoundaryType::Neumann) {
            // 使用梯度外推（需传入梯度）
            // 这里简化为 U_R = U_L
        }

        auto F_num = numerical_flux->compute(U_L, U_R, face.normal);

        for (uInt j = 0; j < N_basis; ++j) {
            PragmaUnroll
            for (uInt k = 0; k < N_state; ++k) {
                result_L(N_state * j + k, 0) += F_num(k, 0) * basis_L[j] * w;
            }
        }
    }

    for (uInt j = 0; j < N_basis; ++j) {
        PragmaUnroll
        for (uInt k = 0; k < N_state; ++k) {
            atomicAdd(&rhs[cell_L](N_state * j + k, 0), result_L(N_state * j + k, 0));
        }
    }
}