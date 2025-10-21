#pragma once

#include "dg/dg_schemes/explicit_convection_gpu/explicit_convection_gpu.cuh"
#include "mesh/device_mesh.cuh"
#include <utility> // for std::integer_sequence

// ========================================================
// 内部面 kernel
// ========================================================
template<uInt Order, uInt N, typename Flux, typename GaussQuadCell, typename GaussQuadFace>
__global__ void eval_internal_faces_kernel(
    const MeshView mesh,
    const DenseMatrix<5*N,1>* U,
    DenseMatrix<5*N,1>* rhs) {
    
    using Basis = DGBasisEvaluator<Order>;
    constexpr uInt num_face_points = GaussQuadFace::num_points;
    constexpr auto Qpoints = GaussQuadFace::get_points();
    constexpr auto Qweights = GaussQuadFace::get_weights();

    uInt local_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_idx >= mesh.numFacesOfType(0)) return; // FaceType::Internal = 0

    uInt face_global_idx = mesh.getFaceGlobalIndex(0, local_idx);
    const GPUTriangleFace& face = mesh.getFace(face_global_idx);
    const uInt cell_L = face.neighbor_cells[0];
    const uInt cell_R = face.neighbor_cells[1];

    // 内部面保证 cell_R != -1
    const DenseMatrix<5*N,1>& coef_L = U[cell_L];
    const DenseMatrix<5*N,1>& coef_R = U[cell_R];
    DenseMatrix<5*N,1> result_L = DenseMatrix<5*N,1>::Zeros();
    DenseMatrix<5*N,1> result_R = DenseMatrix<5*N,1>::Zeros();

    for (uInt g = 0; g < num_face_points; ++g) {
        const vector2f& uv = Qpoints[g];
        const Scalar jac_weight = Qweights[g] * face.area * 2;
        
        auto xi_L = transform_to_cell(face, uv, 0);
        auto xi_R = transform_to_cell(face, uv, 1);
        const auto& basis_L = Basis::eval_all(xi_L[0], xi_L[1], xi_L[2]);
        const auto& basis_R = Basis::eval_all(xi_R[0], xi_R[1], xi_R[2]);

        DenseMatrix<5,1> U_L, U_R;
        PragmaUnroll
        for (uInt bid = 0; bid < N; ++bid) {
            PragmaUnroll
            for (uInt k = 0; k < 5; ++k) {
                U_L(k,0) += basis_L[bid] * coef_L(5*bid+k,0);
                U_R(k,0) += basis_R[bid] * coef_R(5*bid+k,0);
            }
        }
        const DenseMatrix<5,1> LF_flux = Flux::computeNumericalFlux(U_L, U_R, face.normal);

        PragmaUnroll
        for (uInt j = 0; j < N; ++j) {
            Scalar phi_jL = basis_L[j];
            Scalar phi_jR = basis_R[j];

            result_L(5*j+0,0) += LF_flux(0,0) * phi_jL * jac_weight;
            result_L(5*j+1,0) += LF_flux(1,0) * phi_jL * jac_weight;
            result_L(5*j+2,0) += LF_flux(2,0) * phi_jL * jac_weight;
            result_L(5*j+3,0) += LF_flux(3,0) * phi_jL * jac_weight;
            result_L(5*j+4,0) += LF_flux(4,0) * phi_jL * jac_weight;

            result_R(5*j+0,0) -= LF_flux(0,0) * phi_jR * jac_weight;
            result_R(5*j+1,0) -= LF_flux(1,0) * phi_jR * jac_weight;
            result_R(5*j+2,0) -= LF_flux(2,0) * phi_jR * jac_weight;
            result_R(5*j+3,0) -= LF_flux(3,0) * phi_jR * jac_weight;
            result_R(5*j+4,0) -= LF_flux(4,0) * phi_jR * jac_weight;
        }
    }

    PragmaUnroll
    for (uInt j = 0; j < N; ++j) {
        atomicAdd(&rhs[cell_L](5*j+0,0), result_L(5*j+0,0));
        atomicAdd(&rhs[cell_L](5*j+1,0), result_L(5*j+1,0));
        atomicAdd(&rhs[cell_L](5*j+2,0), result_L(5*j+2,0));
        atomicAdd(&rhs[cell_L](5*j+3,0), result_L(5*j+3,0));
        atomicAdd(&rhs[cell_L](5*j+4,0), result_L(5*j+4,0));

        atomicAdd(&rhs[cell_R](5*j+0,0), result_R(5*j+0,0));
        atomicAdd(&rhs[cell_R](5*j+1,0), result_R(5*j+1,0));
        atomicAdd(&rhs[cell_R](5*j+2,0), result_R(5*j+2,0));
        atomicAdd(&rhs[cell_R](5*j+3,0), result_R(5*j+3,0));
        atomicAdd(&rhs[cell_R](5*j+4,0), result_R(5*j+4,0));
    }
}

// ========================================================
// 边界面 kernel（模板化 FaceType）
// ========================================================
template<FaceType FT>
HostDevice DenseMatrix<5,1> computeUR(
    const GPUTriangleFace& face,
    const DenseMatrix<5,1>& U_L,
    vector3f xyz, Scalar time) {

    DenseMatrix<5,1> U_R = U_L; // 默认
    if constexpr (FT == FaceType::Dirichlet) {
        U_R = DenseMatrix<5,1>({rho_xyz(xyz, time),
                                rhou_xyz(xyz, time),
                                rhov_xyz(xyz, time),
                                rhow_xyz(xyz, time),
                                rhoe_xyz(xyz, time)});
    }
    else if constexpr (FT == FaceType::Pseudo3DZ) {
        U_R[3] = -U_L[3];
    }
    else if constexpr (FT == FaceType::Pseudo3DY) {
        U_R[2] = -U_L[2];
    }
    else if constexpr (FT == FaceType::Pseudo3DX) {
        U_R[1] = -U_L[1];
    }
    else if constexpr (FT == FaceType::Symmetry) {
        Scalar dot_product = U_L[1]*face.normal[0] + U_L[2]*face.normal[1] + U_L[3]*face.normal[2];
        U_R[1] -= 2.0 * dot_product * face.normal[0];
        U_R[2] -= 2.0 * dot_product * face.normal[1];
        U_R[3] -= 2.0 * dot_product * face.normal[2];
    }
    // 其他类型保持 U_R = U_L
    return U_R;
}

template<uInt Order, uInt N, typename Flux, typename GaussQuadCell, typename GaussQuadFace, FaceType FT>
__global__ void eval_boundary_faces_kernel(
    const MeshView mesh,
    Scalar time,
    const DenseMatrix<5*N,1>* U,
    DenseMatrix<5*N,1>* rhs) {
    
    using Basis = DGBasisEvaluator<Order>;
    constexpr uInt num_face_points = GaussQuadFace::num_points;
    constexpr auto Qpoints = GaussQuadFace::get_points();
    constexpr auto Qweights = GaussQuadFace::get_weights();

    uInt local_idx = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr uInt ft_index = static_cast<uInt>(FT);
    if (local_idx >= mesh.numFacesOfType(ft_index)) return;

    uInt face_global_idx = mesh.getFaceGlobalIndex(ft_index, local_idx);
    const GPUTriangleFace& face = mesh.getFace(face_global_idx);
    const vector3f& p0 = mesh.getPoint(face.nodes[0]);
    const vector3f& p1 = mesh.getPoint(face.nodes[1]);
    const vector3f& p2 = mesh.getPoint(face.nodes[2]);
    const uInt cell_L = face.neighbor_cells[0]; // 边界面只有 cell_L

    const GPUTetrahedron& cell = mesh.getCell(cell_L);
    const DenseMatrix<5*N,1>& coef_L = U[cell_L];
    DenseMatrix<5*N,1> result_L = DenseMatrix<5*N,1>::Zeros();

    for (uInt g = 0; g < num_face_points; ++g) {
        const vector2f& uv = Qpoints[g];
        const Scalar jac_weight = Qweights[g] * face.area * 2;
        auto xi_L = transform_to_cell(face, uv, 0);
        const auto& basis_L = Basis::eval_all(xi_L[0], xi_L[1], xi_L[2]);

        DenseMatrix<5,1> U_L = DenseMatrix<5,1>::Zeros();
        PragmaUnroll
        for (uInt bid = 0; bid < N; ++bid) {
            PragmaUnroll
            for (uInt k = 0; k < 5; ++k) {
                U_L(k,0) += basis_L[bid] * coef_L(5*bid+k,0);
            }
        }

        // 计算物理坐标
        Scalar x = p0[0]*(1-uv[0]-uv[1]) + p1[0]*uv[0] + p2[0]*uv[1];
        Scalar y = p0[1]*(1-uv[0]-uv[1]) + p1[1]*uv[0] + p2[1]*uv[1];
        Scalar z = p0[2]*(1-uv[0]-uv[1]) + p1[2]*uv[0] + p2[2]*uv[1];
        vector3f xyz{x, y, z};

        DenseMatrix<5,1> U_R = computeUR<FT>(face, U_L, xyz, time);
        auto LF_flux = Flux::computeNumericalFlux(U_L, U_R, face.normal);

        PragmaUnroll
        for (uInt j = 0; j < N; ++j) {
            Scalar phi_jL = basis_L[j];
            result_L(5*j+0,0) += LF_flux(0,0) * phi_jL * jac_weight;
            result_L(5*j+1,0) += LF_flux(1,0) * phi_jL * jac_weight;
            result_L(5*j+2,0) += LF_flux(2,0) * phi_jL * jac_weight;
            result_L(5*j+3,0) += LF_flux(3,0) * phi_jL * jac_weight;
            result_L(5*j+4,0) += LF_flux(4,0) * phi_jL * jac_weight;
        }
    }

    PragmaUnroll
    for (uInt j = 0; j < N; ++j) {
        atomicAdd(&rhs[cell_L](5*j+0,0), result_L(5*j+0,0));
        atomicAdd(&rhs[cell_L](5*j+1,0), result_L(5*j+1,0));
        atomicAdd(&rhs[cell_L](5*j+2,0), result_L(5*j+2,0));
        atomicAdd(&rhs[cell_L](5*j+3,0), result_L(5*j+3,0));
        atomicAdd(&rhs[cell_L](5*j+4,0), result_L(5*j+4,0));
    }
}

// ========================================================
// Helper: 启动内部面 kernel
// ========================================================
template<uInt Order, uInt N, typename Flux, typename GaussQuadCell, typename GaussQuadFace>
void launch_internal_faces_kernel(
    const DeviceMesh& mesh,
    const LongVectorDevice<5*N>& U,
    LongVectorDevice<5*N>& rhs) {
    
    uInt num_internal = mesh.numFacesOfType(0);
    if (num_internal == 0) return;

    dim3 block(256);
    dim3 grid((num_internal + block.x - 1) / block.x);
    eval_internal_faces_kernel<Order, N, Flux, GaussQuadCell, GaussQuadFace><<<grid, block>>>(
        mesh.view(), U.d_blocks, rhs.d_blocks);
}

// ========================================================
// Helper: 启动单个边界类型 kernel
// ========================================================
template<uInt Order, uInt N, typename Flux, typename GaussQuadCell, typename GaussQuadFace, FaceType FT>
void launch_boundary_faces_kernel(
    const DeviceMesh& mesh,
    const LongVectorDevice<5*N>& U,
    LongVectorDevice<5*N>& rhs,
    Scalar time) {
    
    constexpr uInt ft_index = static_cast<uInt>(FT);
    uInt num_faces = mesh.numFacesOfType(ft_index);
    if (num_faces == 0) return;

    dim3 block(256);
    dim3 grid((num_faces + block.x - 1) / block.x);
    eval_boundary_faces_kernel<Order, N, Flux, GaussQuadCell, GaussQuadFace, FT><<<grid, block>>>(
        mesh.view(), time, U.d_blocks, rhs.d_blocks);
}

// ========================================================
// 使用 integer_sequence 展开边界类型循环
// ========================================================
template<uInt Order, uInt N, typename Flux, typename GaussQuadCell, typename GaussQuadFace, uInt... Indices>
void launch_all_boundary_kernels_impl(
    const DeviceMesh& mesh,
    const LongVectorDevice<5*N>& U,
    LongVectorDevice<5*N>& rhs,
    Scalar time,
    std::integer_sequence<uInt, Indices...>) {
    
    // 展开为 launch_boundary_faces_kernel<FaceType(Indices+1)>...
    // 注意：Indices 从 0 开始，但 FaceType::Internal=0 已处理，所以边界类型从 1 开始
    ((launch_boundary_faces_kernel<Order, N, Flux, GaussQuadCell, GaussQuadFace, 
        static_cast<FaceType>(Indices + 1)>(mesh, U, rhs, time)), ...);
}

template<uInt Order, uInt N, typename Flux, typename GaussQuadCell, typename GaussQuadFace>
void launch_all_boundary_kernels(
    const DeviceMesh& mesh,
    const LongVectorDevice<5*N>& U,
    LongVectorDevice<5*N>& rhs,
    Scalar time) {
    
    // 生成序列 0, 1, ..., NumFaceTypes-2 （因为 Internal=0，边界类型共 NumFaceTypes-1 个）
    launch_all_boundary_kernels_impl<Order, N, Flux, GaussQuadCell, GaussQuadFace>(
        mesh, U, rhs, time,
        std::make_integer_sequence<uInt, NumFaceTypes - 1>{});
}

// ========================================================
// 统一的 eval_faces 接口
// ========================================================
template<uInt Order, typename Flux, typename GaussQuadCell, typename GaussQuadFace>
void ExplicitConvectionGPU<Order, Flux, GaussQuadCell, GaussQuadFace>::eval_faces(
    const DeviceMesh& mesh,
    const LongVectorDevice<5*N>& U,
    LongVectorDevice<5*N>& rhs,
    Scalar time) {
    
    // 1. 内部面
    launch_internal_faces_kernel<Order, N, Flux, QuadC, QuadF>(mesh, U, rhs);
    
    // 2. 所有边界面
    launch_all_boundary_kernels<Order, N, Flux, QuadC, QuadF>(mesh, U, rhs, time);
}