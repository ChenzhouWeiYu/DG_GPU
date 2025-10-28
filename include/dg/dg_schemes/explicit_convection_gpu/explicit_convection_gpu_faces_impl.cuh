#pragma once

#include "dg/dg_schemes/explicit_convection_gpu/explicit_convection_gpu.cuh"
#include "mesh/device_mesh.cuh"
#include <utility> // for std::integer_sequence

// ========================================================
// 内部面 kernel
// ========================================================
template<typename Physics, typename FluxScheme, typename Condition, uInt NEQN, uInt NBIS, uInt Order, typename GaussQuadCell, typename GaussQuadFace>
__global__ void eval_internal_faces_kernel(
    const MeshView mesh,
    const DenseMatrix<NEQN*NBIS,1>* U,
    DenseMatrix<NEQN*NBIS,1>* rhs,
    const Physics physic,
    const Condition condition) {
    
    using Basis = DGBasisEvaluator<Order>;
    constexpr uInt num_face_points = GaussQuadFace::num_points;
    constexpr auto Qpoints = GaussQuadFace::get_points();
    constexpr auto Qweights = GaussQuadFace::get_weights();

    uInt local_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_idx >= mesh.num_faces_of_type(0)) return; // FaceType::Internal = 0

    uInt face_global_idx = mesh.get_face_global_index(0, local_idx);
    const GPUTriangleFace& face = mesh.get_face(face_global_idx);
    const uInt cell_L = face.neighbor_cells[0];
    const uInt cell_R = face.neighbor_cells[1];

    // 内部面保证 cell_R != -1
    const DenseMatrix<NEQN*NBIS,1>& coef_L = U[cell_L];
    const DenseMatrix<NEQN*NBIS,1>& coef_R = U[cell_R];
    DenseMatrix<NEQN*NBIS,1> result_L = DenseMatrix<NEQN*NBIS,1>::Zeros();
    DenseMatrix<NEQN*NBIS,1> result_R = DenseMatrix<NEQN*NBIS,1>::Zeros();

    for (uInt g = 0; g < num_face_points; ++g) {
        const vector2f& uv = Qpoints[g];
        const Scalar jac_weight = Qweights[g] * face.area * 2;
        
        auto xi_L = transform_to_cell(face, uv, 0);
        auto xi_R = transform_to_cell(face, uv, 1);
        const auto& basis_L = Basis::eval_all(xi_L[0], xi_L[1], xi_L[2]);
        const auto& basis_R = Basis::eval_all(xi_R[0], xi_R[1], xi_R[2]);

        DenseMatrix<NEQN,1> U_L, U_R;
        PragmaUnroll
        for (uInt bid = 0; bid < NBIS; ++bid) {
            PragmaUnroll
            for (uInt k = 0; k < NEQN; ++k) {
                U_L(k,0) += basis_L[bid] * coef_L(NEQN*bid+k,0);
                U_R(k,0) += basis_R[bid] * coef_R(NEQN*bid+k,0);
            }
        }
        const DenseMatrix<NEQN,1> LF_flux = FluxScheme::compute(physic, U_L, U_R, face.normal);

        PragmaUnroll
        for (uInt j = 0; j < NBIS; ++j) {
            Scalar phi_jL = basis_L[j];
            Scalar phi_jR = basis_R[j];

            result_L(NEQN*j+0,0) += LF_flux(0,0) * phi_jL * jac_weight;
            result_L(NEQN*j+1,0) += LF_flux(1,0) * phi_jL * jac_weight;
            result_L(NEQN*j+2,0) += LF_flux(2,0) * phi_jL * jac_weight;
            result_L(NEQN*j+3,0) += LF_flux(3,0) * phi_jL * jac_weight;
            result_L(NEQN*j+4,0) += LF_flux(4,0) * phi_jL * jac_weight;

            result_R(NEQN*j+0,0) -= LF_flux(0,0) * phi_jR * jac_weight;
            result_R(NEQN*j+1,0) -= LF_flux(1,0) * phi_jR * jac_weight;
            result_R(NEQN*j+2,0) -= LF_flux(2,0) * phi_jR * jac_weight;
            result_R(NEQN*j+3,0) -= LF_flux(3,0) * phi_jR * jac_weight;
            result_R(NEQN*j+4,0) -= LF_flux(4,0) * phi_jR * jac_weight;
        }
    }

    PragmaUnroll
    for (uInt j = 0; j < NBIS; ++j) {
        atomicAdd(&rhs[cell_L](NEQN*j+0,0), result_L(NEQN*j+0,0));
        atomicAdd(&rhs[cell_L](NEQN*j+1,0), result_L(NEQN*j+1,0));
        atomicAdd(&rhs[cell_L](NEQN*j+2,0), result_L(NEQN*j+2,0));
        atomicAdd(&rhs[cell_L](NEQN*j+3,0), result_L(NEQN*j+3,0));
        atomicAdd(&rhs[cell_L](NEQN*j+4,0), result_L(NEQN*j+4,0));

        atomicAdd(&rhs[cell_R](NEQN*j+0,0), result_R(NEQN*j+0,0));
        atomicAdd(&rhs[cell_R](NEQN*j+1,0), result_R(NEQN*j+1,0));
        atomicAdd(&rhs[cell_R](NEQN*j+2,0), result_R(NEQN*j+2,0));
        atomicAdd(&rhs[cell_R](NEQN*j+3,0), result_R(NEQN*j+3,0));
        atomicAdd(&rhs[cell_R](NEQN*j+4,0), result_R(NEQN*j+4,0));
    }
}

// ========================================================
// 边界面 kernel（模板化 FaceType）
// ========================================================
template<typename Condition, uInt NEQN, FaceType FT>
HostDevice DenseMatrix<NEQN,1> computeUR(
    const Condition condition, 
    const GPUTriangleFace& face,
    const DenseMatrix<NEQN,1>& U_L,
    const DenseMatrix<NEQN,1>& U_c,
    vector3f xyz, Scalar time) {

    DenseMatrix<NEQN,1> U_R = U_L; // 默认
    if constexpr (FT == FaceType::Dirichlet) {
        // U_R = DenseMatrix<NEQN,1>({rho_xyz(xyz, time),
        //                         rhou_xyz(xyz, time),
        //                         rhov_xyz(xyz, time),
        //                         rhow_xyz(xyz, time),
        //                         rhoe_xyz(xyz, time)});
        U_R = condition.compute(xyz, time);
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

template<typename Physics, typename FluxScheme, typename Condition, uInt NEQN, uInt NBIS, uInt Order, typename GaussQuadCell, typename GaussQuadFace, FaceType FT>
__global__ void eval_boundary_faces_kernel(
    const MeshView mesh,
    Scalar time,
    const DenseMatrix<NEQN*NBIS,1>* U,
    DenseMatrix<NEQN*NBIS,1>* rhs,
    const Physics physic,
    const Condition condition) {
    
    using Basis = DGBasisEvaluator<Order>;
    constexpr uInt num_face_points = GaussQuadFace::num_points;
    constexpr auto Qpoints = GaussQuadFace::get_points();
    constexpr auto Qweights = GaussQuadFace::get_weights();

    uInt local_idx = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr uInt ft_index = static_cast<uInt>(FT);
    if (local_idx >= mesh.num_faces_of_type(ft_index)) return;

    uInt face_global_idx = mesh.get_face_global_index(ft_index, local_idx);
    const GPUTriangleFace& face = mesh.get_face(face_global_idx);
    const vector3f& p0 = mesh.get_point(face.nodes[0]);
    const vector3f& p1 = mesh.get_point(face.nodes[1]);
    const vector3f& p2 = mesh.get_point(face.nodes[2]);
    const uInt cell_L = face.neighbor_cells[0]; // 边界面只有 cell_L

    const GPUTetrahedron& cell = mesh.get_cell(cell_L);
    const DenseMatrix<NEQN*NBIS,1>& coef_L = U[cell_L];
    DenseMatrix<NEQN*NBIS,1> result_L = DenseMatrix<NEQN*NBIS,1>::Zeros();
    const auto& basis_c = Basis::eval_all(0.25, 0.25, 0.25);
    DenseMatrix<NEQN,1> U_c = DenseMatrix<NEQN,1>::Zeros();
    auto xi_f = transform_to_cell(face, vector2f{1.0/3.0, 1.0/3.0}, 0);
    const auto& basis_f = Basis::eval_all(xi_f[0], xi_f[1], xi_f[2]);
    DenseMatrix<NEQN,1> U_f = DenseMatrix<NEQN,1>::Zeros();
    PragmaUnroll
    for (uInt bid = 0; bid < NBIS; ++bid) {
        PragmaUnroll
        for (uInt k = 0; k < NEQN; ++k) {
            U_c(k,0) += basis_c[bid] * coef_L(NEQN*bid+k,0);
            U_f(k,0) += basis_f[bid] * coef_L(NEQN*bid+k,0);
        }
    }

    for (uInt g = 0; g < num_face_points; ++g) {
        const vector2f& uv = Qpoints[g];
        const Scalar jac_weight = Qweights[g] * face.area * 2;
        auto xi_L = transform_to_cell(face, uv, 0);
        const auto& basis_L = Basis::eval_all(xi_L[0], xi_L[1], xi_L[2]);

        DenseMatrix<NEQN,1> U_L = DenseMatrix<NEQN,1>::Zeros();
        PragmaUnroll
        for (uInt bid = 0; bid < NBIS; ++bid) {
            PragmaUnroll
            for (uInt k = 0; k < NEQN; ++k) {
                U_L(k,0) += basis_L[bid] * coef_L(NEQN*bid+k,0);
            }
        }

        // 计算物理坐标
        Scalar x = p0[0]*(1-uv[0]-uv[1]) + p1[0]*uv[0] + p2[0]*uv[1];
        Scalar y = p0[1]*(1-uv[0]-uv[1]) + p1[1]*uv[0] + p2[1]*uv[1];
        Scalar z = p0[2]*(1-uv[0]-uv[1]) + p1[2]*uv[0] + p2[2]*uv[1];
        vector3f xyz{x, y, z};

        DenseMatrix<NEQN,1> U_R = computeUR<Condition,NEQN,FT>(condition,face, U_L - (U_f - U_c), U_c, xyz, time);
        auto LF_flux = FluxScheme::compute(physic, U_L, U_R, face.normal);
        // auto LF_flux = physic.compute_flux(U_R).multiply(face.normal);

        PragmaUnroll
        for (uInt j = 0; j < NBIS; ++j) {
            Scalar phi_jL = basis_L[j];
            result_L(NEQN*j+0,0) += LF_flux(0,0) * phi_jL * jac_weight;
            result_L(NEQN*j+1,0) += LF_flux(1,0) * phi_jL * jac_weight;
            result_L(NEQN*j+2,0) += LF_flux(2,0) * phi_jL * jac_weight;
            result_L(NEQN*j+3,0) += LF_flux(3,0) * phi_jL * jac_weight;
            result_L(NEQN*j+4,0) += LF_flux(4,0) * phi_jL * jac_weight;
        }
    }

    PragmaUnroll
    for (uInt j = 0; j < NBIS; ++j) {
        atomicAdd(&rhs[cell_L](NEQN*j+0,0), result_L(NEQN*j+0,0));
        atomicAdd(&rhs[cell_L](NEQN*j+1,0), result_L(NEQN*j+1,0));
        atomicAdd(&rhs[cell_L](NEQN*j+2,0), result_L(NEQN*j+2,0));
        atomicAdd(&rhs[cell_L](NEQN*j+3,0), result_L(NEQN*j+3,0));
        atomicAdd(&rhs[cell_L](NEQN*j+4,0), result_L(NEQN*j+4,0));
    }
}

// ========================================================
// Helper: 启动内部面 kernel
// ========================================================
template<typename Physics, typename FluxScheme, typename Condition, uInt NEQN, uInt NBIS, uInt Order, typename GaussQuadCell, typename GaussQuadFace>
void launch_internal_faces_kernel(
    const DeviceMesh& mesh,
    const LongVectorDevice<NEQN*NBIS>& U,
    LongVectorDevice<NEQN*NBIS>& rhs,
    const Physics physic,
    const Condition condition) {
    
    uInt num_internal = mesh.num_faces_of_type(0);
    if (num_internal == 0) return;

    dim3 block(256);
    dim3 grid((num_internal + block.x - 1) / block.x);
    eval_internal_faces_kernel<Physics, FluxScheme, Condition, NEQN, NBIS, Order, GaussQuadCell, GaussQuadFace><<<grid, block>>>(
        mesh.view(), U.d_blocks, rhs.d_blocks, physic,condition);
}

// ========================================================
// Helper: 启动单个边界类型 kernel
// ========================================================
template<typename Physics, typename FluxScheme, typename Condition, uInt NEQN, uInt NBIS, uInt Order, typename GaussQuadCell, typename GaussQuadFace, FaceType FT>
void launch_boundary_faces_kernel(
    const DeviceMesh& mesh,
    const LongVectorDevice<NEQN*NBIS>& U,
    LongVectorDevice<NEQN*NBIS>& rhs,
    const Physics physic,
    const Condition condition,
    Scalar time) {
    
    constexpr uInt ft_index = static_cast<uInt>(FT);
    uInt num_faces = mesh.num_faces_of_type(ft_index);
    if (num_faces == 0) return;

    dim3 block(256);
    dim3 grid((num_faces + block.x - 1) / block.x);
    eval_boundary_faces_kernel<Physics, FluxScheme, Condition, NEQN, NBIS, Order, GaussQuadCell, GaussQuadFace, FT><<<grid, block>>>(
        mesh.view(), time, U.d_blocks, rhs.d_blocks, physic, condition);
}

// ========================================================
// 使用 integer_sequence 展开边界类型循环
// ========================================================
template<typename Physics, typename FluxScheme, typename Condition, uInt NEQN, uInt NBIS, uInt Order, typename GaussQuadCell, typename GaussQuadFace, uInt... Indices>
void launch_all_boundary_kernels_impl(
    const DeviceMesh& mesh,
    const LongVectorDevice<NEQN*NBIS>& U,
    LongVectorDevice<NEQN*NBIS>& rhs,
    const Physics physic,
    const Condition condition,
    Scalar time,
    std::integer_sequence<uInt, Indices...>) {
    
    // 展开为 launch_boundary_faces_kernel<FaceType(Indices+1)>...
    // 注意：Indices 从 0 开始，但 FaceType::Internal=0 已处理，所以边界类型从 1 开始
    ((launch_boundary_faces_kernel<Physics, FluxScheme, Condition, NEQN, NBIS, Order, GaussQuadCell, GaussQuadFace, 
        static_cast<FaceType>(Indices + 1)>(mesh, U, rhs, physic, condition, time)), ...);
}

template<typename Physics, typename FluxScheme, typename Condition, uInt NEQN, uInt NBIS, uInt Order, typename GaussQuadCell, typename GaussQuadFace>
void launch_all_boundary_kernels(
    const DeviceMesh& mesh,
    const LongVectorDevice<NEQN*NBIS>& U,
    LongVectorDevice<NEQN*NBIS>& rhs,
    const Physics physic,
    const Condition condition,
    Scalar time) {
    
    // 生成序列 0, 1, ..., NumFaceTypes-2 （因为 Internal=0，边界类型共 NumFaceTypes-1 个）
    launch_all_boundary_kernels_impl<Physics, FluxScheme, Condition, NEQN, NBIS, Order, GaussQuadCell, GaussQuadFace>(
        mesh, U, rhs, physic, condition, time,
        std::make_integer_sequence<uInt, NumFaceTypes - 1>{});
}

// ========================================================
// 统一的 eval_faces 接口
// ========================================================
template<typename Physics, typename FluxScheme, typename Condition, uInt Order, typename GaussQuadCell, typename GaussQuadFace>
void ExplicitConvectionGPU<Physics, FluxScheme, Condition, Order, GaussQuadCell, GaussQuadFace>::eval_faces(
    const DeviceMesh& mesh,
    const LongVectorDevice<NEQN*NBIS>& U,
    LongVectorDevice<NEQN*NBIS>& rhs,
    Scalar time) {
    
    // 1. 内部面
    launch_internal_faces_kernel<Physics, FluxScheme, Condition, NEQN, NBIS, Order, QuadC, QuadF>(mesh, U, rhs, physics_, condition_);
    
    // 2. 所有边界面
    launch_all_boundary_kernels<Physics, FluxScheme, Condition, NEQN, NBIS, Order, QuadC, QuadF>(mesh, U, rhs, physics_, condition_, time);
}