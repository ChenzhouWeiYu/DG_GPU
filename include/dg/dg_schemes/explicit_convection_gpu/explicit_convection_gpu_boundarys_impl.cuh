#pragma once
#include "dg/dg_schemes/explicit_convection_gpu/explicit_convection_gpu.cuh"
#include "dg/dg_schemes/explicit_convection_gpu/explicit_convection_gpu_impl.cuh"



// device 函数：Basis、Flux 都是可以直接用的

template<uInt Order, uInt N, typename Flux, typename GaussQuadCell, typename GaussQuadFace>
__global__ void eval_boundarys_kernel(const GPUTriangleFace* mesh_faces, uInt num_faces,
                                    const GPUTetrahedron* mesh_cells, uInt num_cells,
                                    const vector3f* mesh_points, Scalar time,
                                    const DenseMatrix<5*N,1>* U,
                                    DenseMatrix<5*N,1>* rhs){
    using Basis = DGBasisEvaluator<Order>;
    // constexpr uInt N = Basis::NumBasis;
    constexpr uInt num_face_points = GaussQuadFace::num_points;
    constexpr auto Qpoints = GaussQuadFace::get_points();
    constexpr auto Qweights = GaussQuadFace::get_weights();

    uInt fid = blockIdx.x * blockDim.x + threadIdx.x;
    if (fid >= num_faces) return;

    const GPUTriangleFace& face = mesh_faces[fid];
    const uInt cell_L = face.neighbor_cells[0];
    const uInt cell_R = face.neighbor_cells[1];

    if (cell_R != uInt(-1)) return; // 只处理边界面

    const GPUTetrahedron& cell = mesh_cells[cell_L];
    const vector3f& face_p0 = mesh_points[face.nodes[0]];
    const vector3f& face_p1 = mesh_points[face.nodes[1]];
    const vector3f& face_p2 = mesh_points[face.nodes[2]];
    const DenseMatrix<5*N,1>& coef_L = U[cell_L];
    const DenseMatrix<5,1> U_avg{coef_L[0],coef_L[1],coef_L[2],coef_L[3],coef_L[4]};
    for (uInt g = 0; g < num_face_points; ++g) {
        const vector2f& uv = Qpoints[g];
        const Scalar jac_weight = Qweights[g] * face.area * 2;
        auto xi_L = transform_to_cell(face, uv, 0);
        auto basis_L = Basis::eval_all(xi_L[0], xi_L[1], xi_L[2]);
        auto grads_L = Basis::grad_all(xi_L[0], xi_L[1], xi_L[2]);

        DenseMatrix<5,1> U_L = DenseMatrix<5,1>::Zeros();
        DenseMatrix<5,3> G_L = DenseMatrix<5,3>::Zeros();
        for (uInt bid = 0; bid < N; ++bid){
            const auto& grad = cell.invJac.multiply(grads_L[bid]);
            for (uInt k = 0; k < 5; ++k){
                U_L(k,0) += basis_L[bid] * coef_L(5*bid+k,0);
                G_L(k,0) += grad[0] * coef_L(5*bid+k,0);
                G_L(k,1) += grad[1] * coef_L(5*bid+k,0);
                G_L(k,2) += grad[2] * coef_L(5*bid+k,0);
            }
        }
        
        Scalar x = face_p0[0] * (1 - uv[0] - uv[1]) +
                    face_p1[0] * uv[0] +
                    face_p2[0] * uv[1];
        Scalar y = face_p0[1] * (1 - uv[0] - uv[1]) +
                    face_p1[1] * uv[0] +
                    face_p2[1] * uv[1];
        Scalar z = face_p0[2] * (1 - uv[0] - uv[1]) +
                    face_p1[2] * uv[0] +
                    face_p2[2] * uv[1];
        vector3f xyz = {x, y, z};
        DenseMatrix<5,1> U_R = U_L; // 默认 U_R = U_L
        if (face.boundaryType == BoundaryType::Dirichlet) {
            U_R = DenseMatrix<5,1>({rho_xyz(xyz, time),
                                    rhou_xyz(xyz, time),
                                    rhov_xyz(xyz, time),
                                    rhow_xyz(xyz, time),
                                    rhoe_xyz(xyz, time)});
            // U_R = U_R + 1e-2 * (U_R - U_L);
        }
        else if (face.boundaryType == BoundaryType::Pseudo3DZ) {
            U_R[3] = -U_L[3]; // 只反转 z 速度
            // U_R[3] = 0.0;
        }
        else if (face.boundaryType == BoundaryType::Pseudo3DY) {
            U_R[2] = -U_L[2]; // 只反转 y 速度
            // U_R[2] = 0.0;
        }
        else if (face.boundaryType == BoundaryType::Pseudo3DX) {
            U_R[1] = -U_L[1]; // 只反转 x 速度
            // U_R[1] = 0.0;
        }
        else if (face.boundaryType == BoundaryType::Symmetry) {
            U_R = U_L - (G_L.multiply(DenseMatrix<3,1>(face.normal))) * cell.m_h * 1.00;
            Scalar dot_product = U_R[1]*face.normal[0] + U_R[2]*face.normal[1] + U_R[3]*face.normal[2];
            // U_R[0] = U_R[0];
            U_R[1] = U_R[1] - 2.0 * dot_product * face.normal[0]; // 反转 x 速度
            U_R[2] = U_R[2] - 2.0 * dot_product * face.normal[1]; // 反转 y 速度
            U_R[3] = U_R[3] - 2.0 * dot_product * face.normal[2]; // 反转 z 速度
            // U_R[4] = U_R[4];
        }
        else if (face.boundaryType == BoundaryType::Neumann){
            // U_R = U_avg + 0.0*(U_avg - U_L);
            U_R = U_L - (G_L.multiply(DenseMatrix<3,1>(face.normal))) * cell.m_h * 1.00;
        }
        else {
            continue; // 其他类型暂时跳过
        }

        auto LF_flux = Flux::computeNumericalFlux(U_L,U_R,face.normal);
        
        for (uInt j = 0; j < N; ++j) {
            Scalar phi_jL = basis_L[j];

            atomicAdd(&rhs[cell_L](5*j+0,0), LF_flux(0,0) * phi_jL * jac_weight);
            atomicAdd(&rhs[cell_L](5*j+1,0), LF_flux(1,0) * phi_jL * jac_weight);
            atomicAdd(&rhs[cell_L](5*j+2,0), LF_flux(2,0) * phi_jL * jac_weight);
            atomicAdd(&rhs[cell_L](5*j+3,0), LF_flux(3,0) * phi_jL * jac_weight);
            atomicAdd(&rhs[cell_L](5*j+4,0), LF_flux(4,0) * phi_jL * jac_weight);
        }
    }
}

// Kernel launcher
template<uInt Order, typename Flux, typename GaussQuadCell, typename GaussQuadFace>
void ExplicitConvectionGPU<Order, Flux, GaussQuadCell, GaussQuadFace>::eval_boundarys(
    const DeviceMesh& mesh, const LongVectorDevice<5*N>& U, LongVectorDevice<5*N>& rhs, Scalar time)
{
    dim3 block(256);
    dim3 grid((mesh.num_faces() + block.x - 1) / block.x);
    eval_boundarys_kernel<Order, N, Flux, QuadC, QuadF><<<grid, block>>>(
                    mesh.device_faces(), mesh.num_faces(), 
                    mesh.device_cells(), mesh.num_cells(), 
                    mesh.device_points(), time, U.d_blocks, rhs.d_blocks);
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    // }
    // cudaDeviceSynchronize();
}