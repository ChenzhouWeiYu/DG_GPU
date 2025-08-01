#pragma once
#include "dg/dg_schemes/explicit_convection_gpu/explicit_convection_gpu.cuh"
#include "dg/dg_schemes/explicit_convection_gpu/explicit_convection_gpu_impl.cuh"

// device 函数：Basis、Flux 都是可以直接用的

template<uInt Order, uInt N, typename Flux, typename GaussQuadCell, typename GaussQuadFace>
__global__ void eval_internals_kernel(const GPUTriangleFace* mesh_faces, uInt num_faces,
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

    if (cell_R == uInt(-1)) return; // 跳过边界面

    const DenseMatrix<5*N,1> coef_L = U[cell_L];
    const DenseMatrix<5*N,1> coef_R = U[cell_R];
    

    for (uInt g = 0; g < num_face_points; ++g) {
        const vector2f& uv = Qpoints[g];
        const Scalar jac_weight = Qweights[g] * face.area * 2;
        
        // printf("%lf,%lf\n",uv[0],uv[1]);
        auto xi_L = transform_to_cell(face, uv, 0);
        // printf("%lf,%lf,%lf\n",xi_L[0],xi_L[1],xi_L[2]);
        auto xi_R = transform_to_cell(face, uv, 1);
        // printf("%lf,%lf,%lf\n",xi_R[0],xi_R[1],xi_R[2]);
        auto basis_L = Basis::eval_all(xi_L[0], xi_L[1], xi_L[2]);
        auto basis_R = Basis::eval_all(xi_R[0], xi_R[1], xi_R[2]);

        DenseMatrix<5,1> U_L, U_R;
        for (uInt bid = 0; bid < N; ++bid) {
            for (uInt k = 0; k < 5; ++k) {
                U_L(k,0) += basis_L[bid] * coef_L(5*bid+k,0);
                U_R(k,0) += basis_R[bid] * coef_R(5*bid+k,0);
            }
        }

        // auto FU_L = Flux::computeFlux(U_L);
        // auto FU_R = Flux::computeFlux(U_R);
        // auto FUn_L = FU_L.multiply(DenseMatrix<3,1>(face.normal));
        // auto FUn_R = FU_R.multiply(DenseMatrix<3,1>(face.normal));

        // Scalar lambda = Flux::computeWaveSpeed(U_L, U_R);
        // auto LF_flux = 0.5 * (FUn_L + FUn_R + lambda * (U_L - U_R));


        // auto LF_flux = Flux::computeLaxFriedrichsFlux(U_L,U_R,face.normal);
        // auto LF_flux = Flux::computeHLLFlux(U_L,U_R,face.normal);
        // auto LF_flux = Flux::computeHLLCFlux(U_L,U_R,face.normal);
        auto LF_flux = Flux::computeNumericalFlux(U_L,U_R,face.normal);


        for (uInt j = 0; j < N; ++j) {
            Scalar phi_jL = basis_L[j];
            Scalar phi_jR = basis_R[j];

            atomicAdd(&rhs[cell_L](5*j+0,0),  LF_flux(0,0) * phi_jL * jac_weight);
            atomicAdd(&rhs[cell_L](5*j+1,0),  LF_flux(1,0) * phi_jL * jac_weight);
            atomicAdd(&rhs[cell_L](5*j+2,0),  LF_flux(2,0) * phi_jL * jac_weight);
            atomicAdd(&rhs[cell_L](5*j+3,0),  LF_flux(3,0) * phi_jL * jac_weight);
            atomicAdd(&rhs[cell_L](5*j+4,0),  LF_flux(4,0) * phi_jL * jac_weight);

            atomicAdd(&rhs[cell_R](5*j+0,0), -LF_flux(0,0) * phi_jR * jac_weight);
            atomicAdd(&rhs[cell_R](5*j+1,0), -LF_flux(1,0) * phi_jR * jac_weight);
            atomicAdd(&rhs[cell_R](5*j+2,0), -LF_flux(2,0) * phi_jR * jac_weight);
            atomicAdd(&rhs[cell_R](5*j+3,0), -LF_flux(3,0) * phi_jR * jac_weight);
            atomicAdd(&rhs[cell_R](5*j+4,0), -LF_flux(4,0) * phi_jR * jac_weight);
        }
    }
}

// Kernel launcher
template<uInt Order, typename Flux, typename GaussQuadCell, typename GaussQuadFace>
void ExplicitConvectionGPU<Order, Flux, GaussQuadCell, GaussQuadFace>::eval_internals(
    const DeviceMesh& mesh, const LongVectorDevice<5*N>& U, LongVectorDevice<5*N>& rhs)
{
    dim3 block(256);
    dim3 grid((mesh.num_faces() + block.x - 1) / block.x);
    eval_internals_kernel<Order, N, Flux, QuadC, QuadF><<<grid, block>>>(mesh.device_faces(), mesh.num_faces(), U.d_blocks, rhs.d_blocks);
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    // }
    // cudaDeviceSynchronize();
}
