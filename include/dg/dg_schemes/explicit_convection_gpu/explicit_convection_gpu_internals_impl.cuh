#pragma once
#include "dg/dg_schemes/explicit_convection_gpu/explicit_convection_gpu.cuh"
#include "dg/dg_schemes/explicit_convection_gpu/explicit_convection_gpu_impl.cuh"

// device 函数：Basis、Flux 都是可以直接用的

template<typename Physics, typename FluxScheme, typename Condition, uInt NEQN, uInt NBIS, uInt Order, typename GaussQuadCell, typename GaussQuadFace>
__global__ void eval_internals_kernel(
    const MeshView mesh,
    const DenseMatrix<NEQN*NBIS,1>* U,
    DenseMatrix<NEQN*NBIS,1>* rhs,
    const Physics physic,
    const Condition condition){
    using Basis = DGBasisEvaluator<Order>;
    // constexpr uInt NBIS = Basis::NumBasis;
    constexpr uInt num_face_points = GaussQuadFace::num_points;
    constexpr auto Qpoints = GaussQuadFace::get_points();
    constexpr auto Qweights = GaussQuadFace::get_weights();

    uInt fid = blockIdx.x * blockDim.x + threadIdx.x;
    if (fid >= mesh.num_faces) return;

    const GPUTriangleFace& face = mesh.getFace(fid);
    const uInt cell_L = face.neighbor_cells[0];
    const uInt cell_R = face.neighbor_cells[1];

    if (cell_R == uInt(-1)) return; // 跳过边界面

    const DenseMatrix<NEQN*NBIS,1>& coef_L = U[cell_L];
    const DenseMatrix<NEQN*NBIS,1>& coef_R = U[cell_R];
    DenseMatrix<NEQN*NBIS,1> result_L = DenseMatrix<NEQN*NBIS,1>::Zeros();
    DenseMatrix<NEQN*NBIS,1> result_R = DenseMatrix<NEQN*NBIS,1>::Zeros();
    

    for (uInt g = 0; g < num_face_points; ++g) {
        const vector2f& uv = Qpoints[g];
        const Scalar jac_weight = Qweights[g] * face.area * 2;
        
        // printf("%lf,%lf\n",uv[0],uv[1]);
        auto xi_L = transform_to_cell(face, uv, 0);
        // printf("%lf,%lf,%lf\n",xi_L[0],xi_L[1],xi_L[2]);
        auto xi_R = transform_to_cell(face, uv, 1);
        // printf("%lf,%lf,%lf\n",xi_R[0],xi_R[1],xi_R[2]);
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
        const DenseMatrix<NEQN,1>& LF_flux = Flux::computeNumericalFlux(U_L,U_R,face.normal);

        PragmaUnroll
        for (uInt j = 0; j < NBIS; ++j) {
            Scalar phi_jL = basis_L[j];
            Scalar phi_jR = basis_R[j];

            PragmaUnroll
            for (uInt k = 0; k < NEQN; ++k) {
                result_L(NEQN*j+k,0) +=  LF_flux(k,0) * phi_jL * jac_weight;
                result_R(NEQN*j+k,0) -=  LF_flux(k,0) * phi_jR * jac_weight;
            }
        }
    }
    PragmaUnroll
    for (uInt j = 0; j < NBIS; ++j) {
        PragmaUnroll
        for (uInt k = 0; k < NEQN; ++k) {
            atomicAdd(&rhs[cell_L](NEQN*j+k,0),  result_L(NEQN*j+k,0));
            atomicAdd(&rhs[cell_R](NEQN*j+k,0),  result_R(NEQN*j+k,0));
        }
    }
}

// Kernel launcher
template<typename Physics, typename FluxScheme, typename Condition, uInt Order, typename GaussQuadCell, typename GaussQuadFace>
void ExplicitConvectionGPU<Physics, FluxScheme, Condition, Order, GaussQuadCell, GaussQuadFace>::eval_internals(
    const DeviceMesh& mesh, const LongVectorDevice<NEQN*NBIS>& U, LongVectorDevice<NEQN*NBIS>& rhs)
{
    dim3 block(256);
    dim3 grid((mesh.num_faces() + block.x - 1) / block.x);
    eval_internals_kernel<Physics, FluxScheme, Condition, NEQN, NBIS, Order, QuadC, QuadF><<<grid, block>>>(
        mesh.view(), U.d_blocks, rhs.d_blocks, physics_, condition_);
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    // }
    // cudaDeviceSynchronize();
}
