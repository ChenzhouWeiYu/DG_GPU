#pragma once

#include "dg/dg_schemes/explicit_convection_gpu/explicit_convection_gpu.cuh"
#include "dg/dg_schemes/explicit_convection_gpu/explicit_convection_gpu_impl.cuh"
// device 函数：Basis、Flux 都是可以直接用的

template<typename Physics, typename FluxScheme, typename Condition, uInt NEQN, uInt NBIS, uInt Order, typename GaussQuadCell, typename GaussQuadFace>
__global__ void eval_cells_kernel(
    const MeshView mesh,
    const DenseMatrix<NEQN*NBIS,1>* U,
    DenseMatrix<NEQN*NBIS,1>* rhs,
    const Physics physic) {
    using Basis = DGBasisEvaluator<Order>;
    // constexpr uInt NBIS = Basis::NumBasis;
    constexpr uInt num_vol_points = GaussQuadCell::num_points;
    constexpr auto Qpoints = GaussQuadCell::get_points();
    constexpr auto Qweights = GaussQuadCell::get_weights();

    uInt cid = blockIdx.x * blockDim.x + threadIdx.x;
    if (cid >= mesh.num_cells) return;
    // printf("cid = %d\n", cid);
    const GPUTetrahedron& cell = mesh.getCell(cid);
    const DenseMatrix<NEQN*NBIS,1>& coef = U[cid];  // NEQN*NBIS 个 DoFs
    DenseMatrix<NEQN*NBIS,1> result = DenseMatrix<NEQN*NBIS,1>::Zeros();  // NEQN*NBIS 个 DoFs
    for (uInt g = 0; g < num_vol_points; ++g) {
        const vector3f& xi = Qpoints[g];
        // 积分点 权重之和为 1/6，这里只需要体积即可，而非 Det[Jac]
        const Scalar& w = Qweights[g] * cell.volume * 6; 

        auto basis = Basis::eval_all(xi[0], xi[1], xi[2]);
        auto grads = Basis::grad_all(xi[0], xi[1], xi[2]);
        DenseMatrix<NEQN,1> U_val = DenseMatrix<NEQN,1>::Zeros();
        PragmaUnroll
        for (uInt bid = 0; bid < NBIS; ++bid) {
            PragmaUnroll
            for (uInt k = 0; k < NEQN; ++k) {
                U_val(k,0) += basis[bid] * coef(NEQN*bid+k,0);
            }
        }
        const DenseMatrix<NEQN,3>& FU = physic.computeFlux(U_val);

        const DenseMatrix<3,3>& Jinv = cell.invJac;
        PragmaUnroll
        for (uInt j = 0; j < NBIS; ++j) {
            DenseMatrix<3,1> grad_phi_j = DenseMatrix<3,1>(grads[j]);
            DenseMatrix<NEQN,1> flux = FU.multiply(Jinv.multiply(grad_phi_j));
            
            // 体积分部分，不会出现多个线程写入到同一个 cid 的情况
            // 但面积分的时候，同一个面的两侧单元，
            // 可能会有多个线程（多个面）同时写入 同一个单元
            PragmaUnroll
            for (uInt k = 0; k < NEQN; ++k) {
                result(NEQN*j+k,0) -= flux(k,0) * w;
            }
        }
    }
    rhs[cid] = result;
}

// Kernel launcher
template<typename Physics, typename FluxScheme, typename Condition, uInt Order, typename GaussQuadCell, typename GaussQuadFace>
void ExplicitConvectionGPU<Physics, FluxScheme, Condition, Order, GaussQuadCell, GaussQuadFace>::eval_cells(
    const DeviceMesh& mesh, const LongVectorDevice<NEQN*NBIS>& U, LongVectorDevice<NEQN*NBIS>& rhs)
{
    dim3 block(256);
    dim3 grid( (mesh.num_cells() + block.x - 1) / block.x );
    eval_cells_kernel<Physics, FluxScheme, Condition, NEQN, NBIS, Order, QuadC, QuadF><<<grid, block>>>(mesh.view(),
                    U.d_blocks, rhs.d_blocks, physics_);
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    // }
    // cudaDeviceSynchronize();
}
