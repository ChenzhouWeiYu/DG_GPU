#pragma once

#include "dg/dg_schemes/explicit_convection_gpu/explicit_convection_gpu.h"
#include "dg/dg_schemes/explicit_convection_gpu/explicit_convection_gpu_impl.h"
// device 函数：Basis、Flux 都是可以直接用的

template<uInt Order, uInt N, typename Flux, typename GaussQuadCell, typename GaussQuadFace>
__global__ void eval_cells_kernel(const GPUTetrahedron* mesh_cells, uInt num_cells,
                                    const DenseMatrix<5*N,1>* U,
                                    DenseMatrix<5*N,1>* rhs){
    using Basis = DGBasisEvaluator<Order>;
    // constexpr uInt N = Basis::NumBasis;
    constexpr uInt num_vol_points = GaussQuadCell::num_points;
    constexpr auto Qpoints = GaussQuadCell::get_points();
    constexpr auto Qweights = GaussQuadCell::get_weights();

    uInt cid = blockIdx.x * blockDim.x + threadIdx.x;
    if (cid >= num_cells) return;
    // printf("cid = %d\n", cid);
    const GPUTetrahedron& cell = mesh_cells[cid];
    const DenseMatrix<5*N,1>& coef = U[cid];  // 5*N 个 DoFs
    for (uInt g = 0; g < num_vol_points; ++g) {
        const vector3f& xi = Qpoints[g];
        // 积分点 权重之和为 1/6，这里只需要体积即可，而非 Det[Jac]
        const Scalar& w = Qweights[g] * cell.volume * 6; 

        auto basis = Basis::eval_all(xi[0], xi[1], xi[2]);
        auto grads = Basis::grad_all(xi[0], xi[1], xi[2]);
        DenseMatrix<5,1> U_val = DenseMatrix<5,1>::Zeros();

        for (uInt bid = 0; bid < N; ++bid) {
            for (uInt k = 0; k < 5; ++k) {
                U_val(k,0) += basis[bid] * coef(5*bid+k,0);
            }
        }
        auto FU = Flux::computeFlux(U_val);

        const auto& Jinv = cell.invJac;

        for (uInt j = 0; j < N; ++j) {
            auto grad_phi_j = DenseMatrix<3,1>(grads[j]);
            auto flux = FU.multiply(Jinv.multiply(grad_phi_j));
            
            // 体积分部分，不会出现多个线程写入到同一个 cid 的情况
            // 但面积分的时候，同一个面的两侧单元，
            // 可能会有多个线程（多个面）同时写入 同一个单元
            rhs[cid](5*j+0,0) -= flux(0,0) * w;
            rhs[cid](5*j+1,0) -= flux(1,0) * w;
            rhs[cid](5*j+2,0) -= flux(2,0) * w;
            rhs[cid](5*j+3,0) -= flux(3,0) * w;
            rhs[cid](5*j+4,0) -= flux(4,0) * w;
        }
    }
}

// Kernel launcher
template<uInt Order, typename Flux, typename GaussQuadCell, typename GaussQuadFace>
void ExplicitConvectionGPU<Order, Flux, GaussQuadCell, GaussQuadFace>::eval_cells(
    const DeviceMesh& mesh, const LongVectorDevice<5*N>& U, LongVectorDevice<5*N>& rhs)
{
    dim3 block(32);
    dim3 grid( (mesh.num_cells() + block.x - 1) / block.x );
    eval_cells_kernel<Order, N, Flux, QuadC, QuadF><<<grid, block>>>(mesh.device_cells(), mesh.num_cells(),
                    U.d_blocks, rhs.d_blocks);
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    // }
    // cudaDeviceSynchronize();
}
