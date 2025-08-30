#pragma once
#include "dg/dg_schemes/explicit_convection_gpu/explicit_convection_gpu.cuh"
#include "dg/dg_schemes/explicit_convection_gpu/explicit_convection_gpu_impl.cuh"
// device 函数：Basis、Flux 都是可以直接用的


template<uInt Order, uInt N, typename Flux, typename GaussQuadCell, typename GaussQuadFace>
__global__ void eval_cells_kernel(
    const GPUTetrahedron* __restrict__ mesh_cells,
    const DenseMatrix<5*N,1>* __restrict__ U,
    DenseMatrix<5*N,1>* __restrict__ rhs,
    uInt range_start, uInt range_end)
{
    uInt tid = blockIdx.x * blockDim.x + threadIdx.x;
    uInt cid = range_start + tid;
    if (cid >= range_end) return;


    using Basis = DGBasisEvaluator<Order>;
    // constexpr uInt N = Basis::NumBasis;
    constexpr uInt num_vol_points = GaussQuadCell::num_points;
    constexpr auto Qpoints = GaussQuadCell::get_points();
    constexpr auto Qweights = GaussQuadCell::get_weights();

    // printf("cid = %d\n", cid);
    const GPUTetrahedron& cell = mesh_cells[cid];
    const DenseMatrix<5*N,1>& coef = U[cid];  // 5*N 个 DoFs
    DenseMatrix<5*N,1> result = DenseMatrix<5*N,1>::Zeros();  // 5*N 个 DoFs
    for (uInt g = 0; g < num_vol_points; ++g) {
        const vector3f& xi = Qpoints[g];
        // 积分点 权重之和为 1/6，这里只需要体积即可，而非 Det[Jac]
        const Scalar& w = Qweights[g] * cell.volume * 6; 

        auto basis = Basis::eval_all(xi[0], xi[1], xi[2]);
        auto grads = Basis::grad_all(xi[0], xi[1], xi[2]);
        DenseMatrix<5,1> U_val = DenseMatrix<5,1>::Zeros();
        PragmaUnroll
        for (uInt bid = 0; bid < N; ++bid) {
            PragmaUnroll
            for (uInt k = 0; k < 5; ++k) {
                U_val(k,0) += basis[bid] * coef(5*bid+k,0);
            }
        }
        const DenseMatrix<5,3>& FU = Flux::computeFlux(U_val);

        const DenseMatrix<3,3>& Jinv = cell.invJac;
        PragmaUnroll
        for (uInt j = 0; j < N; ++j) {
            DenseMatrix<3,1> grad_phi_j = DenseMatrix<3,1>(grads[j]);
            DenseMatrix<5,1> flux = FU.multiply(Jinv.multiply(grad_phi_j));
            
            // 体积分部分，不会出现多个线程写入到同一个 cid 的情况
            // 但面积分的时候，同一个面的两侧单元，
            // 可能会有多个线程（多个面）同时写入 同一个单元
            result(5*j+0,0) -= flux(0,0) * w;
            result(5*j+1,0) -= flux(1,0) * w;
            result(5*j+2,0) -= flux(2,0) * w;
            result(5*j+3,0) -= flux(3,0) * w;
            result(5*j+4,0) -= flux(4,0) * w;
        }
    }
    rhs[cid] = result;
}

// Kernel launcher
template<uInt Order, typename Flux, typename GaussQuadCell, typename GaussQuadFace>
void ExplicitConvectionGPU<Order, Flux, GaussQuadCell, GaussQuadFace>::eval_cells(
    const DeviceMesh& mesh, const LongVectorDevice<5*N>& U, LongVectorDevice<5*N>& rhs)
{
    // dim3 block(256);
    // dim3 grid( (mesh.num_cells() + block.x - 1) / block.x );
    // eval_cells_kernel<Order, N, Flux, QuadC, QuadF><<<grid, block>>>(mesh.device_cells(), mesh.num_cells(),
    //                 U.d_blocks, rhs.d_blocks);
    for(int g=0; g<dev_cnt_; ++g) {
        cudaSetDevice(g);

        // eval_cells
        uInt start,end;
        split_range(mesh.num_cells(), g, dev_cnt_, start, end);
        dim3 block(256), grid((end-start+block.x-1)/block.x);
        eval_cells_kernel<Order, N, Flux, QuadC, QuadF>
            <<<grid,block>>>(mgpu_mesh_[g].device_cells(),
                            mgpu_U_[g].d_blocks,
                            mgpu_rhs_[g].d_blocks,
                            start,end);
        // cudaError_t err = cudaGetLastError();
        // if (err != cudaSuccess) {
        //     printf("ExplicitConvectionGPU::eval_cells      ranges: %6ld %6ld  CUDA kernel launch error: %s, GPU id: %d\n", start, end, cudaGetErrorString(err), g);
        // }
        // cudaDeviceSynchronize();
    }
}
