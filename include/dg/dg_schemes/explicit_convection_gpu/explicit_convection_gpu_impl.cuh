#pragma once

#include "dg/dg_schemes/explicit_convection_gpu/explicit_convection_gpu.cuh"


__device__ inline vector3f transform_to_cell(const GPUTriangleFace& face, const vector2f& uv, uInt side) {
    const vector3f& nc0 = face.natural_coords[side][0];
    const vector3f& nc1 = face.natural_coords[side][1];
    const vector3f& nc2 = face.natural_coords[side][2];
    
    Scalar uv0 = 1 - uv[0] - uv[1];
    Scalar uv1 = uv[0];
    Scalar uv2 = uv[1];

    // 手动展开每个分量的乘法和加法
    Scalar x = nc0[0] * uv0 + nc1[0] * uv1 + nc2[0] * uv2;
    Scalar y = nc0[1] * uv0 + nc1[1] * uv1 + nc2[1] * uv2;
    Scalar z = nc0[2] * uv0 + nc1[2] * uv1 + nc2[2] * uv2;

    vector3f result{x, y, z};
    return result;
}

inline void split_range(uInt n, int dev_id, int dev_cnt, uInt& start, uInt& end) {
    uInt per = (n + dev_cnt - 1) / dev_cnt;
    start = std::min<uInt>(n, (uInt)dev_id * per);
    end   = std::min<uInt>(n, start + per);
}

template<uInt N>
__global__ void add_kernel(DenseMatrix<5*N,1>* a,
                           const DenseMatrix<5*N,1>* b,
                           uInt n){
    uInt tid = blockIdx.x*blockDim.x + threadIdx.x;
    if(tid>=n) return;
    for(int k=0; k<5*N; ++k)
        a[tid][k] += b[tid][k];
}


template<uInt Order, typename Flux, typename GaussQuadCell, typename GaussQuadFace>
void ExplicitConvectionGPU<Order, Flux, GaussQuadCell, GaussQuadFace>::eval(
    const DeviceMesh& mesh, 
    const LongVectorDevice<5*N>& U, 
    LongVectorDevice<5*N>& rhs, Scalar time)
{   

    for(int g=0; g<dev_cnt_; ++g) {
        cudaSetDevice(g);
        // 拷贝外部 U 到每个 GPU
        cudaMemcpy(mgpu_U_[g].d_blocks, U.d_blocks,
                mesh.num_cells()*sizeof(DenseMatrix<5*N,1>),
                cudaMemcpyDeviceToDevice);
        cudaMemset(mgpu_rhs_[g].d_blocks, 0, mesh.num_cells()*sizeof(DenseMatrix<5*N,1>));
    }

    // 每个 GPU launch 三个 kernel
    // for(int g=0; g<dev_cnt_; ++g) {
    //     cudaSetDevice(g);

    //     // eval_cells
    //     uInt start,end;
    //     split_range(mesh.num_cells(), g, dev_cnt_, start, end);
    //     dim3 block(256), grid((end-start+block.x-1)/block.x);
    //     eval_cells_kernel<Order, N, Flux, QuadC, QuadF>
    //         <<<grid,block>>>(mgpu_mesh_[g].device_cells(),
    //                         mgpu_U_[g].d_blocks,
    //                         mgpu_rhs_[g].d_blocks,
    //                         start,end);

    //     // eval_internals
    //     split_range(mesh.num_faces(), g, dev_cnt_, start, end);
    //     grid = dim3((end-start+block.x-1)/block.x);
    //     eval_internals_kernel<Order, N, Flux, QuadC, QuadF>
    //         <<<grid,block>>>(mgpu_mesh_[g].device_faces(),
    //                         mgpu_U_[g].d_blocks,
    //                         mgpu_rhs_[g].d_blocks,
    //                         start,end);

    //     // eval_boundarys
    //     split_range(mesh.num_faces(), g, dev_cnt_, start, end);
    //     grid = dim3((end-start+block.x-1)/block.x);
    //     eval_boundarys_kernel<Order, N, Flux, QuadC, QuadF>
    //         <<<grid,block>>>(mgpu_mesh_[g].device_faces(),
    //                         mgpu_mesh_[g].device_cells(),
    //                         mgpu_mesh_[g].device_points(),
    //                         time,
    //                         mgpu_U_[g].d_blocks,
    //                         mgpu_rhs_[g].d_blocks,
    //                         start,end);
    // }
    
    eval_cells(mesh, U, rhs);
    eval_internals(mesh, U, rhs);
    eval_boundarys(mesh, U, rhs, time);
    // 全局规约：
    // 先把 GPU0 的 rhs 拷贝到外部 rhs 上
    cudaMemcpyPeer(rhs.d_blocks, 0, 
            mgpu_rhs_[0].d_blocks, 0, 
            mesh.num_cells()*sizeof(DenseMatrix<5*N,1>));
    // 将每个 GPU 的 rhs 拷贝到 GPU0 上，并累加到 rhs 上
    for(int g=1; g<dev_cnt_; ++g) {
        cudaSetDevice(0);
        cudaMemcpyPeer(mgpu_rhs_[0].d_blocks, 0, 
                mgpu_rhs_[g].d_blocks, g,
                mesh.num_cells()*sizeof(DenseMatrix<5*N,1>));
        dim3 block(256), grid((mesh.num_cells()+block.x-1)/block.x);
        add_kernel<N><<<grid,block>>>(rhs.d_blocks, mgpu_rhs_[0].d_blocks, mesh.num_cells());
        cudaDeviceSynchronize();  // 等待 kernel 完成
    }



}


