#pragma once
#include "base/type.h"
#include "mesh/device_mesh.cuh"
#include "matrix/matrix.h"
#include "dg/dg_basis/dg_basis.h"


template<typename T>
__global__ void reduce_min_kernel(const T* input, T* output, uInt n) {
    extern __shared__ T sdata[];  // 动态共享内存

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // 初始化
    sdata[tid] = (gid < n) ? input[gid] : std::numeric_limits<T>::max();
    __syncthreads();

    // 块内规约（倒序展开）
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (sdata[tid + stride] < sdata[tid])
                sdata[tid] = sdata[tid + stride];
        }
        __syncthreads();
    }

    // 每个 block 的结果写入 output[blockIdx.x]
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}



template<typename T>
T device_reduce_min(const T* d_input, uInt n) {
    // printf("device_reduce_min: n = %ld\n", n); fflush(stdout);
    
    const uInt block_size = 256;
    const uInt max_blocks = 65535;
    uInt grid_size = (n + block_size - 1) / block_size;
    grid_size = min(grid_size, max_blocks);
    
    // printf("grid_size = %ld\n", grid_size); fflush(stdout);

    // 临时存储每个 block 的最小值
    T* d_block_mins;
    cudaMalloc(&d_block_mins, grid_size * sizeof(T));
    
    // printf("cudaMalloc succeeded\n"); fflush(stdout);


    // 第一次规约
    reduce_min_kernel<<<grid_size, block_size, block_size * sizeof(T)>>>(
        d_input, d_block_mins, n);
    
    // 如果只有一个 block，直接返回
    if (grid_size == 1) {
        T h_min;
        cudaMemcpy(&h_min, d_block_mins, sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(d_block_mins);
        return h_min;
    }
    if (grid_size == 2) {
        // T d_min = std::min(d_block_mins[0],d_block_mins[1]);
        T h_min[2];
        cudaMemcpy(h_min, d_block_mins, 2*sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(d_block_mins);
        return std::min(h_min[0],h_min[1]);
    }
    T h_min = device_reduce_min(d_block_mins, grid_size);
    
    cudaFree(d_block_mins);
    return h_min;
    

    // 否则递归或在 host 上处理（grid_size 通常很小）
    // std::vector<T> h_block_mins(grid_size);
    // cudaMemcpy(h_block_mins.data(), d_block_mins, grid_size * sizeof(T), cudaMemcpyDeviceToHost);
    // cudaFree(d_block_mins);
    
    // return *std::min_element(h_block_mins.begin(), h_block_mins.end());
}



template<typename T>
__global__ void reduce_min_kernel_inplace(T* data, uInt n, uInt grid_size) {
    extern __shared__ T sdata[];

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // strided 全局索引：k, k+grid_size, k+2*grid_size, ...
    uInt stride = grid_size;
    uInt gid = bid + tid * stride;

    // 加载：每个线程加载一个 strided 元素
    sdata[tid] = (gid < n) ? data[gid] : std::numeric_limits<T>::max();
    __syncthreads();

    // 块内规约（倒序展开）
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] < sdata[tid])
                sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }

    // 写回原数组的 [bid] 位置
    if (tid == 0) {
        data[bid] = sdata[0];
    }
}

template<typename T>
T device_reduce_min_inplace(T* d_data, uInt n) {
    const uInt block_size = 256;
    const uInt max_blocks = 65535;

    while (n > 1) {
        uInt grid_size = (n + block_size - 1) / block_size;
        grid_size = min(grid_size, max_blocks);

        // 调用 in-place kernel
        reduce_min_kernel_inplace<<<grid_size, block_size, block_size * sizeof(T)>>>(
            d_data, n, grid_size);

        // 更新 n = grid_size，继续规约
        n = grid_size;
    }

    // 最终结果在 d_data[0]
    T h_min;
    cudaMemcpy(&h_min, d_data, sizeof(T), cudaMemcpyDeviceToHost);
    return h_min;
}






template<uInt Order, typename QuadC, typename Basis>
__global__ void reconstruct_and_speed_kernel(
    const GPUTetrahedron* d_cells, uInt num_cells,
    const DenseMatrix<5 * Basis::NumBasis, 1>* coef,
    // DenseMatrix<5 * QuadC::num_points, 1>* U_h,
    // Scalar* wave_speeds,
    Scalar* h_i_lam,
    Scalar gamma)
{
    const uInt cellId = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellId >= num_cells) return;

    constexpr auto Qpoints = QuadC::get_points();
    constexpr auto Qweights = QuadC::get_weights();

    const auto& cell = d_cells[cellId];
    const DenseMatrix<5 * Basis::NumBasis, 1>& coef_cell = coef[cellId];
    DenseMatrix<5 * QuadC::num_points, 1> U_reconstructed;
    Scalar lambda = 0.0;

    for (uInt q = 0; q < QuadC::num_points; ++q) {
        auto Uq = DGBasisEvaluator<Order>::template coef2filed<5, Scalar>(coef_cell, Qpoints[q]);

        for (int i = 0; i < 5; ++i)
            U_reconstructed(5 * q + i, 0) = Uq(i, 0);

        Scalar rho = Uq(0, 0), rhou = Uq(1, 0), rhov = Uq(2, 0), rhow = Uq(3, 0), rhoE = Uq(4, 0);
        Scalar ke = (rhou*rhou + rhov*rhov + rhow*rhow) / (2.0 * rho + 1e-12);
        Scalar p = (gamma - 1.0) * (rhoE - ke);
        Scalar a = sqrt(gamma * p / rho);
        Scalar u = rhou / rho, v = rhov / rho, w = rhow / rho;
        Scalar vel = sqrt(u*u + v*v + w*w);
        lambda += (a + vel) * Qweights[q] * 6.0;
    }

    // U_h[cellId] = U_reconstructed;
    h_i_lam[cellId] = cell.m_h/lambda;
}





template<uInt Order, typename QuadC, typename Basis>
Scalar compute_CFL_time_step(
    const ComputingMesh& cpu_mesh,
    const DeviceMesh& gpu_mesh,
    const LongVectorDevice<5 * Basis::NumBasis>& coef_device,
    Scalar CFL,
    Scalar gamma)
{
    const uInt num_cells = gpu_mesh.num_cells();
    const uInt Q = QuadC::num_points;

    // 分配 GPU 缓存
    LongVectorDevice<5 * Q> U_h(num_cells);
    Scalar* d_h_i_lam;
    cudaMalloc(&d_h_i_lam, num_cells * sizeof(Scalar));

    // 启动 kernel：重构 & 波速估计
    dim3 block(128);
    dim3 grid((num_cells + block.x - 1) / block.x);
    reconstruct_and_speed_kernel<Order, QuadC, Basis>
        <<<grid, block>>>(gpu_mesh.device_cells(), num_cells,
                          coef_device.d_blocks,
                        //   U_h.d_blocks,
                          d_h_i_lam,
                          gamma);
    
    // Scalar min_dt = device_reduce_min(d_h_i_lam, num_cells);
    Scalar min_dt = device_reduce_min_inplace(d_h_i_lam, num_cells);
    // 下载结果
    // std::vector<Scalar> h_wave_speeds(num_cells);
    // cudaMemcpy(h_wave_speeds.data(), d_h_i_lam,
    //            num_cells * sizeof(Scalar), cudaMemcpyDeviceToHost);

    cudaFree(d_h_i_lam);
    // Scalar min_dt = *std::min_element(h_wave_speeds.begin(),h_wave_speeds.end());

    Scalar dt = CFL * min_dt / std::pow(2 * Order + 1, 1); // k=1
    return dt;
}