// dg/dg_schemes/explicit_convection_gpu/explicit_convection_gpu_impl.cuh
#pragma once
#include "dg/dg_schemes/explicit_convection_gpu/explicit_convection_gpu.cuh"
#include "dg/dg_schemes/explicit_convection_gpu/explicit_convection_gpu_kernels.cuh"

template<uInt Order, uInt N_state, typename GaussQuadCell, typename GaussQuadFace>
void ExplicitConvectionGPU<Order, N_state, GaussQuadCell, GaussQuadFace>::eval(
    const DeviceMesh& mesh,
    const LongVectorDevice<N_state * N_basis>& U,
    LongVectorDevice<N_state * N_basis>& rhs,
    Scalar time
) {
    eval_cells(mesh, U, rhs);
    eval_internals(mesh, U, rhs);
    eval_boundarys(mesh, U, rhs, time);
}

// 其他 launcher 略，类似


// Kernel launcher
template<uInt Order, uInt N_state, typename GaussQuadCell, typename GaussQuadFace>
void ExplicitConvectionGPU<Order, N_state, GaussQuadCell, GaussQuadFace>::eval_cells(
    const DeviceMesh& mesh,
    const LongVectorDevice<N_state * N_basis>& U,
    LongVectorDevice<N_state * N_basis>& rhs
) {
    dim3 block(256);
    dim3 grid( (mesh.num_cells() + block.x - 1) / block.x );
    eval_cells_kernel<Order, N_state, QuadC, QuadF><<<grid, block>>>(
        mesh.device_cells(), 
        mesh.num_cells(),
        U.d_blocks, 
        rhs.d_blocks,
        &physical_flux
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

template<uInt Order, uInt N_state, typename GaussQuadCell, typename GaussQuadFace>
void ExplicitConvectionGPU<Order, N_state, GaussQuadCell, GaussQuadFace>::eval_internals(
    const DeviceMesh& mesh,
    const LongVectorDevice<N_state * N_basis>& U,
    LongVectorDevice<N_state * N_basis>& rhs
) {
    dim3 block(256);
    dim3 grid((mesh.num_faces() + block.x - 1) / block.x);
    eval_internals_kernel<Order, N_state, QuadF><<<grid, block>>>(
        mesh.device_faces(),
        mesh.num_faces(),
        U.d_blocks,
        rhs.d_blocks,
        &numerical_flux
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

template<uInt Order, uInt N_state, typename GaussQuadCell, typename GaussQuadFace>
void ExplicitConvectionGPU<Order, N_state, GaussQuadCell, GaussQuadFace>::eval_boundarys(
    const DeviceMesh& mesh,
    const LongVectorDevice<N_state * N_basis>& U,
    LongVectorDevice<N_state * N_basis>& rhs,
    Scalar time
) {
    dim3 block(256);
    dim3 grid((mesh.num_faces() + block.x - 1) / block.x);
    eval_boundarys_kernel<Order, N_state, QuadF><<<grid, block>>>(
        mesh.device_faces(),
        mesh.num_faces(),
        mesh.device_cells(),
        mesh.num_cells(),
        mesh.device_points(),
        time,
        U.d_blocks,
        rhs.d_blocks,
        &numerical_flux
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}