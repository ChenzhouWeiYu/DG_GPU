// dg/dg_schemes/explicit_convection_gpu/explicit_convection_gpu_kernels.cuh
#pragma once
#include "base/type.h"
#include "dg/dg_basis/dg_basis.h"
#include "mesh/device_mesh.cuh"
#include "matrix/matrix.h"
#include "dg/utils.h"
#include "dg/dg_flux/physical_flux/physical_flux_base.h"
#include "dg/dg_flux/numerical_flux/numerical_flux_base.h"

// 声明核函数
template<uInt Order, uInt N_state, typename GaussQuadCell, typename GaussQuadFace>
__global__ void eval_cells_kernel(
    const GPUTetrahedron* cells,
    uInt num_cells,
    const DenseMatrix<N_state * DGBasisEvaluator<Order>::NumBasis, 1>* U,
    DenseMatrix<N_state * DGBasisEvaluator<Order>::NumBasis, 1>* rhs,
    const PhysicalFlux<N_state - 5>* physical_flux
);

template<uInt Order, uInt N_state, typename GaussQuadCell, typename GaussQuadFace>
__global__ void eval_internals_kernel(
    const GPUTriangleFace* faces,
    uInt num_faces,
    const DenseMatrix<N_state * DGBasisEvaluator<Order>::NumBasis, 1>* U,
    DenseMatrix<N_state * DGBasisEvaluator<Order>::NumBasis, 1>* rhs,
    const NumericalFlux<N_state - 5>* numerical_flux
);

template<uInt Order, uInt N_state, typename GaussQuadCell, typename GaussQuadFace>
__global__ void eval_boundarys_kernel(
    const GPUTriangleFace* faces,
    uInt num_faces,
    const GPUTetrahedron* cells,
    uInt num_cells,
    const vector3f* points,
    Scalar time,
    const DenseMatrix<N_state * DGBasisEvaluator<Order>::NumBasis, 1>* U,
    DenseMatrix<N_state * DGBasisEvaluator<Order>::NumBasis, 1>* rhs,
    const NumericalFlux<N_state - 5>* numerical_flux
);