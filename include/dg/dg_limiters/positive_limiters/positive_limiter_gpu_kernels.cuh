// include/DG/DG_Schemes/PositiveLimiterGPU_kernels.h
#pragma once

#include "base/type.h"
#include "matrix/dense_matrix.h"
#include "mesh/device_mesh.h"
#include "dg/dg_basis/dg_basis.h"

template<uInt Order, uInt NumBasis, typename QuadC, typename QuadF>
__global__ void construct_cell_extrema_kernel(const GPUTetrahedron*, uInt,
    const DenseMatrix<5 * NumBasis, 1>*, DenseMatrix<5, 1>*, DenseMatrix<5, 1>*);

template<uInt Order, uInt NumBasis, typename QuadC, typename QuadF>
__global__ void construct_cell_avg_extrema_kernel(const GPUTetrahedron*, uInt,
    const DenseMatrix<5 * NumBasis, 1>*, DenseMatrix<5, 1>*, DenseMatrix<5, 1>*);

template<uInt Order, uInt NumBasis, typename QuadC, typename QuadF>
__global__ void gatter_cell_extrema_kernel(const GPUTetrahedron*, uInt,
    DenseMatrix<5, 1>*, DenseMatrix<5, 1>*, const DenseMatrix<5, 1>*, const DenseMatrix<5, 1>*);

template<uInt Order, uInt NumBasis, typename QuadC, typename QuadF>
__global__ void apply_extrema_limiter_kernel(const GPUTetrahedron*, uInt,
    DenseMatrix<5 * NumBasis, 1>*, const DenseMatrix<5, 1>*, const DenseMatrix<5, 1>*);

template<uInt Order, uInt NumBasis, typename QuadC, typename QuadF>
__global__ void apply_2_kernel(const GPUTetrahedron*, uInt,
    DenseMatrix<5 * NumBasis, 1>*, Scalar);
