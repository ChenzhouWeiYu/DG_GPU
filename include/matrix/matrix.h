#pragma once
#include "base/type.h"

#include "matrix/long_vector.h"
#include "matrix/dense_matrix.h"
#include "matrix/sparse_matrix.h"

#ifdef __CUDACC__
#include "matrix/long_vector_device.cuh"
#endif

using Vector3 = DenseMatrix<3, 1>; // 主要用于法向量