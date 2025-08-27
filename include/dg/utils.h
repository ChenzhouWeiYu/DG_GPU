// dg/utils/utils.h
#pragma once
#include "base/type.h"
#include "matrix/matrix.h"
#include "mesh/device_mesh.cuh"
#include "dg/dg_basis/dg_basis.h"

namespace dg_utils {

    // 面上自然坐标 → 体上自然坐标
    HostDevice ForceInline
    vector3f transform_to_cell(const GPUTriangleFace& face, const vector2f& uv, uInt side) {
        const vector3f& nc0 = face.natural_coords[side][0];
        const vector3f& nc1 = face.natural_coords[side][1];
        const vector3f& nc2 = face.natural_coords[side][2];
        
        Scalar u = uv[0], v = uv[1], w = 1.0f - u - v;
        return nc0 * w + nc1 * u + nc2 * v;
    }

    // 从系数重构守恒变量 U(x)
    template<uInt Order, uInt N_state, uInt N_basis>
    HostDevice ForceInline
    DenseMatrix<N_state, 1> reconstruct_state(
        const DenseMatrix<N_state * N_basis, 1>& coef,
        const std::array<Scalar,N_basis>& basis
    ) {
        DenseMatrix<N_state, 1> U;
        for (uInt k = 0; k < N_state; ++k) {
            U(k, 0) = 0.0;
            for (uInt bid = 0; bid < N_basis; ++bid) {
                U(k, 0) += basis[bid] * coef(N_state * bid + k, 0);
            }
        }
        return U;
    }

    // 从系数重构梯度 ∇U(x)
    template<uInt Order, uInt N_state, uInt N_basis>
    HostDevice ForceInline
    DenseMatrix<N_state, 3> reconstruct_gradient(
        const DenseMatrix<N_state * N_basis, 1>& coef,
        const std::array<std::array<Scalar,3>,N_basis>& grads,
        const DenseMatrix<3, 3>& Jinv
    ) {
        DenseMatrix<N_state, 3> G = DenseMatrix<N_state, 3>::Zeros();
        for (uInt bid = 0; bid < N_basis; ++bid) {
            auto grad_ref = grads[bid];  // ∂φ/∂ξ
            auto grad_phys = Jinv.multiply(grad_ref);  // ∂φ/∂x
            for (uInt k = 0; k < N_state; ++k) {
                for (uInt d = 0; d < 3; ++d) {
                    G(k, d) += grad_phys[d] * coef(N_state * bid + k, 0);
                }
            }
        }
        return G;
    }

} // namespace dg_utils