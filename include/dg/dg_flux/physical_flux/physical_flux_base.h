// dg/dg_flux/physical_flux/physical_flux_base.h
#pragma once
#include "base/type.h"
#include "matrix/matrix.h"
#include "dg/dg_flux/combustion_flux/thermo_dynamics.h"

template<uInt N>
class PhysicalFlux {
public:
    virtual ~PhysicalFlux() = default;

    // 计算完整通量张量 F = [F_x, F_y, F_z] ∈ R^{(5+N)×3}
    virtual DenseMatrix<5 + N, 3> compute(const typename Thermodynamics<N>::Primitive& prim) const = 0;

    // 计算 F · rhs_dir，用于体积分中的 ∇φ · F
    // rhs_dir 是测试函数梯度方向（如 ∂φ/∂x, ∂φ/∂y, ∂φ/∂z）
    virtual DenseMatrix<5 + N, 1> compute_dot(
        const typename Thermodynamics<N>::Primitive& prim,
        const Vector3& rhs_dir
    ) const {
        auto F = compute(prim);
        DenseMatrix<5 + N, 1> result;
        for (uInt i = 0; i < 5 + N; ++i) {
            result(i, 0) = F(i, 0)*rhs_dir[0] + F(i, 1)*rhs_dir[1] + F(i, 2)*rhs_dir[2];
        }
        return result;
    }
};