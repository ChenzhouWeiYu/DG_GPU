#pragma once
#include "base/type.h"
#include "matrix/matrix.h"


template<typename Flux>
class FluxTraits {
    // 主通量计算
    constexpr bool has_flux = requires(const DenseMatrix<5,1>& U) {
        { Flux::computeFlux(U) } -> std::same_as<DenseMatrix<5,3>>;
    };

    // 雅可比矩阵
    constexpr bool has_jacobian = requires(const DenseMatrix<5,1>& U) {
        { Flux::computeJacobian(U) } -> std::same_as<std::array<DenseMatrix<5,5>,3>>;
    };

    // 波速计算（可选）
    constexpr bool has_wave_speed = requires(
        const DenseMatrix<5,1>& U_L, 
        const DenseMatrix<5,1>& U_R) 
    {
        { Flux::computeWaveSpeed(U_L, U_R) } -> std::same_as<Scalar>;
    };

    static_assert(has_flux, "Flux must implement computeFlux()");
};