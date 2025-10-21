#pragma once
#include "dg/physics/physics_base.h"


/// 多组分物理模型（程序中设置物性参数）
template<uInt Ns>
class MultiComponentPhysics : public PhysicsBase<MultiComponentPhysics<Ns>, Ns + 4> {
public:
    static constexpr uInt NEQN = Ns + 4;

    // 物性参数（运行时设置）
    std::array<Scalar, Ns> gamma_list; // 每个组分的 gamma（简化模型）
    std::array<Scalar, Ns> M_list;     // 摩尔质量 [kg/kmol]

    HostDevice
    MultiComponentPhysics() {
        // 默认初始化（空气 + 甲烷示例）
        if constexpr (Ns >= 1) gamma_list[0] = 1.4;  // N2
        if constexpr (Ns >= 2) gamma_list[1] = 1.4;  // O2
        if constexpr (Ns >= 3) gamma_list[2] = 1.3;  // CH4
        // ... 可扩展

        if constexpr (Ns >= 1) M_list[0] = 28.0;  // N2
        if constexpr (Ns >= 2) M_list[1] = 32.0;  // O2
        if constexpr (Ns >= 3) M_list[2] = 16.0;  // CH4
    }

    HostDevice
    DenseMatrix<Ns + 4, 3> computeFluxImpl(const DenseMatrix<Ns + 4, 1>& U) const;

    HostDevice
    Scalar computePressureImpl(const DenseMatrix<Ns + 4, 1>& U) const;

    HostDevice
    Scalar computeSoundSpeedImpl(const DenseMatrix<Ns + 4, 1>& U) const;

    HostDevice
    Scalar mixtureGamma(const std::array<Scalar, Ns>& Y) const;
};