// condition_interface.h
#pragma once
#include "base/type.h"
#include "matrix/dense_matrix.h"

template<typename Derived, typename Physics>
class ConditionInterface {
public:
    using PhysicsType = Physics;
    static constexpr uInt NEQN = Physics::NEQN;

    explicit ConditionInterface(const Physics& physics) : physics_(physics) {}


    HostDevice
    DenseMatrix<NEQN,1> compute(const vector3f& xyz, Scalar t) const {
        return static_cast<const Derived*>(this)->computeImpl(xyz, t);
    }

    // 辅助函数：从已知量推导其他量
    HostDevice
    Scalar computeRho(const vector3f& xyz, Scalar t) const {
        return static_cast<const Derived*>(this)->rhoImpl(xyz, t);
    }

    HostDevice
    Scalar computeU(const vector3f& xyz, Scalar t) const {
        return static_cast<const Derived*>(this)->uImpl(xyz, t);
    }

    HostDevice
    Scalar computeV(const vector3f& xyz, Scalar t) const {
        return static_cast<const Derived*>(this)->vImpl(xyz, t);
    }

    HostDevice
    Scalar computeW(const vector3f& xyz, Scalar t) const {
        return static_cast<const Derived*>(this)->wImpl(xyz, t);
    }

    HostDevice
    Scalar computeP(const vector3f& xyz, Scalar t) const {
        return static_cast<const Derived*>(this)->pImpl(xyz, t);
    }

    HostDevice
    Scalar computeT(const vector3f& xyz, Scalar t) const {
        return static_cast<const Derived*>(this)->tImpl(xyz, t);
    }

    HostDevice
    Scalar computeE(const vector3f& xyz, Scalar t) const {
        return static_cast<const Derived*>(this)->eImpl(xyz, t);
    }

protected:
    Physics physics_;

    // 默认实现：从已知量推导
    HostDevice
    Scalar rhoImpl(const vector3f& xyz, Scalar t) const {
        return static_cast<const Derived*>(this)->rhoImpl(xyz, t);
    }

    HostDevice
    Scalar uImpl(const vector3f& xyz, Scalar t) const {
        return static_cast<const Derived*>(this)->uImpl(xyz, t);
    }
    HostDevice
    Scalar vImpl(const vector3f& xyz, Scalar t) const {
        return static_cast<const Derived*>(this)->vImpl(xyz, t);
    }
    HostDevice
    Scalar wImpl(const vector3f& xyz, Scalar t) const {
        return static_cast<const Derived*>(this)->wImpl(xyz, t);
    }

    // ... 其他默认实现 ...

    // 如果用户只实现了 p，那么 e 可以这样写：
    // e = (p / ((gamma-1)*rho)) + 0.5*(u²+v²+w²)
    // 但更通用的是调用 physics 的函数
    // HostDevice
    // Scalar eFromP(const vector3f& xyz, Scalar t) const {
    //     Scalar rho = computeRho(xyz, t);
    //     Scalar u   = computeU(xyz, t);
    //     Scalar v   = computeV(xyz, t);
    //     Scalar w   = computeW(xyz, t);
    //     Scalar p   = computeP(xyz, t);
    //     // 使用 physics 的状态方程
    //     return physics_.computeInternalEnergyFromP(rho, p, u, v, w);
    // }
};