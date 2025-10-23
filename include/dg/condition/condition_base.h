// condition_base.h
#pragma once
#include "base/type.h"
#include "matrix/dense_matrix.h"

template<typename Physics>
class ConditionBase {
public:
    using PhysicsType = Physics;
    static constexpr uInt NEQN = Physics::NEQN;

    // 构造函数：传入物理模型
    explicit ConditionBase(const Physics& physics) : physics_(physics) {}

    // 计算物理量（纯虚函数，但使用 CRTP 实现）
    HostDevice virtual DenseMatrix<NEQN,1> compute(const vector3f& xyz, Scalar t) const = 0;
protected:
    const Physics& physics_;
};