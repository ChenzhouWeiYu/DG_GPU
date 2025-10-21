#pragma once
#include "base/type.h"
#include "matrix/dense_matrix.h"

/// 物理模型 CRTP 基类
template<typename Derived, uInt NEQN>
class PhysicsBase {
public:

    HostDevice
    DenseMatrix<NEQN, 3> computeFlux(const DenseMatrix<NEQN, 1>& U) const {
        return static_cast<const Derived*>(this)->computeFluxImpl(U);
    }

    HostDevice
    Scalar computePressure(const DenseMatrix<NEQN, 1>& U) const {
        return static_cast<const Derived*>(this)->computePressureImpl(U);
    }

    HostDevice
    Scalar computeSoundSpeed(const DenseMatrix<NEQN, 1>& U) const {
        return static_cast<const Derived*>(this)->computeSoundSpeedImpl(U);
    }    

protected:
    HostDevice 
    inline Scalar positive(Scalar val) const { 
        return std::max(val, 1e-16); 
    }
};