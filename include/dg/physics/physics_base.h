#pragma once
#include "base/type.h"
#include "matrix/dense_matrix.h"

/// 物理模型 CRTP 基类
template<typename Derived, uInt NEQN>
class PhysicsBase {
public:
    HostDevice __forceinline__
    DenseMatrix<NEQN, 3> compute_flux(const DenseMatrix<NEQN, 1>& U) const {
        return static_cast<const Derived*>(this)->compute_flux_impl(U);
    }
    
    HostDevice __forceinline__
    DenseMatrix<NEQN, 1> compute_flux_1d(const DenseMatrix<NEQN, 1>& U) const {
        return static_cast<const Derived*>(this)->compute_flux_1d_impl(U);
    }
    
    HostDevice __forceinline__
    DenseMatrix<NEQN, 1> compute_flux_dot_vec(
        const DenseMatrix<NEQN, 1>& U,
        Scalar vx, Scalar vy, Scalar vz) const {
        auto flux = compute_flux(U);
        DenseMatrix<NEQN, 1> result;
        for (uInt i = 0; i < NEQN; ++i) {
            result(i, 0) = flux(i, 0) * vx + flux(i, 1) * vy + flux(i, 2) * vz;
        }
        return result;
    }

    HostDevice __forceinline__
    DenseMatrix<NEQN, 1> compute_flux_dot_vec(
        const DenseMatrix<NEQN, 1>& U,
        const vector3f& vec) const {
        return compute_flux_dot_vec(U, vec[0], vec[1], vec[2]);
    }

    HostDevice __forceinline__
    DenseMatrix<NEQN, 1> compute_source(const DenseMatrix<NEQN, 1>& U) const {
        return static_cast<const Derived*>(this)->compute_source_impl(U);
    }

    HostDevice __forceinline__
    Scalar compute_pressure(const DenseMatrix<NEQN, 1>& U) const {
        return static_cast<const Derived*>(this)->compute_pressure_impl(U);
    }

    HostDevice __forceinline__
    Scalar compute_sound_speed(const DenseMatrix<NEQN, 1>& U) const {
        return static_cast<const Derived*>(this)->compute_sound_speed_impl(U);
    }    

    HostDevice __forceinline__
    DenseMatrix<NEQN, 1> compute_pressure_gradient(const DenseMatrix<NEQN, 1>& U) const {
        return static_cast<const Derived*>(this)->compute_pressure_gradient_impl(U);
    }

    HostDevice __forceinline__
    Scalar compute_pressure_directional_derivative(
        const DenseMatrix<NEQN, 1>& U,
        const DenseMatrix<NEQN, 1>& V) const {
        auto grad_p = compute_pressure_gradient(U);
        Scalar result = 0.0;
        for (uInt i = 0; i < NEQN; ++i) {
            result += grad_p[i] * V[i];
        }
        return result;
    }



protected:
    HostDevice __forceinline__
    Scalar positive(Scalar val) const { 
        return fmax(val, 1e-16); 
        // return val;
    }
};