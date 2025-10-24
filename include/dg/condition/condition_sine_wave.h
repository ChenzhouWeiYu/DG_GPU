// condition_sine_wave.h
#pragma once
#include "condition_interface.h"

template<typename Physics>
class SineWaveCondition : public ConditionInterface<SineWaveCondition<Physics>, Physics> {
public:
    using Base = ConditionInterface<SineWaveCondition<Physics>, Physics>;
    using Base::Base;       // 构造函数
    using Base::computeRho;
    using Base::computeU;
    using Base::computeV;
    using Base::computeW;
    using Base::computeP;
    using Base::computeT;
    using Base::computeE;
    // using Base::physics_;

    HostDevice __forceinline__
    Scalar rhoImpl(const vector3f& xyz, Scalar t) const {
        return 1.0 + 0.9 * sin(2*M_PI*(xyz[0] - t));
    }

    HostDevice __forceinline__
    Scalar uImpl(const vector3f& xyz, Scalar t) const {
        return 1.0;
    }

    HostDevice __forceinline__
    Scalar vImpl(const vector3f& xyz, Scalar t) const {
        return 0.0;
    }

    HostDevice __forceinline__
    Scalar wImpl(const vector3f& xyz, Scalar t) const {
        return 0.0;
    }

    HostDevice __forceinline__
    Scalar pImpl(const vector3f& xyz, Scalar t) const {
        return 1.0;
    }

    // e 由 p 推导
    HostDevice __forceinline__
    Scalar eImpl(const vector3f& xyz, Scalar t) const {
        Scalar rho = computeRho(xyz, t);
        Scalar u   = computeU(xyz, t);
        Scalar v   = computeV(xyz, t);
        Scalar w   = computeW(xyz, t);
        Scalar p   = computeP(xyz, t);
        // printf("p = %f gamma = %f\n", p, this->physics_.get_gamma());
        return p / (this->physics_.get_gamma() - 1) / rho + Scalar(0.5)*(u*u + v*v + w*w);
    }

    // computeImpl：组装守恒变量
    HostDevice __forceinline__
    DenseMatrix<5,1> computeImpl(const vector3f& xyz, Scalar t) const {
        Scalar rho = computeRho(xyz, t);
        Scalar u   = computeU(xyz, t);
        Scalar v   = computeV(xyz, t);
        Scalar w   = computeW(xyz, t);
        Scalar e   = computeE(xyz, t);
        return {rho, rho*u, rho*v, rho*w, rho*e};
    }
};