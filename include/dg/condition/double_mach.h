// condition_sine_wave.h
#pragma once
#include "condition_interface.h"

template<typename Physics>
class DoubleMachIBCondition : public ConditionInterface<DoubleMachIBCondition<Physics>, Physics> {
public:
    using Base = ConditionInterface<DoubleMachIBCondition<Physics>, Physics>;
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
        Scalar x = xyz[0], y = xyz[1], z = xyz[2];
        return x<1.0/6.0+y*(1+20*t)/std::sqrt(3.0) ? 8 : 1.4;
    }

    HostDevice __forceinline__
    Scalar uImpl(const vector3f& xyz, Scalar t) const {
        Scalar x = xyz[0], y = xyz[1], z = xyz[2];
        return x<1.0/6.0+y*(1+20*t)/std::sqrt(3.0) ? 8.25*std::cos(-M_PI/6.0) : 0;
    }

    HostDevice __forceinline__
    Scalar vImpl(const vector3f& xyz, Scalar t) const {
        Scalar x = xyz[0], y = xyz[1], z = xyz[2];
        return x<1.0/6.0+y*(1+20*t)/std::sqrt(3.0) ? 8.25*std::sin(-M_PI/6.0) : 0;
    }

    HostDevice __forceinline__
    Scalar wImpl(const vector3f& xyz, Scalar t) const {
        Scalar x = xyz[0], y = xyz[1], z = xyz[2];
        return 0.0;
    }

    HostDevice __forceinline__
    Scalar pImpl(const vector3f& xyz, Scalar t) const {
        Scalar x = xyz[0], y = xyz[1], z = xyz[2];
        return x<1.0/6.0+y*(1+20*t)/std::sqrt(3.0) ? 116.5 : 1;
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