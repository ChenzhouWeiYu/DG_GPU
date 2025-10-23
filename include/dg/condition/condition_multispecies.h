// condition_multispecies.h
#pragma once
#include "condition_interface.h"

template<uInt Ns, typename Physics>
class MultiSpeciesCondition : public ConditionInterface<MultiSpeciesCondition<Ns, Physics>, Physics> {
public:
    static constexpr uInt NEQN = Ns + 4;
    using Base = ConditionInterface<MultiSpeciesCondition<Ns, Physics>, Physics>;
    using Base::Base;

    HostDevice
    Scalar rhoImpl(const vector3f& xyz, Scalar t) const {
        return 1.0; // 假设密度为常数
    }

    HostDevice
    Scalar uImpl(const vector3f& xyz, Scalar t) const {
        return 1.0;
    }

    HostDevice
    Scalar vImpl(const vector3f& xyz, Scalar t) const {
        return 0.0;
    }

    HostDevice
    Scalar wImpl(const vector3f& xyz, Scalar t) const {
        return 0.0;
    }

    HostDevice
    Scalar pImpl(const vector3f& xyz, Scalar t) const {
        return 1.0;
    }

    HostDevice
    Scalar yImpl(uInt i, const vector3f& xyz, Scalar t) const {
        if (i == 0) return 0.5 * (1.0 - tanh(fabs(xyz[0]) - 0.5)); // H2
        else return 1.0 - yImpl(0, xyz, t); // O2
    }

    HostDevice
    Scalar eImpl(const vector3f& xyz, Scalar t) const {
        return this->eFromP(xyz, t);
    }

    HostDevice
    DenseMatrix<NEQN,1> computeImpl(const vector3f& xyz, Scalar t) const {
        Scalar rho = computeRho(xyz, t);
        Scalar u   = computeU(xyz, t);
        Scalar v   = computeV(xyz, t);
        Scalar w   = computeW(xyz, t);
        Scalar e   = computeE(xyz, t);

        DenseMatrix<NEQN,1> U;
        for (uInt i = 0; i < Ns; ++i) {
            U(i,0) = rho * yImpl(i, xyz, t);
        }
        U(Ns,0) = rho * u;
        U(Ns+1,0) = rho * v;
        U(Ns+2,0) = rho * w;
        U(Ns+3,0) = rho * e;
        return U;
    }
};