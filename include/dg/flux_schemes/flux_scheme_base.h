#pragma once
#include "base/type.h"
#include "matrix/dense_matrix.h"

template<typename Physics>
class FluxSchemeBase {
public:
    static constexpr uInt NEQN = Physics::NEQN;

    HostDevice
    static DenseMatrix<NEQN, 1> compute(
        const Physics& physics,
        const DenseMatrix<NEQN, 1>& UL,
        const DenseMatrix<NEQN, 1>& UR,
        Scalar nx, Scalar ny, Scalar nz);
        

protected:
    HostDevice 
    static inline Scalar positive(Scalar val) { 
        return std::max(val, 1e-16); 
    }
};