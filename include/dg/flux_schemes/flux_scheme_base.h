#pragma once
#include "base/type.h"
#include "matrix/dense_matrix.h"

template<typename Physics>
class FluxSchemeBase {
public:
    static constexpr uInt NEQN = Physics::NEQN;

    // HostDevice
    // static DenseMatrix<NEQN, 1> compute(
    //     const Physics& physics,
    //     const DenseMatrix<NEQN, 1>& UL,
    //     const DenseMatrix<NEQN, 1>& UR,
    //     Scalar nx, Scalar ny, Scalar nz);

    
    // HostDevice
    // static DenseMatrix<NEQN, 1> compute(
    //     const Physics& physics,
    //     const DenseMatrix<NEQN, 1>& UL,
    //     const DenseMatrix<NEQN, 1>& UR,
    //     vector3f vec) {
    //         return compute(physics, UL, UR, vec[0], vec[1], vec[2]);
    //     }
        

protected:
    HostDevice __forceinline__ 
    static Scalar positive(Scalar val) { 
        return fmax(val, 1e-16); 
        // return val;
    }
};