// include/DG/Schemes/ImplicitConvectionRuntime.h
#pragma once
#include "DG/Flux/ConvectiveRuntimeFlux.h"

template<uInt Order>
class ImplicitConvectionRuntime {
public:
    static void configure(Scalar gamma, Scalar mach_ref = 1.0) {
        ConvectiveRuntimeFlux::configure(gamma, mach_ref);
    }

    static void assemble(
        const ComputingMesh& mesh,
        const LongVector<5*Order>& U,
        BlockSparseMatrix<5*Order,5*Order>& mat,
        LongVector<5*Order>& rhs);
};