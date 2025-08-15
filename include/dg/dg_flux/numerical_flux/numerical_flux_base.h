// dg/dg_flux/numerical_flux/numerical_flux_base.h
#pragma once
#include "base/type.h"
#include "matrix/matrix.h"
#include "dg/dg_flux/combustion_flux/thermo_dynamics.h"
#include "dg/dg_flux/physical_flux/physical_flux_base.h"

template<uInt N>
class NumericalFlux {
protected:
    const Thermodynamics<N>& thermo;
    const PhysicalFlux<N>& physical_flux;  // 组合物理通量

public:
    // 构造时传入 thermo 和 physical_flux
    NumericalFlux(const Thermodynamics<N>& t, const PhysicalFlux<N>& pflux)
        : thermo(t), physical_flux(pflux) {}

    virtual ~NumericalFlux() = default;

    virtual DenseMatrix<5 + N, 1> compute(
        const DenseMatrix<5 + N, 1>& UL,
        const DenseMatrix<5 + N, 1>& UR,
        const Vector3& normal
    ) const = 0;
};