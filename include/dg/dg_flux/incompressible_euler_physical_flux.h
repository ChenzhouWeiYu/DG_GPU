// include/DG/Flux/EulerPhysicalFlux.h
#pragma once
#include "base/type.h"
#include "matrix/matrix.h"

class InCompressibleEulerPhysicalFlux {
public:
    static constexpr Scalar epslion = 1e-16;
    
    // 改为静态成员函数
    static DenseMatrix<4,3> computeFlux(const DenseMatrix<4,1>& U);
    static std::array<DenseMatrix<4,4>,3> computeJacobian(const DenseMatrix<4,1>& U);
    static Scalar computeWaveSpeed(const DenseMatrix<5,1>& U_L, const DenseMatrix<4,1>& U_R);

private:
    static Scalar positive(Scalar val);
    static Scalar enthalpy(const DenseMatrix<4,1>& U);
    static Scalar soundSpeed(const DenseMatrix<4,1>& U);
    static Scalar waveSpeedMagnitude(const DenseMatrix<4,1>& U);
    static Scalar computePressure(const DenseMatrix<4,1>& U);
};
