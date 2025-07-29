// include/DG/Flux/EulerPhysicalFlux.h
#pragma once
#include "base/type.h"
#include "matrix/matrix.h"

template<uInt GammaNumerator = 7, uInt GammaDenominator = 5>
class DiffusionPhysicalFlux {
public:
    static constexpr Scalar gamma = 
        static_cast<Scalar>(GammaNumerator) / static_cast<Scalar>(GammaDenominator);
    static constexpr Scalar Pr = 0.72;
    static constexpr Scalar epslion = 1e-16;
    
    // 改为静态成员函数
    static DenseMatrix<5,3> computeFlux(const DenseMatrix<5,1>& U,const DenseMatrix<5,3>& grad_U,const Scalar mu);
    static std::array<std::array<DenseMatrix<5,5>,3>,3> computeJacobian(const DenseMatrix<5,1>& U,const Scalar mu);
    static DenseMatrix<5,5> computevKu(const DenseMatrix<3,1>& v,const DenseMatrix<3,1>& u,const DenseMatrix<5,1>& U,const Scalar mu);
    static DenseMatrix<5,5> computevKu(const DenseMatrix<3,1>& v,const DenseMatrix<3,1>& u,const std::array<std::array<DenseMatrix<5,5>,3>,3>& J_G_U);
private:
    static Scalar positive(Scalar val);
    static Scalar computePressure(const DenseMatrix<5,1>& U);
};

// #include "EulerPhysicalFlux.cpp"

using MonatomicFluxD = DiffusionPhysicalFlux<5,3>;  // gamma=5/3
using AirFluxD = DiffusionPhysicalFlux<7,5>;         // gamma=7/5

