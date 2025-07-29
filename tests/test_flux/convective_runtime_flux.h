// include/DG/Flux/ConvectiveRuntimeFlux.h
#pragma once
#include "base/type.h"
#include "matrix/matrix.h"

class ConvectiveRuntimeFlux {
public:
    // 物理参数配置
    static void configure(Scalar gamma, Scalar mach_ref = 1.0) {
        s_gamma = gamma;
        s_mach_ref = mach_ref;
        s_epsilon = 1e-16 * mach_ref; // 根据参考马赫数缩放阈值
    }

    // 获取当前参数
    static Scalar getGamma() { return s_gamma; }
    static Scalar getMachRef() { return s_mach_ref; }

    // 核心计算接口
    static DenseMatrix<5,3> computeFlux(const DenseMatrix<5,1>& U);
    static std::array<DenseMatrix<5,5>,3> computeJacobian(const DenseMatrix<5,1>& U);
    static Scalar computeWaveSpeed(const DenseMatrix<5,1>& U_L, 
                                 const DenseMatrix<5,1>& U_R);

private:
    // C++17内联静态成员
    static inline Scalar s_gamma = 1.4;
    static inline Scalar s_mach_ref = 1.0;
    static inline Scalar s_epsilon = 1e-16;

    // 内部工具方法
    static Scalar computePressure(const DenseMatrix<5,1>& U) {
        const Scalar rho = std::max(U[0], s_epsilon);
        const vector3f vel{U[1]/rho, U[2]/rho, U[3]/rho};
        return (s_gamma-1) * (U[4] - 0.5*rho*vel.dot(vel));
    }
};
