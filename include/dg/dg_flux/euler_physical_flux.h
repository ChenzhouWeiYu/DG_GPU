// include/DG/Flux/EulerPhysicalFlux.h
#pragma once
#include "base/type.h"
#include "matrix/matrix.h"

enum class NumericalFluxType : uint8_t {LF,LaxFriedrichs,Roe,HLL,HLLC,HLLEM};
#define FOREACH_FLUX_TYPE(F,SecondParams) \
    F(LF,SecondParams) \
    F(LaxFriedrichs,SecondParams) \
    F(Roe,SecondParams) \
    F(HLL,SecondParams) \
    F(HLLC,SecondParams) \
    F(HLLEM,SecondParams)

template<uInt GammaNumerator = 7, uInt GammaDenominator = 5, NumericalFluxType FluxType = NumericalFluxType::LF>
class EulerPhysicalFlux {
public:
    // static constexpr Scalar gamma = 
    //     static_cast<Scalar>(GammaNumerator) / static_cast<Scalar>(GammaDenominator);
    
    HostDevice static Scalar get_gamma() {return static_cast<Scalar>(GammaNumerator) / static_cast<Scalar>(GammaDenominator);}
    HostDevice static Scalar get_epslion() {return 1e-12;}
    
    // 改为静态成员函数
    HostDevice static DenseMatrix<5,3> computeFlux(const DenseMatrix<5,1>& U){
        const Scalar rho = positive(U[0]);
        const Scalar u = U[1]/rho, v = U[2]/rho, w = U[3]/rho, e = U[4]/rho;
        // const Scalar u2 = u*u + v*v + w*w;
        const Scalar p = computePressure(U);
        return {rho*u,rho*v,rho*w,
                rho*u*u+p,rho*v*u  ,rho*w*u  ,
                rho*u*v  ,rho*v*v+p,rho*w*v  ,
                rho*u*w  ,rho*v*w  ,rho*w*w+p,
                u*(rho*e+p),v*(rho*e+p),w*(rho*e+p)};
    }


    HostDevice static std::array<DenseMatrix<5,5>,3> computeJacobian(const DenseMatrix<5,1>& U) {
        const Scalar rho = positive(U[0]);
        const Scalar u = U[1]/U[0], v = U[2]/U[0], w = U[3]/U[0], e = U[4]/U[0];
        const Scalar u2 = u*u + v*v + w*w;
        const Scalar p = computePressure(U);
        Scalar gamma = get_gamma();
        const DenseMatrix<5,5> Jac_x = {0,1,0,0,0,  
                        -u*u+0.5*(gamma-1)*u2, 2*u-(gamma-1)*u, -(gamma-1)*v,-(gamma-1)*w,gamma-1,
                        -u*v,v,u,0,0,
                        -u*w,w,0,u,0,
                        u*(-gamma*e+(gamma-1)*u2),gamma*e-0.5*(gamma-1)*(2*u*u+u2),-(gamma-1)*u*v,-(gamma-1)*u*w,gamma*u};
        const DenseMatrix<5,5> Jac_y = {0,0,1,0,0,  
                        -u*v,v,u,0,0,
                        -v*v+0.5*(gamma-1)*u2, -(gamma-1)*u, 2*v-(gamma-1)*v,-(gamma-1)*w,gamma-1,
                        -v*w,0,w,v,0,
                        v*(-gamma*e+(gamma-1)*u2),-(gamma-1)*u*v,gamma*e-0.5*(gamma-1)*(2*v*v+u2),-(gamma-1)*v*w,gamma*v};
        const DenseMatrix<5,5> Jac_z = {0,0,0,1,0,  
                        -u*w,w,0,u,0,
                        -v*w,0,w,v,0,
                        -w*w+0.5*(gamma-1)*u2,-(gamma-1)*u,-(gamma-1)*v,2*w-(gamma-1)*w,gamma-1,
                        w*(-gamma*e+(gamma-1)*u2),-(gamma-1)*u*w,-(gamma-1)*v*w,gamma*e-0.5*(gamma-1)*(2*w*w+u2),gamma*w};
        return {Jac_x, Jac_y, Jac_z};
    }


    HostDevice static Scalar computeWaveSpeed(const DenseMatrix<5,1>& U_L, const DenseMatrix<5,1>& U_R){
        return std::max(
            waveSpeedMagnitude(U_L),
            waveSpeedMagnitude(U_R)
        );
    }

    HostDevice static Scalar waveSpeedMagnitude(const DenseMatrix<5,1>& U) {
        const Scalar rho = positive(U[0]);
        const Scalar u = U[1]/rho, v = U[2]/rho, w = U[3]/rho, e = U[4]/rho;
        return std::sqrt(u*u+v*v+w*w) + soundSpeed(U);
    }

    HostDevice static 
    DenseMatrix<5,1> computeNumericalFlux(
        const DenseMatrix<5,1>& UL,
        const DenseMatrix<5,1>& UR,
        const DenseMatrix<3,1>& normal){
        if constexpr (FluxType == NumericalFluxType::LF){
            return computeLaxFriedrichsFlux(UL,UR,normal);
        }
        if constexpr (FluxType == NumericalFluxType::LaxFriedrichs){
            return computeLaxFriedrichsFlux(UL,UR,normal);
        }
        if constexpr (FluxType == NumericalFluxType::Roe){
            return computeRoeFlux(UL,UR,normal);
        }
        if constexpr (FluxType == NumericalFluxType::HLL){
            return computeHLLFlux(UL,UR,normal);
        }
        if constexpr (FluxType == NumericalFluxType::HLLC){
            return computeHLLCFlux(UL,UR,normal);
        }
        if constexpr (FluxType == NumericalFluxType::HLLEM){
            return computeHLLEMFlux(UL,UR,normal);
        }
    }


private:
    HostDevice static 
    DenseMatrix<5,1> computeLaxFriedrichsFlux(
        const DenseMatrix<5,1>& UL,
        const DenseMatrix<5,1>& UR,
        const DenseMatrix<3,1>& normal)
    {
        auto FL = computeFlux(UL).multiply(normal);
        auto FR = computeFlux(UR).multiply(normal);

        Scalar aL = soundSpeed(UL);
        Scalar aR = soundSpeed(UR);
        Scalar rhoL = positive(UL[0]), rhoR = positive(UR[0]);
        Scalar uL = UL[1] / rhoL, vL = UL[2] / rhoL, wL = UL[3] / rhoL;
        Scalar uR = UR[1] / rhoR, vR = UR[2] / rhoR, wR = UR[3] / rhoR;
        Scalar unL = std::sqrt(uL*uL+vL*vL+wL*wL);
        Scalar unR = std::sqrt(uR*uR+vR*vR+wR*wR);

        Scalar SL = std::min(unL - aL, unR - aR);
        Scalar SR = std::max(unL + aL, unR + aR);
        // Scalar lambda = std::max(SR,-SL);

        Scalar lambda = computeWaveSpeed(UL, UR);
        DenseMatrix<5,1> flux;
        for (int i = 0; i < 5; ++i)
            flux[i] = 0.5 * (FL[i] + FR[i] + 0.5 * lambda * (UL[i] - UR[i]));
        return flux;
    }


    // HostDevice static DenseMatrix<5,1> computeRoeFlux(
    //     const DenseMatrix<5,1>& UL,
    //     const DenseMatrix<5,1>& UR,
    //     const DenseMatrix<3,1>& normal)
    // {
    //     Scalar gamma = get_gamma();

    //     // 提取原始变量
    //     Scalar rhoL = positive(UL[0]), rhoR = positive(UR[0]);
    //     Scalar uL = UL[1]/rhoL, vL = UL[2]/rhoL, wL = UL[3]/rhoL;
    //     Scalar uR = UR[1]/rhoR, vR = UR[2]/rhoR, wR = UR[3]/rhoR;
    //     Scalar pL = computePressure(UL), pR = computePressure(UR);
    //     Scalar HL = (UL[4] + pL) / rhoL;
    //     Scalar HR = (UR[4] + pR) / rhoR;

    //     // Roe 平均
    //     Scalar sqrtRhoL = std::sqrt(rhoL), sqrtRhoR = std::sqrt(rhoR);
    //     Scalar denom = sqrtRhoL + sqrtRhoR;
    //     Scalar uT = (sqrtRhoL*uL + sqrtRhoR*uR)/denom;
    //     Scalar vT = (sqrtRhoL*vL + sqrtRhoR*vR)/denom;
    //     Scalar wT = (sqrtRhoL*wL + sqrtRhoR*wR)/denom;
    //     Scalar HT = (sqrtRhoL*HL + sqrtRhoR*HR)/denom;

    //     // Roe 特征结构
    //     Scalar V2 = uT*uT + vT*vT + wT*wT;
    //     Scalar a2 = positive((gamma - 1.0) * (HT - 0.5 * V2));
    //     Scalar a = std::sqrt(a2);
    //     Scalar un = uT*normal[0] + vT*normal[1] + wT*normal[2];

    //     // delta U
    //     DenseMatrix<5,1> deltaU = UR - UL;

    //     // 构造右特征矩阵 R（列向量是右特征向量）
    //     DenseMatrix<5,5> R;
    //     // Scalar beta = 1.0 / (2 * a2);

    //     Scalar nx = normal[0], ny = normal[1], nz = normal[2];
    //     Scalar qn = un;

    //     R = {
    //         1.0,        0.0,         0.0,        0.0,       1.0,
    //         uT - a*nx,  ny,          nz,         0.0,       uT + a*nx,
    //         vT - a*ny,  0.0,         nx,         0.0,       vT + a*ny,
    //         wT - a*nz,  0.0,         0.0,        nx,        wT + a*nz,
    //         HT - a*qn,  uT*ny - a*ny, uT*nz - a*nz, vT*nx - a*nx, HT + a*qn
    //     };

    //     // 计算 L（左特征矩阵） = R^{-1}
    //     DenseMatrix<5,5> L = R.inverse();
    //     // printf("%lf,  %lf,  %lf,  %lf\n",L.multiply(R).trace(),R.multiply(L).trace(),L.multiply(R).norm(),R.multiply(L).norm());

    //     // 计算对角矩阵 Lambda
    //     DenseMatrix<5,5> Lambda{
    //         std::fabs(qn - a), 0, 0, 0, 0,
    //         0, std::fabs(qn), 0, 0, 0,
    //         0, 0, std::fabs(qn), 0, 0,
    //         0, 0, 0, std::fabs(qn), 0,
    //         0, 0, 0, 0, std::fabs(qn + a)
    //     };

    //     // 构造耗散项: R * Lambda * L * deltaU
    //     DenseMatrix<5,1> alpha = L.multiply(deltaU); // L * deltaU
    //     DenseMatrix<5,1> diss;
    //     for (int i = 0; i < 5; ++i)
    //         diss[i] = 0.0;
    //     for (int k = 0; k < 5; ++k) {
    //         Scalar coeff = Lambda(k,k) * alpha[k];
    //         for (int i = 0; i < 5; ++i) {
    //             diss[i] += coeff * R(i,k);
    //         }
    //     }

    //     auto FL = computeFlux(UL).multiply(normal);
    //     auto FR = computeFlux(UR).multiply(normal);

    //     DenseMatrix<5,1> flux;
    //     for (int i = 0; i < 5; ++i)
    //         flux[i] = 0.5 * (FL[i] + FR[i] - diss[i]);
    //     return flux;
    // }

    HostDevice static 
    DenseMatrix<5, 1> computeRoeFlux(
        const DenseMatrix<5, 1>& UL,
        const DenseMatrix<5, 1>& UR,
        const DenseMatrix<3, 1>& normal)
    {
        Scalar gamma = get_gamma();
        // 1. 构造旋转矩阵 Q 和 Q^{-1} (即 Q^T)
        const DenseMatrix<3, 3>& Q = computeRotationMatrix(normal);
        const DenseMatrix<3, 3>& Q_inv = Q.transpose();

        // 2. 计算 Q * U_L 和 Q * U_R (局部坐标系下的守恒变量)
        DenseMatrix<5, 1> QUL, QUR;
        QUL[0] = UL[0]; 
        QUR[0] = UR[0]; 
        // printf("%lf,  %lf,  %lf,  %lf,  %lf,  %lf,  %lf,  %lf,  %lf,  %lf,  %lf,  %lf\n",UL[0],UR[0],QUL[0],QUR[0],UL[1],UR[1],UL[2],UR[2],UL[3],UR[3],UL[4],UR[4]);

        // 旋转动量
        QUL[1] = Q(0,0)*UL[1] + Q(0,1)*UL[2] + Q(0,2)*UL[3]; // rho*u_n
        QUL[2] = Q(1,0)*UL[1] + Q(1,1)*UL[2] + Q(1,2)*UL[3]; // rho*u_t1
        QUL[3] = Q(2,0)*UL[1] + Q(2,1)*UL[2] + Q(2,2)*UL[3]; // rho*u_t2
        QUL[4] = UL[4]; 

        QUR[1] = Q(0,0)*UR[1] + Q(0,1)*UR[2] + Q(0,2)*UR[3]; 
        QUR[2] = Q(1,0)*UR[1] + Q(1,1)*UR[2] + Q(1,2)*UR[3]; 
        QUR[3] = Q(2,0)*UR[1] + Q(2,1)*UR[2] + Q(2,2)*UR[3]; 
        QUR[4] = UR[4]; 

        // 3. 在局部坐标系下计算一维 Roe 通量 F'
        Scalar rhoL = positive(QUL[0]), rhoR = positive(QUR[0]);
        Scalar uL = QUL[1]/rhoL, vL = QUL[2]/rhoL, wL = QUL[3]/rhoL;
        Scalar uR = QUR[1]/rhoR, vR = QUR[2]/rhoR, wR = QUR[3]/rhoR;
        Scalar pL = computePressure(QUL), pR = computePressure(QUR);
        Scalar HL = (QUL[4] + pL) / rhoL;
        Scalar HR = (QUR[4] + pR) / rhoR;

        // 计算 Roe 平均量
        Scalar sqrtRhoL = std::sqrt(rhoL), sqrtRhoR = std::sqrt(rhoR);
        Scalar denom = sqrtRhoL + sqrtRhoR;
        Scalar uT = (sqrtRhoL*uL + sqrtRhoR*uR)/denom;
        Scalar vT = (sqrtRhoL*vL + sqrtRhoR*vR)/denom;
        Scalar wT = (sqrtRhoL*wL + sqrtRhoR*wR)/denom;
        Scalar HT = (sqrtRhoL*HL + sqrtRhoR*HR)/denom;
        // printf("%le,  %le, | %le,  %le, | %lf,  %lf, | %lf, | %lf,  %lf,  %lf,  %lf\n",UL[0],UR[0],rhoL,rhoR,sqrtRhoL, sqrtRhoR,denom,uT,vT,wT,HT);
        // 在局部坐标系下，法向就是 x 方向 (1,0,0)
        // 因此，法向速度 un = uT, 法向速度平方 qn = uT*uT
        Scalar un = uT;
        Scalar V2 = uT*uT + vT*vT + wT*wT;
        Scalar a2 = positive((gamma - 1.0) * (HT - 0.5 * V2));
        Scalar a = std::sqrt(a2);

        // 构造右特征矩阵 R (5x5)
        // 在局部坐标系下，法向为 x 轴，切向为 y,z 轴
        DenseMatrix<5,5> R{
            1.0,         0.0,         0.0,        1.0,       1.0,
            uT - a,      0.0,         0.0,        uT,       uT + a,
            vT,          1,          0.0,        vT,       vT,
            wT,          0.0,         1,         wT,       wT,
            HT - a*un,   vT*1,       wT*1,      0.5*V2,        HT + a*un
        };
        // printf("%lf,  %lf,  %lf,  %lf,  %lf\n%lf,  %lf,  %lf,  %lf,  %lf\n%lf,  %lf,  %lf,  %lf,  %lf\n%lf,  %lf,  %lf,  %lf,  %lf\n%lf,  %lf,  %lf,  %lf,  %lf\n",
        //     R(0,0), R(0,1), R(0,2), R(0,3), R(0,4), R(1,0), R(1,1), R(1,2), R(1,3), R(1,4), R(2,0), R(2,1), R(2,2), R(2,3), R(2,4), R(3,0), R(3,1), R(3,2), R(3,3), R(3,4), R(4,0), R(4,1), R(4,2), R(4,3), R(4,4));


        // 计算左特征矩阵 L = R^{-1}
        // const auto& lu_result = R.lu();
        // printf("%lf,  %lf,  %lf,  %lf,  %lf\n%lf,  %lf,  %lf,  %lf,  %lf\n%lf,  %lf,  %lf,  %lf,  %lf\n%lf,  %lf,  %lf,  %lf,  %lf\n%lf,  %lf,  %lf,  %lf,  %lf\n",
        //     lu_result.LU(0,0), lu_result.LU(0,1), lu_result.LU(0,2), lu_result.LU(0,3), lu_result.LU(0,4), lu_result.LU(1,0), lu_result.LU(1,1), lu_result.LU(1,2), lu_result.LU(1,3), lu_result.LU(1,4), lu_result.LU(2,0), lu_result.LU(2,1), lu_result.LU(2,2), lu_result.LU(2,3), lu_result.LU(2,4), lu_result.LU(3,0), lu_result.LU(3,1), lu_result.LU(3,2), lu_result.LU(3,3), lu_result.LU(3,4), lu_result.LU(4,0), lu_result.LU(4,1), lu_result.LU(4,2), lu_result.LU(4,3), lu_result.LU(4,4));
        const auto& L_delta_U = R.solve(QUR - QUL);
        // printf("%lf,  %lf,  %lf,  %lf,  %lf\n",L_delta_U[0],L_delta_U[1],L_delta_U[2],L_delta_U[3],L_delta_U[4]);

        // 构造对角矩阵 Lambda (特征值的绝对值)
        // DenseMatrix<5,5> Lambda{
        //     std::fabs(un - a), 0, 0, 0, 0,
        //     0, std::fabs(un), 0, 0, 0,
        //     0, 0, std::fabs(un), 0, 0,
        //     0, 0, 0, std::fabs(un), 0,
        //     0, 0, 0, 0, std::fabs(un + a)
        // };
        const DenseMatrix<5,1> Lambda = {std::fabs(un - a), std::fabs(un), std::fabs(un), std::fabs(un), std::fabs(un + a)};

        // 计算局部通量 F_prime = 0.5*(F(QUL) + F(QUR)) - 0.5*dissipation
        DenseMatrix<5, 1> F_prime_L = computeFlux(QUL).multiply(DenseMatrix<3,1>{1.0, 0.0, 0.0}); // x方向通量
        DenseMatrix<5, 1> F_prime_R = computeFlux(QUR).multiply(DenseMatrix<3,1>{1.0, 0.0, 0.0});
        DenseMatrix<5, 1> F_prime = 0.5 * (F_prime_L + F_prime_R) - 0.5 * R.multiply(Lambda * L_delta_U);

        // 4. 计算 Q^{-1} * F_prime (将局部通量变换回全局坐标系)
        DenseMatrix<5, 1> global_flux;
        global_flux[0] = F_prime[0]; 
        global_flux[1] = Q(0,0)*F_prime[1] + Q(1,0)*F_prime[2] + Q(2,0)*F_prime[3]; 
        global_flux[2] = Q(0,1)*F_prime[1] + Q(1,1)*F_prime[2] + Q(2,1)*F_prime[3]; 
        global_flux[3] = Q(0,2)*F_prime[1] + Q(1,2)*F_prime[2] + Q(2,2)*F_prime[3]; 
        global_flux[4] = F_prime[4]; 

        return global_flux;
    }

    HostDevice static 
    DenseMatrix<5, 1> computeHLLFlux(
        const DenseMatrix<5, 1>& UL,
        const DenseMatrix<5, 1>& UR,
        const DenseMatrix<3, 1>& normal)
    {
        Scalar gamma = get_gamma();
        Scalar aL    = soundSpeed(UL),        aR    = soundSpeed(UR);
        Scalar rhoL  = positive(UL[0]),       rhoR  = positive(UR[0]);

        Scalar uL    = UL[1] / rhoL, vL = UL[2] / rhoL, wL = UL[3] / rhoL;
        Scalar uR    = UR[1] / rhoR, vR = UR[2] / rhoR, wR = UR[3] / rhoR;
        Scalar unL   = uL * normal[0] + vL * normal[1] + wL * normal[2];
        Scalar unR   = uR * normal[0] + vR * normal[1] + wR * normal[2];
        
        Scalar SL    = std::min(unL - aL, unR - aR);
        Scalar SR    = std::max(unL + aL, unR + aR);

        auto FL = computeFlux(UL).multiply(normal);
        auto FR = computeFlux(UR).multiply(normal);

        // 等价于按 SL  SR 的条件分支
        Scalar SL0 = std::min(0.0,SL),   SR0 = std::max(0.0,SR);
        DenseMatrix<5,1> HLL_flux = (SR0 * FL - SL0 * FR + SL0 * SR0 * (UR - UL)) / (SR0 - SL0);
        return HLL_flux;
    }

    // HostDevice static 
    // DenseMatrix<5, 1> computeHLLCFlux(
    //     const DenseMatrix<5, 1>& UL,
    //     const DenseMatrix<5, 1>& UR,
    //     const DenseMatrix<3, 1>& normal)
    // {
    //     Scalar gamma = get_gamma();
    //     Scalar aL    = soundSpeed(UL),        aR    = soundSpeed(UR);
    //     Scalar rhoL  = positive(UL[0]),       rhoR  = positive(UR[0]);
    //     Scalar pL    = computePressure(UL),   pR    = computePressure(UR);

    //     Scalar uL    = UL[1] / rhoL, vL = UL[2] / rhoL, wL = UL[3] / rhoL;
    //     Scalar uR    = UR[1] / rhoR, vR = UR[2] / rhoR, wR = UR[3] / rhoR;
    //     Scalar unL   = uL * normal[0] + vL * normal[1] + wL * normal[2];
    //     Scalar unR   = uR * normal[0] + vR * normal[1] + wR * normal[2];
        
    //     Scalar SL    = std::min(unL - aL, unR - aR);
    //     Scalar SR    = std::max(unL + aL, unR + aR);
    //     Scalar SM = (pR - pL + rhoL * unL * (SL - unL) - rhoR * unR * (SR - unR)) /
    //                 (rhoL * (SL - unL) - rhoR * (SR - unR));

    //     auto FL = computeFlux(UL).multiply(normal);
    //     auto FR = computeFlux(UR).multiply(normal);

    //     DenseMatrix<5, 1> UstarL, UstarR;

    //     Scalar coefL = rhoL * (SL - unL) / (SL - SM);
    //     Scalar coefR = rhoR * (SR - unR) / (SR - SM);

    //     UstarL[0] = coefL;
    //     UstarR[0] = coefR;

    //     for (int i = 0; i < 3; ++i) {
    //         Scalar viL = UL[i + 1] / rhoL;
    //         Scalar viR = UR[i + 1] / rhoR;
    //         UstarL[i + 1] = coefL * (viL + (SM - unL) * normal[i]);
    //         UstarR[i + 1] = coefR * (viR + (SM - unR) * normal[i]);
    //     }

    //     UstarL[4] = coefL * (UL[4] / rhoL + (SM - unL) * (SM + pL / (rhoL * (SL - unL))));
    //     UstarR[4] = coefR * (UR[4] / rhoR + (SM - unR) * (SM + pR / (rhoR * (SR - unR))));

    //     if (0.0 <= SL) return FL;
    //     if (SL <= 0.0 && 0.0 <= SM) return FL + SL * (UstarL - UL);
    //     if (SM <= 0.0 && 0.0 <= SR) return FR + SR * (UstarR - UR);
    //     return FR;
    // }

    HostDevice static
    DenseMatrix<3,3> computeRotationMatrix(const DenseMatrix<3,1>& normal){
        // 1. 构造旋转矩阵 Q 和 Q^{-1} (即 Q^T)
        // 构造一个右手正交基: n, t1, t2
        DenseMatrix<3, 3> Q; // 3x3 旋转矩阵

        // 第一行 (法向): 直接使用归一化的面法向
        // 注意：假设传入的 normal 已经是单位向量，否则需要归一化
        Q[0] = normal[0]; Q[1] = normal[1]; Q[2] = normal[2]; 
        // Q_inv[0] = normal[0]; Q_inv[3] = normal[1]; Q_inv[6] = normal[2]; 

        // 构造第一个切向向量 t1 (第二行)
        // 使用法向与一个参考向量（如 (1,0,0)）的叉乘
        // 为了避免与参考向量平行，这里使用 (0,1,0) 作为参考
        DenseMatrix<3,1> ref_vec = {0.0, 0.0, 1.0};
        DenseMatrix<3,1> t1_vec = {0.0, 0.0, 0.0};
        t1_vec[0] = normal[1]*ref_vec[2] - normal[2]*ref_vec[1]; 
        t1_vec[1] = normal[2]*ref_vec[0] - normal[0]*ref_vec[2]; 
        t1_vec[2] = normal[0]*ref_vec[1] - normal[1]*ref_vec[0]; 
        // 归一化
        Scalar t1_norm = std::sqrt(t1_vec[0]*t1_vec[0] + t1_vec[1]*t1_vec[1] + t1_vec[2]*t1_vec[2]);
        if (t1_norm > 0.5) {
            t1_vec[0] /= t1_norm; t1_vec[1] /= t1_norm; t1_vec[2] /= t1_norm;
        } else {
            // 如果叉乘为零，则法向与参考向量平行，需要选择另一个参考向量
            ref_vec = {1.0, 0.0, 0.0};
            t1_vec[0] = normal[1]*ref_vec[2] - normal[2]*ref_vec[1];
            t1_vec[1] = normal[2]*ref_vec[0] - normal[0]*ref_vec[2];
            t1_vec[2] = normal[0]*ref_vec[1] - normal[1]*ref_vec[0];
            t1_norm = std::sqrt(t1_vec[0]*t1_vec[0] + t1_vec[1]*t1_vec[1] + t1_vec[2]*t1_vec[2]);
            if (t1_norm > get_epslion()) {
                t1_vec[0] /= t1_norm; t1_vec[1] /= t1_norm; t1_vec[2] /= t1_norm;
            } else {
                // 极端情况，依然为 0，则使用 (0,0,1) 作为参考
                t1_vec = {0.0, 1.0, 0.0};
            }
        }
        Q[3] = t1_vec[0]; Q[4] = t1_vec[1]; Q[5] = t1_vec[2];

        // 构造第二个切向向量 t2 (第三行): n × t1
        DenseMatrix<3,1> t2_vec = {0.0, 0.0, 0.0};
        t2_vec[0] = normal[1]*t1_vec[2] - normal[2]*t1_vec[1];
        t2_vec[1] = normal[2]*t1_vec[0] - normal[0]*t1_vec[2];
        t2_vec[2] = normal[0]*t1_vec[1] - normal[1]*t1_vec[0];
        // t2_vec 已经是单位向量且正交
        Q[6] = t2_vec[0]; Q[7] = t2_vec[1]; Q[8] = t2_vec[2];
        return Q;
    }

    HostDevice static 
    DenseMatrix<5, 1> computeStabilizationViscosity(
        const DenseMatrix<5, 1>& QUL,
        const DenseMatrix<5, 1>& QUR,
        Scalar gamma)
    {
        Scalar rhoL = positive(QUL[0]), rhoR = positive(QUR[0]);
        Scalar uL = QUL[1] / rhoL, vL = QUL[2] / rhoL, wL = QUL[3] / rhoL;
        Scalar uR = QUR[1] / rhoR, vR = QUR[2] / rhoR, wR = QUR[3] / rhoR;
        Scalar pL = computePressure(QUL), pR = computePressure(QUR);
        Scalar HL = (QUL[4] + pL) / rhoL, HR = (QUR[4] + pR) / rhoR; // 总焓
        Scalar aL = std::sqrt(std::max(0.0, (gamma - 1.0) * (HL - 0.5*(uL*uL + vL*vL + wL*wL))));
        Scalar aR = std::sqrt(std::max(0.0, (gamma - 1.0) * (HR - 0.5*(uR*uR + vR*vR + wR*wR))));

        Scalar sqrt_rhoL = std::sqrt(rhoL);
        Scalar sqrt_rhoR = std::sqrt(rhoR);
        Scalar inv_sum_sqrt_rho = 1.0 / (sqrt_rhoL + sqrt_rhoR);

        Scalar rho_tilde = sqrt_rhoL * sqrt_rhoR;
        Scalar u_tilde = (sqrt_rhoL * uL + sqrt_rhoR * uR) * inv_sum_sqrt_rho;
        Scalar v_tilde = (sqrt_rhoL * vL + sqrt_rhoR * vR) * inv_sum_sqrt_rho;
        Scalar w_tilde = (sqrt_rhoL * wL + sqrt_rhoR * wR) * inv_sum_sqrt_rho;
        Scalar H_tilde = (sqrt_rhoL * HL + sqrt_rhoR * HR) * inv_sum_sqrt_rho;
        Scalar a_tilde = std::sqrt(std::max(0.0, (gamma - 1.0) * (H_tilde - 0.5*(u_tilde*u_tilde + v_tilde*v_tilde + w_tilde*w_tilde))));

        Scalar delta_rho = QUR[0] - QUL[0];
        Scalar delta_p   = pR - pL;
        Scalar delta_v   = vR - vL; 
        Scalar delta_w   = wR - wL; 

        // 耗散控制
        Scalar h = std::min(pL / pR, pR / pL); 
        h = std::max(h, 0.0); // 防止负数
        Scalar g = (1.0 + std::cos(M_PI * h)) * 0.5; 
        Scalar g = std::max(g,0.5);

        Scalar Mach_L = uL / aL; // 左侧单元马赫数
        Scalar Mach_R = uR / aR; // 右侧单元马赫数
        Scalar Mach_Tilde = u_tilde / a_tilde; // Roe平均马赫数
        Scalar phi = std::max({0.0, 1.0 - std::abs(Mach_Tilde), 1.0 - std::abs(Mach_L), 1.0 - std::abs(Mach_R)});

        Scalar coeff_EV  = -g * phi * 0.5 * a_tilde * (delta_rho - delta_p / (a_tilde * a_tilde));
        Scalar coeff_SVy = -g * phi * 0.5 * rho_tilde * a_tilde * delta_v;
        Scalar coeff_SVz = -g * phi * 0.5 * rho_tilde * a_tilde * delta_w;

        DenseMatrix<5, 1> total_viscosity;
        total_viscosity[0] = coeff_EV * 1.0;
        total_viscosity[1] = coeff_EV * u_tilde;
        total_viscosity[2] = coeff_EV * v_tilde + coeff_SVy * 1.0;
        total_viscosity[3] = coeff_EV * w_tilde + coeff_SVz * 1.0;
        total_viscosity[4] = coeff_EV * (0.5*(u_tilde*u_tilde + v_tilde*v_tilde + w_tilde*w_tilde)) 
                        + coeff_SVy * v_tilde + coeff_SVz * w_tilde;

        return total_viscosity;
    }
    HostDevice static 
    DenseMatrix<5, 1> computeHLLCFlux(
        const DenseMatrix<5, 1>& UL,
        const DenseMatrix<5, 1>& UR,
        const DenseMatrix<3, 1>& normal)
    {
        Scalar gamma = get_gamma();
        // 旋转矩阵是正交矩阵，所以逆矩阵就是转置
        const DenseMatrix<3, 3>& Q = computeRotationMatrix(normal);
        // const DenseMatrix<3, 3>& Q_inv = Q.transpose();

        // 只需要转动量？
        DenseMatrix<5, 1> QUL{UL[0], 
                              Q(0,0)*UL[1] + Q(0,1)*UL[2] + Q(0,2)*UL[3],
                              Q(1,0)*UL[1] + Q(1,1)*UL[2] + Q(1,2)*UL[3],
                              Q(2,0)*UL[1] + Q(2,1)*UL[2] + Q(2,2)*UL[3],
                              UL[4]};
        DenseMatrix<5, 1> QUR{UR[0],
                              Q(0,0)*UR[1] + Q(0,1)*UR[2] + Q(0,2)*UR[3],
                              Q(1,0)*UR[1] + Q(1,1)*UR[2] + Q(1,2)*UR[3],
                              Q(2,0)*UR[1] + Q(2,1)*UR[2] + Q(2,2)*UR[3],
                              UR[4]};
        // QUL[0] = UL[0]; // 密度不变
        // QUR[0] = UR[0]; // 密度不变

        // // 旋转动量: (Q * momentum)
        // QUL[1] = Q(0,0)*UL[1] + Q(0,1)*UL[2] + Q(0,2)*UL[3]; // rho*u_n
        // QUL[2] = Q(1,0)*UL[1] + Q(1,1)*UL[2] + Q(1,2)*UL[3]; // rho*u_t1
        // QUL[3] = Q(2,0)*UL[1] + Q(2,1)*UL[2] + Q(2,2)*UL[3]; // rho*u_t2
        // QUL[4] = UL[4]; // 能量不变

        // QUR[1] = Q(0,0)*UR[1] + Q(0,1)*UR[2] + Q(0,2)*UR[3]; // rho*u_n
        // QUR[2] = Q(1,0)*UR[1] + Q(1,1)*UR[2] + Q(1,2)*UR[3]; // rho*u_t1
        // QUR[3] = Q(2,0)*UR[1] + Q(2,1)*UR[2] + Q(2,2)*UR[3]; // rho*u_t2
        // QUR[4] = UR[4]; // 能量不变

        // 3. 在局部坐标系下计算一维 HLLC 通量 F'
        // 此时，局部坐标系的 x 轴就是法向，所以 unL, unR 就是 QUL[1]/QUL[0], QUR[1]/QUR[0]
        Scalar aL    = soundSpeed(QUL),        aR    = soundSpeed(QUR);
        Scalar rhoL  = positive(QUL[0]),       rhoR  = positive(QUR[0]);
        Scalar pL    = computePressure(QUL),   pR    = computePressure(QUR);

        Scalar uL    = QUL[1] / rhoL, vL = QUL[2] / rhoL, wL = QUL[3] / rhoL, eL = QUL[4] / rhoL;
        Scalar uR    = QUR[1] / rhoR, vR = QUR[2] / rhoR, wR = QUR[3] / rhoR, eR = QUR[4] / rhoR;

        Scalar SL    = std::min(uL - aL, uR - aR);
        Scalar SR    = std::max(uL + aL, uR + aR);
        Scalar SM = (pR - pL + rhoL * uL * (SL - uL) - rhoR * uR * (SR - uR)) /
                    (rhoL * (SL - uL) - rhoR * (SR - uR));

        // 计算局部坐标系下的左右通量
        DenseMatrix<5, 3> F_local_L = computeFlux(QUL); // 5x3 矩阵
        DenseMatrix<5, 3> F_local_R = computeFlux(QUR); // 5x3 矩阵
        // 取第一个方向 (x方向，即法向) 的通量
        DenseMatrix<5, 1> F_prime_L, F_prime_R;
        for (int i = 0; i < 5; ++i) {
            F_prime_L[i] = F_local_L[i*3 + 0]; // Flux in x-dir (normal dir)
            F_prime_R[i] = F_local_R[i*3 + 0]; // Flux in x-dir (normal dir)
        }

        DenseMatrix<5, 1> UstarL_prime, UstarR_prime; // 局部坐标系下的中间状态

        Scalar coefL = rhoL * (SL - uL) / (SL - SM);
        Scalar coefR = rhoR * (SR - uR) / (SR - SM);

        UstarL_prime[0] = coefL;
        UstarR_prime[0] = coefR;

        // 在局部坐标系下，切向速度就是 QUL[2]/rhoL, QUL[3]/rhoL
        UstarL_prime[1] = coefL * SM; // 法向动量
        UstarL_prime[2] = coefL * vL; // 切向动量 1
        UstarL_prime[3] = coefL * wL; // 切向动量 2
        UstarL_prime[4] = coefL * (eL + (SM - uL) * (SM + pL / (rhoL * (SL - uL))));

        UstarR_prime[1] = coefR * SM; // 法向动量
        UstarR_prime[2] = coefR * vR; // 切向动量 1
        UstarR_prime[3] = coefR * wR; // 切向动量 2
        UstarR_prime[4] = coefR * (eR + (SM - uR) * (SM + pR / (rhoR * (SR - uR))));

        // 计算局部坐标系下的 HLLC 通量 F'
        DenseMatrix<5, 1> F_prime;
        if (SM >= 0.0) {
            F_prime = (SL >= 0.0) ? F_prime_L : F_prime_L + SL * (UstarL_prime - QUL);
        }
        else{
            F_prime = (SR <= 0.0) ? F_prime_R : F_prime_R + SR * (UstarR_prime - QUR);
        }

        // 调用函数计算总粘性项，将粘性项加到原始通量上
        F_prime = F_prime + computeStabilizationViscosity(QUL, QUR, gamma);

        // 4. 计算 Q^{-1} * F_prime (将局部通量变换回全局坐标系)
        // 只有通量的动量部分 (索引 1,2,3) 需要变换
        DenseMatrix<5, 1> global_flux{F_prime[0],
                                      Q(0,0)*F_prime[1] + Q(1,0)*F_prime[2] + Q(2,0)*F_prime[3],
                                      Q(0,1)*F_prime[1] + Q(1,1)*F_prime[2] + Q(2,1)*F_prime[3],
                                      Q(0,2)*F_prime[1] + Q(1,2)*F_prime[2] + Q(2,2)*F_prime[3],
                                      F_prime[4]};
        // global_flux[0] = F_prime[0]; // 质量通量不变
        // global_flux[1] = Q(0,0)*F_prime[1] + Q(1,0)*F_prime[2] + Q(2,0)*F_prime[3]; // x-momentum
        // global_flux[2] = Q(0,1)*F_prime[1] + Q(1,1)*F_prime[2] + Q(2,1)*F_prime[3]; // y-momentum
        // global_flux[3] = Q(0,2)*F_prime[1] + Q(1,2)*F_prime[2] + Q(2,2)*F_prime[3]; // z-momentum
        // global_flux[4] = F_prime[4]; // 能量通量不变

        return global_flux;
    }

    HostDevice static 
    DenseMatrix<5, 1> computeHLLEMFlux(
        const DenseMatrix<5, 1>& UL,
        const DenseMatrix<5, 1>& UR,
        const DenseMatrix<3, 1>& normal)
    {
        Scalar gamma = get_gamma();
        Scalar aL    = soundSpeed(UL),        aR    = soundSpeed(UR);
        Scalar rhoL  = positive(UL[0]),       rhoR  = positive(UR[0]);

        Scalar uL    = UL[1] / rhoL, vL = UL[2] / rhoL, wL = UL[3] / rhoL;
        Scalar uR    = UR[1] / rhoR, vR = UR[2] / rhoR, wR = UR[3] / rhoR;
        Scalar unL   = uL * normal[0] + vL * normal[1] + wL * normal[2];
        Scalar unR   = uR * normal[0] + vR * normal[1] + wR * normal[2];
        
        Scalar SL    = std::min(unL - aL, unR - aR);
        Scalar SR    = std::max(unL + aL, unR + aR);

        auto FL = computeFlux(UL).multiply(normal);
        auto FR = computeFlux(UR).multiply(normal);

        Scalar SL0 = std::min(0.0,SL),   SR0 = std::max(0.0,SR);
        DenseMatrix<5,1> HLL_flux = (SR0 * FL - SL0 * FR + SL0 * SR0 * (UR - UL)) / (SR0 - SL0);

        // 全都采用 alpha 
        // DenseMatrix<5,1> alpha = (SL0 * SR0 / (SR0 - SL0)) * DenseMatrix<5,1>::Ones();
        // 限制为只对密度采用 alpha
        DenseMatrix<5,1> alpha{(SL0 * SR0 / (SR0 - SL0)),0.0,0.0,0.0,0.0};
        DenseMatrix<5,1> HLLEM_flux = HLL_flux - alpha * (UR - UL);
        return HLLEM_flux;
    }

private:
    HostDevice static Scalar positive(Scalar val) { 
        return std::max(val, get_epslion()); 
    }

    HostDevice static Scalar enthalpy(const DenseMatrix<5,1>& U) {
        return (U[4] + computePressure(U)) / positive(U[0]);
    }

    HostDevice static Scalar soundSpeed(const DenseMatrix<5,1>& U) {
        Scalar gamma = get_gamma();
        return std::sqrt(gamma * positive(computePressure(U) / positive(U[0])));
    }


    HostDevice static Scalar computePressure(const DenseMatrix<5,1>& U) {
        const Scalar rho = positive(U[0]);
        const Scalar u = U[1]/rho, v = U[2]/rho, w = U[3]/rho, e = U[4]/rho;
        Scalar gamma = get_gamma();
        return (gamma-1)*(rho*e - 0.5*rho*(u*u + v*v + w*w));
    }
};

using MonatomicFluxC = EulerPhysicalFlux<5,3,NumericalFluxType::LF>;  // gamma=5/3
using AirFluxC = EulerPhysicalFlux<7,5,NumericalFluxType::LF>;        // gamma=7/5

#define DEFINE_FLUX_TYPE_ALIAS(NAME,Order) \
using MonatomicFluxC##NAME = EulerPhysicalFlux<5,3,NumericalFluxType::NAME>; \
using AirFluxC##NAME = EulerPhysicalFlux<7,5,NumericalFluxType::NAME>; \
using NAME##53C = EulerPhysicalFlux<5,3,NumericalFluxType::NAME>; \
using NAME##75C = EulerPhysicalFlux<7,5,NumericalFluxType::NAME>;

FOREACH_FLUX_TYPE(DEFINE_FLUX_TYPE_ALIAS,0)

#undef DEFINE_FLUX_TYPE_ALIAS