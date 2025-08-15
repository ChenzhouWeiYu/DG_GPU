// dg/dg_flux/combustion_flux/thermo_dynamics.h
#pragma once
#include "base/type.h"
#include "matrix/matrix.h"
#include "dg/dg_flux/combustion_flux/thermo_utils.h"

template<uInt N>
class Thermodynamics;

// === 特化 1: N = 0 → 单组分，常 gamma ===
template<>
class Thermodynamics<0> {
    Scalar gamma, R;

public:
    explicit Thermodynamics(Scalar gamma_, Scalar R_ = 287.0) : gamma(gamma_), R(R_) {}

    struct Primitive {
        Scalar rho, u, v, w, p, T, a, gamma_eff;
    };

    HostDevice ForceInline
    inline Primitive reconstruct(const DenseMatrix<5, 1>& U) const {
        Scalar rho = utils::safe_positive(U[0]);
        Scalar u = U[1]/rho, v = U[2]/rho, w = U[3]/rho;
        Scalar ke = 0.5*(u*u + v*v + w*w);
        Scalar e = utils::safe_positive(U[4]/rho - ke);
        Scalar p = (gamma - 1) * rho * e;
        Scalar T = p / (rho * R);
        Scalar a = std::sqrt(gamma * p / rho);
        // 更加安全的版本
        // Scalar T = utils::safe_positive(p / (rho * R));
        // Scalar a = std::sqrt(utils::safe_positive(gamma * p / rho));
        return {rho, u, v, w, p, T, a, gamma};
    }
};

// === 特化 2: N >= 2 → 多组分 ===
template<uInt N>
class Thermodynamics {
    std::array<Species, N> species;
    
    // 牛顿法求解 T(e)，固定5步，无分支
    HostDevice ForceInline
    inline Scalar solve_temperature(const std::array<Scalar, N>& Y, Scalar e_target) const {
        // 初值估计：忽略生成焓，用平均 cv
        constexpr Scalar Ru = 8.3144626; 
        Scalar hfo_avg = 0.0, cv_ref = 0.0, R_mix = 0.0;
        PragmaUnroll
        for (uInt i = 0; i < N; ++i) {
            hfo_avg += Y[i] * species[i].get_h_form();
            R_mix    += Y[i] / species[i].get_M();
            cv_ref   += Y[i] * (species[i].compute_cp_over_Ru(1000.0) - 1.0/Ru/1000.0);
        }
        R_mix *= Ru;  // R_u * sum(Y_i / M_i)
        cv_ref *= Ru; // 转为 J/(kg·K)

        Scalar T = utils::safe_positive((e_target - hfo_avg + 1.0e4) / (cv_ref + 1.0e-8));

        // 固定5步牛顿迭代
        for (int k = 0; k < 5; ++k) {
            Scalar e_mix = 0.0, cv_mix = 0.0;
            PragmaUnroll
            for (uInt i = 0; i < N; ++i) {
                Scalar cp_over_Ru = species[i].compute_cp_over_Ru(T);
                Scalar cp = cp_over_Ru * Ru;
                Scalar cv = cp - Ru / species[i].get_M();
                Scalar h_over_Ru = species[i].compute_h_over_Ru(T);
                Scalar h = h_over_Ru * Ru;
                Scalar e_i = h - Ru * T / species[i].get_M();
                e_mix += Y[i] * e_i;
                cv_mix += Y[i] * cv;
            }
            Scalar residual = e_mix - e_target;
            T = utils::safe_positive(T - residual / (cv_mix + 1.0e-8));
        }
        return T;
    }

public:
    explicit Thermodynamics(std::array<Species, N> sp) : species(sp) {}

    struct Primitive {
        Scalar rho, u, v, w, p, T, a, gamma_eff;
        std::array<Scalar, N> Y;
    };

    HostDevice ForceInline
    inline Primitive reconstruct(const DenseMatrix<5 + N, 1>& U) const {
        // Step 1: 提取守恒变量
        constexpr Scalar Ru = 8.3144626; 
        Scalar rho = utils::safe_positive(U[0]);
        Scalar rhou = U[1], rhov = U[2], rhow = U[3];
        Scalar rhoE = U[4];
        std::array<Scalar, N> rhoY;
        PragmaUnroll
        for (uInt i = 0; i < N; ++i) {
            rhoY[i] = U[5 + i];
        }

        // Step 2: 计算速度和动能
        Scalar u = rhou / rho;
        Scalar v = rhov / rho;
        Scalar w = rhow / rho;
        Scalar ke = 0.5 * (u*u + v*v + w*w);

        // Step 3: 计算质量分数并归一化（强制 sum Y_i = 1）
        std::array<Scalar, N> Y;
        Scalar sum_Y = 0.0;
        PragmaUnroll
        for (uInt i = 0; i < N; ++i) {
            Y[i] = utils::safe_positive(rhoY[i] / rho);
            sum_Y += Y[i];
        }
        // 归一化
        PragmaUnroll
        for (uInt i = 0; i < N; ++i) {
            Y[i] /= sum_Y;
        }

        // Step 4: 计算内能
        Scalar e = utils::safe_positive(rhoE / rho - ke);

        // Step 5: 求解温度
        Scalar T = solve_temperature(Y, e);

        // Step 6: 计算混合物属性
        Scalar R_mix = 0.0, cp_mix = 0.0;
        PragmaUnroll
        for (uInt i = 0; i < N; ++i) {
            Scalar M_i = species[i].get_M();
            R_mix += Y[i] / M_i;
            cp_mix += Y[i] * (species[i].compute_cp_over_Ru(T) * Ru);
        }
        R_mix *= Ru;  // R = R_u * sum(Y_i / M_i)
        Scalar cv_mix = cp_mix - R_mix;
        Scalar p = rho * R_mix * T;
        Scalar gamma_eff = cp_mix / cv_mix;
        Scalar a = std::sqrt(utils::safe_positive(gamma_eff * p / rho));

        return {rho, u, v, w, p, T, a, gamma_eff, Y};
    }
};