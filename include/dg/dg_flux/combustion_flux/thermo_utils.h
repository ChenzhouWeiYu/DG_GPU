// dg/dg_flux/combustion_flux/thermo_utils.h
#pragma once
#include "base/type.h"
#include "matrix/matrix.h"

class Species {
    // NASAPolynomial :  a1, a2, ..., a7, b1, b2
    using NASAPolynomial = std::array<Scalar, 9>; 
    const Scalar M;              // 摩尔质量 (kg/mol)
    const Scalar h_form;         // 生成焓 (J/kg)
    const std::array<NASAPolynomial, 4> nasa_coeffs;
    const Scalar T_low, T_mid, T_high;

public:
    Species(Scalar M, Scalar h_form,
            const std::array<NASAPolynomial, 4>& coeffs,
            Scalar T_low, Scalar T_mid, Scalar T_high)
        : M(M), h_form(h_form), nasa_coeffs(coeffs),
          T_low(T_low), T_mid(T_mid), T_high(T_high) {}

    HostDevice ForceInline
    inline Scalar get_M() const { return M; }
    HostDevice ForceInline
    inline Scalar get_h_form() const { return h_form; }

    // 选择 NASA 系数段
    HostDevice ForceInline
    inline const NASAPolynomial& select_coeffs(Scalar T) const {
        return (T < T_mid) ?
            ((T < T_low) ? nasa_coeffs[0] : nasa_coeffs[1]) :
            ((T < T_high) ? nasa_coeffs[2] : nasa_coeffs[3]);
    }

    // 计算 c_p(T)/R_u
    HostDevice ForceInline
    inline Scalar compute_cp_over_Ru(Scalar T) const {
        const auto& c = select_coeffs(T);
        Scalar T_inv = 1.0 / T;
        return c[0]*T_inv*T_inv + c[1]*T_inv + c[2] + c[3]*T + c[4]*T*T + c[5]*T*T*T + c[6]*T*T*T*T;
    }

    // 计算 h(T)/R_u （注意：返回的是 h/R_u，单位 J/mol → J/kg 需除以 M）
    HostDevice ForceInline
    inline Scalar compute_h_over_Ru(Scalar T) const {
        const auto& c = select_coeffs(T);
        Scalar T_inv = 1.0 / T;
        Scalar lnT = std::log(T);
        return -c[0]*T_inv*T_inv + c[1]*lnT + c[2] + 0.5*c[3]*T + (1.0/3.0)*c[4]*T*T
               + 0.25*c[5]*T*T*T + 0.2*c[6]*T*T*T*T + c[7]*T_inv;
    }

    // 计算 s(T)/R_u
    HostDevice ForceInline
    inline Scalar compute_s_over_Ru(Scalar T) const {
        const auto& c = select_coeffs(T);
        Scalar lnT = std::log(T);
        return -c[0]/T + c[1]*lnT + c[2]*T + 0.5*c[3]*T*T + (1.0/3.0)*c[4]*T*T*T
               + 0.25*c[5]*T*T*T*T + 0.2*c[6]*T*T*T*T*T + c[8];
    }
};


namespace utils {
    inline Scalar safe_positive(Scalar x) {
        constexpr Scalar eps = std::numeric_limits<Scalar>::epsilon();
        return (x < eps) ? eps : x;
    }
} // namespace utils


// class Species {
//     const Scalar M;
//     const Scalar h_form;
//     const std::array<Scalar, 7> cp_ll;
//     const std::array<Scalar, 7> cp_lh;
//     const std::array<Scalar, 7> cp_hl;
//     const std::array<Scalar, 7> cp_hh;
//     const Scalar T_mid, T_low, T_high;
// public:
//     // 显式构造函数
//     Species(Scalar M, Scalar h_form, std::array<Scalar, 7> cp_ll, std::array<Scalar, 7> cp_lh, 
//         std::array<Scalar, 7> cp_hl, std::array<Scalar, 7> cp_hh, Scalar T_mid, Scalar T_low, Scalar T_high) : 
//         M(M), h_form(h_form), cp_ll(cp_ll), cp_lh(cp_lh), cp_hl(cp_hl), cp_hh(cp_hh), T_mid(T_mid), T_low(T_low), T_high(T_high) {};

//     // 可能需要的接口
//     const Scalar get_M() const { return M; }
//     const Scalar get_h_form() const { return h_form; }
//     const std::array<Scalar, 7>& select_nasa_coeffs(Scalar T) const {
//         return (T < T_mid) ? (T < T_low ? cp_ll : cp_lh) : (T < T_high ? cp_hl : cp_hh);
//     }

//     // 热容 温度积分 焓 等计算
//     const Scalar compute_cp_over_Ru(Scalar T) const {
//         return utils::compute_cp_over_Ru(select_nasa_coeffs(T), T);
//     }
//     const Scalar compute_h_over_Ru(Scalar T) const {
//         return utils::compute_h_over_Ru(select_nasa_coeffs(T), T) + h_form;
//     }
//     const Scalar compute_s_over_Ru(Scalar T) const {
//         return utils::compute_s_over_Ru(select_nasa_coeffs(T), T);
//     }
// };

// namespace utils {
//     inline Scalar safe_positive(Scalar x) {
//         constexpr Scalar eps = std::numeric_limits<Scalar>::epsilon();
//         return (x < eps) ? eps : x;
//     }

//     // inline Scalar compute_cp_over_Ru(const std::array<Scalar, 7>& a, Scalar T) {
//     //     // NASA 7-coefficient polynomial
//     //     // 表达式为 C_p^0(T)/Ru = a[0]/T^2 + a[1]/T + a[2] + a[3]*T + a[4]*T^2 + a[5]*T^3 + a[6]*T^4
//     //     Scalar inv_T = 1.0 / T;
//     //     Scalar T2 = T * T;
//     //     Scalar T3 = T2 * T;
//     //     Scalar T4 = T3 * T;
//     //     return a[0]*inv_T*inv_T + a[1]*inv_T + a[2] + a[3]*T + a[4]*T2 + a[5]*T3 + a[6]*T4;
//     // }

//     // inline Scalar compute_h_over_Ru(const std::array<Scalar, 7>& a, Scalar T) {
//     //     // 这里实现的是
//     //     // h^0(T)/Ru = - a[0]/T + a[1]*ln(T) + a[2]*T + a[3]*T^2/2 + a[4]*T^3/3 + a[5]*T^4/4 + a[6]*T^5/5 + b1
//     //     // 而不是
//     //     // h^0(T)/(RuT) = - a[0]/T^2 + a[1]*ln(T)/T + a[2] + a[3]*T + a[4]*T^2 + a[5]*T^3 + a[6]*T^4 + b1/T
//     //     Scalar inv_T = 1.0 / T;
//     //     Scalar T2 = T * T;
//     //     Scalar T3 = T2 * T;
//     //     Scalar T4 = T3 * T;
//     //     return -a[0]*inv_T + a[1]*std::log(T) + a[2]*T + a[3]*T2/2.0 + a[4]*T3/3.0 + a[5]*T4/4.0 + a[6]*T4*T/5.0;
//     // }

//     // inline Scalar compute_s_over_Ru(const std::array<Scalar, 7>& a, Scalar T) {
//     //     // 这里实现的是
//     //     // s^0(T)/Ru = - a[0]/T^2 - a[1]/T + a[2]*ln(T) + a[3]*T + a[4]*T^2/2 + a[5]*T^3/3 + a[6]*T^4/4 + b2
//     //     Scalar inv_T = 1.0 / T;
//     //     Scalar T2 = T * T;
//     //     Scalar T3 = T2 * T;
//     //     Scalar T4 = T3 * T;
//     //     return -a[0]*inv_T*inv_T - a[1]*inv_T + a[2]*std::log(T) + a[3]*T + a[4]*T2/2.0 + a[5]*T3/3.0 + a[6]*T4/4.0;
//     // }

//     // inline const std::array<Scalar, 7>& select_nasa_coeffs(const Species& spec, Scalar T) {
//     //     return (T < spec.T_mid) ? spec.cp_low : spec.cp_high;
//     // }
// } 