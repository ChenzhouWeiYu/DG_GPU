#pragma once
#include "base/type.h"
#include "matrix/matrix.h"

template<uInt BasisID>
struct DGBasis {
    static Scalar eval(Scalar x, Scalar y, Scalar z);
    static std::array<Scalar,3> grad(Scalar x, Scalar y, Scalar z);
    static constexpr uInt Order = 0;
};

#include "DG_Basis_Func.h"





template <uInt... Is, typename F>
HostDevice void static_for_impl(std::index_sequence<Is...>, F&& f) {
    (f(std::integral_constant<uInt, Is>{}), ...);
}

template <uInt N, typename F>
HostDevice void static_for(F&& f) {
    static_for_impl(std::make_index_sequence<N>{}, std::forward<F>(f));
}

#include "DG_Basis_Evaluator.h"

// // 计算所有基函数在给定点的值
// auto values = DGBasisEvaluator<3>::eval_all(0.5, 0.5, 0.0);

// // 计算所有基函数在给定点的梯度
// auto grads = DGBasisEvaluator<3>::grad_all(0.5, 0.5, 0.0);

// // 访问第5个基函数的值和梯度
// Scalar val_p5 = values[4]; 
// auto grad_p5 = grads[4];



