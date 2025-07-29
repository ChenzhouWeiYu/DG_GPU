// include/DG/DG_Basis/DG_Basis.h
#pragma once
#include "base/type.h"
#include "matrix/matrix.h"

// 定义一个模板类DGBasis，用于计算三维 DG 基函数 (Dubiner 基函数)
template<uInt BasisID>
struct DGBasis {
    // 计算基函数在(x, y, z)处的值
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y, Type z);
    // 计算基函数在(x, y, z)处的梯度
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,3> grad(Type x, Type y, Type z);
    // 基函数的阶数
    static constexpr uInt Order = 0;
};

// 定义一个模板类DGBasis2D，用于计算二维DG基函数 (Dubiner 基函数)
template<uInt BasisID>
struct DGBasis2D {
    // 计算基函数在(x, y)处的值
    template<typename Type>
    HostDevice constexpr static Type eval(Type x, Type y);
    // 计算基函数在(x, y)处的梯度
    template<typename Type>
    HostDevice constexpr static std::array<Scalar,2> grad(Type x, Type y);
    // 基函数的阶数
    static constexpr uInt Order = 0;
};

// 包含三维DG基函数的实现文件
#include "dg/dg_basis/dg_basis_func.h" 
// 包含二维DG基函数的实现文件
#include "dg/dg_basis/dg_basis_func_2d.h"





// 定义一个模板函数static_for_impl，用于静态循环
template <uInt... Is, typename F>
HostDevice void static_for_impl(std::index_sequence<Is...>, F&& f) {
    // 对每个索引调用函数f
    (f(std::integral_constant<uInt, Is>{}), ...);
}

// 定义一个模板函数static_for，用于静态循环
template <uInt N, typename F>
HostDevice void static_for(F&& f) {
    // 调用static_for_impl函数
    static_for_impl(std::make_index_sequence<N>{}, std::forward<F>(f));
}

// 三维DG基函数评估器的声明 (eval_all, grad_all, etc.)
// 也支持 coef -> field 的重构，但一般都是程序中写 for 循环进行重构
#include "dg/dg_basis/dg_basis_evaluator.h"
// 二维DG基函数评估器的声明 (eval_all, grad_all, etc.)
// 也支持 coef -> field 的重构，但一般都是程序中写 for 循环进行重构
#include "dg/dg_basis/dg_basis_evaluator_2d.h"


