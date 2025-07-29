#pragma once
#include "base/type.h"
#include "dg/dg_basis/dg_basis.h"
#include "dg/dg_basis/dg_basis_evaluator_2d.h"



// 计算基函数在给定点的值
template<uInt Order>
template<typename Type>
HostDevice std::array<Type, DGBasisEvaluator2D<Order>::NumBasis> DGBasisEvaluator2D<Order>::eval_all(const Type x, const Type y) {
    std::array<Type, NumBasis> values{};
    static_for<NumBasis>([&](auto p) {
        constexpr uInt BasisID = decltype(p)::value;
        values[BasisID] = DGBasis2D<BasisID>::eval(x, y);
    });
    return values;
}
// 计算基函数在给定点的梯度
template<uInt Order>
template<typename Type>
HostDevice std::array<std::array<Type,2>, DGBasisEvaluator2D<Order>::NumBasis> DGBasisEvaluator2D<Order>::grad_all(const Type x, const Type y) {
    std::array<std::array<Type,2>, NumBasis> grads{};
    static_for<NumBasis>([&](auto p) {
        constexpr uInt BasisID = decltype(p)::value;
        grads[BasisID] = DGBasis2D<BasisID>::grad(x, y);
    });
    return grads;
}

// 标量场 的 coef，在 单个点 (x,y) 上计算，得到标量

template<uInt Order>
template<typename Type>
HostDevice Type DGBasisEvaluator2D<Order>::coef2filed(const std::array<Type, NumBasis>& coef, const Type x, const Type y){
    const auto& basis = eval_all(x,y);
    return vec_dot(basis,coef);
}

template<uInt Order>
template<typename Type>
HostDevice Type DGBasisEvaluator2D<Order>::coef2filed(const std::array<Type, NumBasis>& coef, const std::array<Type,2>& p){
    return coef2filed(coef, (Type)p[0], (Type)p[1]);  // 标量版本
}

// 向量场 的 coef，在 单个点 (x,y) 上计算，得到向量
template<uInt Order>
template<uInt N, typename Type>
HostDevice DenseMatrix<N,1> DGBasisEvaluator2D<Order>::coef2filed(const std::array<DenseMatrix<N,1>, NumBasis>& coef, const Type x, const Type y){
    DenseMatrix<N,1> ret = DenseMatrix<N,1>::Zeros();
    const auto& basis = eval_all(x,y);

    for(uInt id=0; id<NumBasis; id++){
        ret += basis[id]*coef[id];
    }
    return ret;
}
template<uInt Order>
template<uInt N, typename Type>
HostDevice DenseMatrix<N,1> DGBasisEvaluator2D<Order>::coef2filed(const std::array<DenseMatrix<N,1>, NumBasis>& coef, const std::array<Type,2>& p){
    return coef2filed(coef, (Type)p[0], (Type)p[1]);  // 向量版本
}

template<uInt Order>
template<uInt N, typename Type>
HostDevice DenseMatrix<N,1> DGBasisEvaluator2D<Order>::coef2filed(const DenseMatrix<N*NumBasis,1>& coef, const Type x, const Type y){
    DenseMatrix<N,1> ret = DenseMatrix<N,1>::Zeros();
    const auto& basis = eval_all(x,y);

    for(uInt id=0; id<NumBasis; id++){
        const DenseMatrix<N,1>& block = MatrixView<N*NumBasis,1,N,1>::Sub(coef,N*id,0);
        ret += basis[id] * block;
    }
    return ret;
}
template<uInt Order>
template<uInt N, typename Type>
HostDevice DenseMatrix<N,1> DGBasisEvaluator2D<Order>::coef2filed(const DenseMatrix<N*NumBasis,1>& coef, const std::array<Type,2>& p){
    return coef2filed<N,Type>(coef, (Type)p[0], (Type)p[1]);
}

// 标量/向量场 的 coef，在 一系列 (x,y) 上计算，得到 标量/向量场（一系列标量/向量）
template<uInt Order>
template<typename Type_coef, typename Type>
std::vector<Type_coef> DGBasisEvaluator2D<Order>::coef2filed(const std::array<Type_coef, NumBasis>& coef, const std::vector<Type>& x, const std::vector<Type>& y){
    std::vector<Type_coef> ret(x.size());
    for(uInt id=0; id<x.size(); id++){
        ret[id] = coef2filed(coef,x[id],y[id]);
    }
    return ret;
}
template<uInt Order>
template<typename Type_coef, typename Type>
std::vector<Type_coef> DGBasisEvaluator2D<Order>::coef2filed(const std::array<Type_coef, NumBasis>& coef, const std::vector<std::array<Type,2>>& p){
    std::vector<Type_coef> ret(p.size());
    for(uInt id=0; id<p.size(); id++){
        ret[id] = coef2filed(coef,p[id]);
    }
    return ret;
}
template<uInt Order>
template<uInt N, typename Type>
std::vector<DenseMatrix<N,1>> DGBasisEvaluator2D<Order>::coef2filed(const DenseMatrix<N*NumBasis,1>& coef, const std::vector<std::array<Type,2>>& p){
    std::vector<DenseMatrix<N,1>> ret(p.size());
    for(uInt id=0; id<p.size(); id++){
        ret[id] = coef2filed<N,Type>(coef,p[id]);
    }
    return ret;
}
template<uInt Order>
template<uInt N, uInt M, typename Type>
HostDevice std::array<DenseMatrix<N,1>,M> DGBasisEvaluator2D<Order>::coef2filed(const DenseMatrix<N*NumBasis,1>& coef, const std::array<std::array<Type,2>,M>& p){
    std::array<DenseMatrix<N,1>,M> ret(p.size());
    for(uInt id=0; id<p.size(); id++){
        ret[id] = coef2filed<N,Type>(coef,p[id]);
    }
    return ret;
}