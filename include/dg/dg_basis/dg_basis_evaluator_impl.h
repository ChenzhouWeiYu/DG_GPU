#pragma once
#include "base/type.h"
#include "dg/dg_basis/dg_basis.h"
#include "dg/dg_basis/dg_basis_evaluator.h"



// 计算基函数在给定点的值
template<uInt Order>
template<typename Type>
HostDevice std::array<Type, DGBasisEvaluator<Order>::NumBasis> DGBasisEvaluator<Order>::eval_all(const Type x, const Type y, const Type z) {
    std::array<Type, NumBasis> values{};
    static_for<NumBasis>([&](auto p) {
        constexpr uInt BasisID = decltype(p)::value;
        values[BasisID] = DGBasis<BasisID>::eval(x, y, z);
    });
    return values;
}
// 计算基函数在给定点的梯度
template<uInt Order>
template<typename Type>
HostDevice std::array<std::array<Type,3>, DGBasisEvaluator<Order>::NumBasis> DGBasisEvaluator<Order>::grad_all(const Type x, const Type y, const Type z) {
    std::array<std::array<Type,3>, NumBasis> grads{};
    static_for<NumBasis>([&](auto p) {
        constexpr uInt BasisID = decltype(p)::value;
        grads[BasisID] = DGBasis<BasisID>::grad(x, y, z);
    });
    return grads;
}

// 标量场 的 coef，在 单个点 (x,y,z) 上计算，得到标量

template<uInt Order>
template<typename Type>
HostDevice Type DGBasisEvaluator<Order>::coef2filed(const std::array<Type, NumBasis>& coef, const Type x, const Type y, const Type z){
    const auto& basis = eval_all(x,y,z);
    return vec_dot(basis,coef);
}

template<uInt Order>
template<typename Type>
HostDevice Type DGBasisEvaluator<Order>::coef2filed(const std::array<Type, NumBasis>& coef, const std::array<Type,3>& p){
    return coef2filed(coef, (Type)p[0], (Type)p[1], (Type)p[2]);
}

// 向量场 的 coef，在 单个点 (x,y,z) 上计算，得到向量
template<uInt Order>
template<uInt N, typename Type>
HostDevice DenseMatrix<N,1> DGBasisEvaluator<Order>::coef2filed(const std::array<DenseMatrix<N,1>, NumBasis>& coef, const Type x, const Type y, const Type z){
    DenseMatrix<N,1> ret = DenseMatrix<N,1>::Zeros();
    const auto& basis = eval_all(x,y,z);

    for(uInt id=0; id<NumBasis; id++){
        ret += basis[id]*coef[id];
    }
    return ret;
}
template<uInt Order>
template<uInt N, typename Type>
HostDevice DenseMatrix<N,1> DGBasisEvaluator<Order>::coef2filed(const std::array<DenseMatrix<N,1>, NumBasis>& coef, const std::array<Type,3>& p){
    return coef2filed(coef, (Type)p[0], (Type)p[1], (Type)p[2]);
}

template<uInt Order>
template<uInt N, typename Type>
HostDevice DenseMatrix<N,1> DGBasisEvaluator<Order>::coef2filed(const DenseMatrix<N*NumBasis,1>& coef, const Type x, const Type y, const Type z){
    DenseMatrix<N,1> ret = DenseMatrix<N,1>::Zeros();
    const auto& basis = eval_all(x,y,z);

    for(uInt id=0; id<NumBasis; id++){
        const DenseMatrix<N,1>& block = MatrixView<N*NumBasis,1,N,1>::Sub(coef,N*id,0);
        ret += basis[id] * block;
    }
    return ret;
}
template<uInt Order>
template<uInt N, typename Type>
HostDevice DenseMatrix<N,1> DGBasisEvaluator<Order>::coef2filed(const DenseMatrix<N*NumBasis,1>& coef, const std::array<Type,3>& p){
    return coef2filed<N,Type>(coef, (Type)p[0], (Type)p[1], (Type)p[2]);
}

// 标量/向量场 的 coef，在 一系列 (x,y,z) 上计算，得到 标量/向量场（一系列标量/向量）
template<uInt Order>
template<typename Type_coef, typename Type>
std::vector<Type_coef> DGBasisEvaluator<Order>::coef2filed(const std::array<Type_coef, NumBasis>& coef, const std::vector<Type>& x, const std::vector<Type>& y, const std::vector<Type>& z){
    std::vector<Type_coef> ret(x.size());
    for(uInt id=0; id<x.size(); id++){
        ret[id] = coef2filed(coef,x[id],y[id],z[id]);
    }
    return ret;
}
template<uInt Order>
template<typename Type_coef, typename Type>
std::vector<Type_coef> DGBasisEvaluator<Order>::coef2filed(const std::array<Type_coef, NumBasis>& coef, const std::vector<std::array<Type,3>>& p){
    std::vector<Type_coef> ret(p.size());
    for(uInt id=0; id<p.size(); id++){
        ret[id] = coef2filed(coef,p[id]);
    }
    return ret;
}
template<uInt Order>
template<uInt N, typename Type>
std::vector<DenseMatrix<N,1>> DGBasisEvaluator<Order>::coef2filed(const DenseMatrix<N*NumBasis,1>& coef, const std::vector<std::array<Type,3>>& p){
    std::vector<DenseMatrix<N,1>> ret(p.size());
    for(uInt id=0; id<p.size(); id++){
        ret[id] = coef2filed<N,Type>(coef,p[id]);
    }
    return ret;
}
template<uInt Order>
template<uInt N, uInt M, typename Type>
HostDevice std::array<DenseMatrix<N,1>,M> DGBasisEvaluator<Order>::coef2filed(const DenseMatrix<N*NumBasis,1>& coef, const std::array<std::array<Type,3>,M>& p){
    std::array<DenseMatrix<N,1>,M> ret(p.size());
    for(uInt id=0; id<p.size(); id++){
        ret[id] = coef2filed<N,Type>(coef,p[id]);
    }
    return ret;
}