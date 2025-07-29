#pragma once
#include "DG_Basis.h"

template<uInt Order>
class DGBasisEvaluator {
public:
    static constexpr uInt OrderBasis = Order;
    static constexpr uInt NumBasis = (Order+3)*(Order+2)*(Order+1)/6;
    
    static const DGBasisEvaluator& instance() {
    static DGBasisEvaluator inst;
    return inst;
}
    template<typename Type>
    HostDevice static std::array<Type, NumBasis> eval_all(Type x, Type y, Type z) {
        std::array<Type, NumBasis> values{};
        static_for<NumBasis>([&](auto p) {
            constexpr uInt BasisID = decltype(p)::value;
            values[BasisID] = DGBasis<BasisID>::eval(x, y, z);
        });
        return values;
    }
    template<typename Type>
    HostDevice static std::array<std::array<Type,3>, NumBasis> grad_all(Type x, Type y, Type z) {
        std::array<std::array<Type,3>, NumBasis> grads{};
        static_for<NumBasis>([&](auto p) {
            constexpr uInt BasisID = decltype(p)::value;
            grads[BasisID] = DGBasis<BasisID>::grad(x, y, z);
        });
        return grads;
    }

    template<typename Func>
    static auto func2coef(const Func& func){
        using QuadC = typename AutoQuadSelector<OrderBasis, GaussLegendreTet::Auto>::type;
        constexpr auto Qpoints = QuadC::get_points();
        constexpr auto Qweights = QuadC::get_weights();
        using Type = Scalar128;
        using ReturnType = decltype(func(Qpoints[0])); // 自动推导返回类型

        std::array<ReturnType, NumBasis> result;

        std::array<std::array<Type, NumBasis>, QuadC::num_points> phi;
        
        for(uInt g=0; g<QuadC::num_points; ++g) {
            const auto& p = Qpoints[g];
            phi[g] = eval_all((Type)p[0], (Type)p[1], (Type)p[2]);
        }
        ReturnType rhs;
        Type diag;
        
        for(uInt k=0;k<NumBasis;k++){
            diag = 0.0;
            rhs = 0.0;
            for(uInt g=0; g<QuadC::num_points; ++g) {
                const auto& p = Qpoints[g];
                diag += phi[g][k]*phi[g][k] * Qweights[g];
                rhs += phi[g][k]*func(p) * Qweights[g];
            }
            result[k] = rhs/diag;
        }
        return result;
    }

    

    // 标量场 的 coef，在 单个点 (x,y,z) 上计算，得到标量
    template<typename Type>
    HostDevice static Type coef2filed(const std::array<Type, NumBasis>& coef, Type x, Type y, Type z){
        const auto& basis = eval_all(x,y,z);
        return vec_dot(basis,coef);
    }
    template<typename Type>
    HostDevice static Type coef2filed(const std::array<Type, NumBasis>& coef, vector3f p){
        return coef2filed(coef, (Type)p[0], (Type)p[1], (Type)p[2]);
    }

    // 向量场 的 coef，在 单个点 (x,y,z) 上计算，得到向量
    template<uInt N, typename Type>
    HostDevice static DenseMatrix<N,1> coef2filed(const std::array<DenseMatrix<N,1>, NumBasis>& coef, Type x, Type y, Type z){
        DenseMatrix<N,1> ret = 0.0;
        const auto& basis = eval_all(x,y,z);

        for(uInt id=0; id<NumBasis; id++){
            ret += basis[id]*coef[id];
        }
        return ret;
    }
    template<uInt N, typename Type>
    HostDevice static DenseMatrix<N,1> coef2filed(const std::array<DenseMatrix<N,1>, NumBasis>& coef, vector3f p){
        return coef2filed(coef, (Type)p[0], (Type)p[1], (Type)p[2]);
    }

    template<uInt N, typename Type>
    HostDevice static DenseMatrix<N,1> coef2filed(const DenseMatrix<N*NumBasis,1>& coef, Type x, Type y, Type z){
        DenseMatrix<N,1> ret = DenseMatrix<N,1>::Zeros();
        const auto& basis = eval_all(x,y,z);

        for(uInt id=0; id<NumBasis; id++){
            const DenseMatrix<N,1>& block = MatrixView<N*NumBasis,1,N,1>::Sub(coef,N*id,0);
            ret += basis[id] * block;
        }
        return ret;
    }
    template<uInt N, typename Type>
    HostDevice static DenseMatrix<N,1> coef2filed(const DenseMatrix<N*NumBasis,1>& coef, vector3f p){
        return coef2filed<N,Type>(coef, (Type)p[0], (Type)p[1], (Type)p[2]);
    }
    
    // 标量/向量场 的 coef，在 一系列 (x,y,z) 上计算，得到 标量/向量场（一系列标量/向量）
    template<typename Type_coef, typename Type>
    HostDevice static std::vector<Type_coef> coef2filed(const std::array<Type_coef, NumBasis>& coef, const std::vector<Type>& x, const std::vector<Type>& y, const std::vector<Type>& z){
        std::vector<Type_coef> ret(x.size());
        for(uInt id=0; id<x.size(); id++){
            ret[id] = coef2filed(coef,x[id],y[id],z[id]);
        }
        return ret;
    }
    template<typename Type_coef, typename Type>
    HostDevice static std::vector<Type_coef> coef2filed(const std::array<Type_coef, NumBasis>& coef, const std::vector<vector3f>& p){
        std::vector<Type_coef> ret(p.size());
        for(uInt id=0; id<p.size(); id++){
            ret[id] = coef2filed(coef,p[id]);
        }
        return ret;
    }
    template<uInt N, typename Type>
    HostDevice static std::vector<DenseMatrix<N,1>> coef2filed(const DenseMatrix<N*NumBasis,1>& coef, const std::vector<vector3f>& p){
        std::vector<DenseMatrix<N,1>> ret(p.size());
        for(uInt id=0; id<p.size(); id++){
            ret[id] = coef2filed(coef,p[id]);
        }
        return ret;
    }
    template<uInt N, uInt M>
    HostDevice static std::array<DenseMatrix<N,1>,M> coef2filed(const DenseMatrix<N*NumBasis,1>& coef, const std::array<vector3f,M>& p){
        std::array<DenseMatrix<N,1>,M> ret(p.size());
        for(uInt id=0; id<p.size(); id++){
            ret[id] = coef2filed(coef,p[id]);
        }
        return ret;
    }
    
};
