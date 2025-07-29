#pragma once
#include "base/type.h"
#include "dg/dg_basis/dg_basis.h"


template<uInt Order>
class DGBasisEvaluator2D {
public:
    // 定义基函数的阶数和数量
    static constexpr uInt OrderBasis = Order;
    static constexpr uInt NumBasis = (Order+2)*(Order+1)/2;
    
    // 获取DGBasisEvaluator类的实例
    static const DGBasisEvaluator2D& instance() {
        static DGBasisEvaluator2D inst;
        return inst;
    }

    // 计算基函数在给定点的值
    template<typename Type>
    HostDevice static std::array<Type, NumBasis> eval_all(const Type x, const Type y);
    // 计算基函数在给定点的梯度
    template<typename Type>
    HostDevice static std::array<std::array<Type,2>, NumBasis> grad_all(const Type x, const Type y);

    // 计算函数在给定点的系数
    template<typename Func>
    static auto func2coef(const Func& func){
        using QuadC = typename AutoQuadSelector<OrderBasis, GaussLegendreTri::Auto>::type;
        constexpr auto Qpoints = QuadC::get_points();
        constexpr auto Qweights = QuadC::get_weights();
        using Type = Scalar;
        using ReturnType = decltype(func(Qpoints[0])); // 自动推导返回类型

        std::array<ReturnType, NumBasis> result;

        std::array<std::array<Type, NumBasis>, QuadC::num_points> phi;
        
        for(uInt g=0; g<QuadC::num_points; ++g) {
            const auto& p = Qpoints[g];
            phi[g] = eval_all((Type)p[0], (Type)p[1]);
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
    HostDevice static Type coef2filed(const std::array<Type, NumBasis>& coef, const Type x, const Type y);
    template<typename Type>
    HostDevice static Type coef2filed(const std::array<Type, NumBasis>& coef, const std::array<Type,2>& p);

    // 向量场 的 coef，在 单个点 (x,y,z) 上计算，得到向量
    template<uInt N, typename Type>
    HostDevice static DenseMatrix<N,1> coef2filed(const std::array<DenseMatrix<N,1>, NumBasis>& coef, const Type x, const Type y);
    template<uInt N, typename Type>
    HostDevice static DenseMatrix<N,1> coef2filed(const std::array<DenseMatrix<N,1>, NumBasis>& coef, const std::array<Type,2>& p);

    template<uInt N, typename Type>
    HostDevice static DenseMatrix<N,1> coef2filed(const DenseMatrix<N*NumBasis,1>& coef, const Type x, const Type y);
    template<uInt N, typename Type>
    HostDevice static DenseMatrix<N,1> coef2filed(const DenseMatrix<N*NumBasis,1>& coef, const std::array<Type,2>& p);
    
    // 标量/向量场 的 coef，在 一系列 (x,y,z) 上计算，得到 标量/向量场（一系列标量/向量）
    template<typename Type_coef, typename Type>
    static std::vector<Type_coef> coef2filed(const std::array<Type_coef, NumBasis>& coef, const std::vector<Type>& x, const std::vector<Type>& y);
    template<typename Type_coef, typename Type>
    static std::vector<Type_coef> coef2filed(const std::array<Type_coef, NumBasis>& coef, const std::vector<std::array<Type,2>>& p);
    template<uInt N, typename Type>
    static std::vector<DenseMatrix<N,1>> coef2filed(const DenseMatrix<N*NumBasis,1>& coef, const std::vector<std::array<Type,2>>& p);
    template<uInt N, uInt M, typename Type>
    HostDevice static std::array<DenseMatrix<N,1>,M> coef2filed(const DenseMatrix<N*NumBasis,1>& coef, const std::array<std::array<Type,2>,M>& p);
};


#define explict_template_instantiation(Order) \
extern template class DGBasisEvaluator2D<Order>;\

explict_template_instantiation(0) 
explict_template_instantiation(1) 
explict_template_instantiation(2) 
explict_template_instantiation(3) 
explict_template_instantiation(4) 
explict_template_instantiation(5) 
explict_template_instantiation(6) 
explict_template_instantiation(7) 
explict_template_instantiation(8) 
explict_template_instantiation(9) 
explict_template_instantiation(10)

#undef explict_template_instantiation
