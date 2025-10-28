// include/DG/DG_Basis/DG_Basis_Evaluator.h
#pragma once
#include "base/type.h"
#include "dg/dg_basis/dg_basis.h"

// 定义一个模板类DGBasisEvaluator，用于计算DG基函数的值和梯度
template<uInt Order>
class DGBasisEvaluator {
public:
    // 定义基函数的阶数和数量
    static constexpr uInt OrderBasis = Order;
    static constexpr uInt NumBasis = (Order+3)*(Order+2)*(Order+1)/6;
    
    // 获取DGBasisEvaluator类的实例
    static const DGBasisEvaluator& instance() {
        static DGBasisEvaluator inst;
        return inst;
    }

    // 计算基函数在给定点的值
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<Type, NumBasis> eval_all(const Type x, const Type y, const Type z) {
        std::array<Type, NumBasis> values{};
        static_for<NumBasis>([&](auto p) {
            constexpr uInt BasisID = decltype(p)::value;
            values[BasisID] = DGBasis<BasisID>::eval(x, y, z);
        });
        return values;
    }
    // 计算基函数在给定点的梯度
    template<typename Type>
    HostDevice constexpr static ForceInline std::array<std::array<Type,3>, NumBasis> grad_all(const Type x, const Type y, const Type z) {
        std::array<std::array<Type,3>, NumBasis> grads{};
        static_for<NumBasis>([&](auto p) {
            constexpr uInt BasisID = decltype(p)::value;
            grads[BasisID] = DGBasis<BasisID>::grad(x, y, z);
        });
        return grads;
    }
    // 计算函数在给定点的系数
    template<typename Func>
    static auto func2coef(const Func& func){
        using QuadC = typename AutoQuadSelector<OrderBasis, GaussLegendreTet::Auto>::type;
        constexpr auto Qpoints = QuadC::get_points();
        constexpr auto Qweights = QuadC::get_weights();
        using Type = Scalar;
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

    template<typename Func>
    static auto func2coef_with_bounds(const Func& func) {
        using QuadC = typename AutoQuadSelector<OrderBasis, GaussLegendreTet::Auto>::type;
        constexpr auto Qpoints = QuadC::get_points();
        // constexpr auto Qweights = QuadC::get_weights();
        using Type = Scalar;
        using ReturnType = decltype(func(Qpoints[0])); // 如 DenseMatrix<NEQN,1>

        // === 步骤 1: 最小二乘计算原始系数 ===
        std::array<ReturnType, NumBasis> coef_raw = func2coef(func);

        // === 步骤 2: 在积分点上计算解析值和多项式值 ===
        std::array<ReturnType, QuadC::num_points+4> f_quad;      // 解析值
        std::array<ReturnType, QuadC::num_points+4> poly_quad;   // 多项式值

        std::array<std::array<Type, NumBasis>, QuadC::num_points+4> phi;
        for (uInt g = 0; g < QuadC::num_points+4; ++g) {
            vector3f p = g < QuadC::num_points ? Qpoints[g] : vector3f{0,0,0};
            if(g >= QuadC::num_points) p[g-QuadC::num_points] = 1.0;

            phi[g] = eval_all((Type)p[0], (Type)p[1], (Type)p[2]);
            f_quad[g] = func(p);
            
            // 计算多项式值
            poly_quad[g] = ReturnType::Zeros(); // 假设有 Zero() 静态函数
            for (uInt k = 0; k < NumBasis; ++k) {
                poly_quad[g] = poly_quad[g] + phi[g][k] * coef_raw[k];
            }
        }

        // === 步骤 3: 计算限制因子 theta ===
        Scalar theta = 1.0;
        constexpr uInt VecSize = ReturnType::Size; // 从 DenseMatrix 获取大小

        for (uInt comp = 0; comp < VecSize; ++comp) {
            // 提取解析值的 min/max
            auto [f_min, f_max] = compute_minmax_component(f_quad, comp);
            // 提取多项式值的 min/max
            auto [poly_min, poly_max] = compute_minmax_component(poly_quad, comp);

            Scalar theta_comp = 1.0;
            if (poly_max > f_max) {
                // 缩放高阶模态使 poly_max = f_max
                Scalar avg = coef_raw[0][comp]; // 常数模 = 单元平均
                if (poly_max - avg > 1e-12) {
                    theta_comp = (f_max - avg) / (poly_max - avg);
                }
            }
            if (poly_min < f_min) {
                Scalar avg = coef_raw[0][comp];
                if (avg - poly_min > 1e-12) {
                    Scalar theta_min = (f_min - avg) / (poly_min - avg);
                    theta_comp = fmin(theta_comp, theta_min);
                }
            }
            theta = fmin(theta, theta_comp);
        }

        // === 步骤 4: 应用限制 ===
        std::array<ReturnType, NumBasis> coef_limited;
        coef_limited[0] = coef_raw[0]; // 常数模不变
        for (uInt k = 1; k < NumBasis; ++k) {
            coef_limited[k] = coef_raw[0] + theta * (coef_raw[k] - coef_raw[0]);
        }
        return coef_limited;
    }
    // 提取 ReturnType (如 DenseMatrix<NEQN,1>) 的第 comp 个分量
    template<typename VecType>
    static inline Scalar get_component(const VecType& v, uInt comp) {
        if constexpr (std::is_same_v<VecType, Scalar>) {
            return v;
        } else {
            return v[comp]; // 假设 DenseMatrix 有 .data 成员
        }
    }

    // 计算数组中第 comp 分量的 min/max
    template<typename VecType, uInt NQuadC>
    static inline std::pair<Scalar, Scalar> compute_minmax_component(
        const std::array<VecType, NQuadC>& arr, uInt comp) {
        Scalar min_val = get_component(arr[0], comp);
        Scalar max_val = min_val;
        for (uInt i = 1; i < NQuadC; ++i) {
            Scalar val = get_component(arr[i], comp);
            min_val = fmin(min_val, val);
            max_val = fmax(max_val, val);
        }
        return {min_val, max_val};
    }

    

    // 标量场 的 coef，在 单个点 (x,y,z) 上计算，得到标量
    template<typename Type>
    HostDevice static Type coef2filed(const std::array<Type, NumBasis>& coef, const Type x, const Type y, const Type z);
    template<typename Type>
    HostDevice static Type coef2filed(const std::array<Type, NumBasis>& coef, const std::array<Type,3>& p);

    // 向量场 的 coef，在 单个点 (x,y,z) 上计算，得到向量
    template<uInt N, typename Type>
    HostDevice static DenseMatrix<N,1> coef2filed(const std::array<DenseMatrix<N,1>, NumBasis>& coef, const Type x, const Type y, const Type z);
    template<uInt N, typename Type>
    HostDevice static DenseMatrix<N,1> coef2filed(const std::array<DenseMatrix<N,1>, NumBasis>& coef, const std::array<Type,3>& p);

    template<uInt N, typename Type>
    HostDevice static DenseMatrix<N,1> coef2filed(const DenseMatrix<N*NumBasis,1>& coef, const Type x, const Type y, const Type z);
    template<uInt N, typename Type>
    HostDevice static DenseMatrix<N,1> coef2filed(const DenseMatrix<N*NumBasis,1>& coef, const std::array<Type,3>& p);
    
    // 标量/向量场 的 coef，在 一系列 (x,y,z) 上计算，得到 标量/向量场（一系列标量/向量）
    template<typename Type_coef, typename Type>
    static std::vector<Type_coef> coef2filed(const std::array<Type_coef, NumBasis>& coef, const std::vector<Type>& x, const std::vector<Type>& y, const std::vector<Type>& z);
    template<typename Type_coef, typename Type>
    static std::vector<Type_coef> coef2filed(const std::array<Type_coef, NumBasis>& coef, const std::vector<std::array<Type,3>>& p);
    template<uInt N, typename Type>
    static std::vector<DenseMatrix<N,1>> coef2filed(const DenseMatrix<N*NumBasis,1>& coef, const std::vector<std::array<Type,3>>& p);
    template<uInt N, uInt M, typename Type>
    HostDevice static std::array<DenseMatrix<N,1>,M> coef2filed(const DenseMatrix<N*NumBasis,1>& coef, const std::array<std::array<Type,3>,M>& p);
};





#define explict_template_instantiation(Order) \
extern template class DGBasisEvaluator<Order>;\

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
