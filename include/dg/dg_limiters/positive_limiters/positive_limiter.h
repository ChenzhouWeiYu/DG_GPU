#pragma once
#include "base/type.h"
#include "dg/dg_basis/dg_basis.h"
#include "mesh/mesh.h"
#include "matrix/matrix.h"

/**
 * @brief 先保极值、后保正，两步限制器。
 * @tparam Order 多项式阶数（如 2 表示 P2 空间）。
 * @tparam QuadType 积分公式类型（如 GaussLegendreTet::Degree20Points448）。
 */
template <uInt Order, typename QuadType, bool OnlyNeigbAvg = false>
class PositiveLimiter {
public:
    const ComputingMesh& mesh_;                     // 网格数据
    using Basis = DGBasisEvaluator<Order>;          // 基函数计算器
    LongVector<5> per_cell_max;
    LongVector<5> per_cell_min;
    std::array<std::array<Scalar,Basis::NumBasis>,QuadType::num_points> basis_table;
    Scalar param_gamma = 1.4;

public:
    /**
     * @brief 构造函数，预计算逆 Jacobi 矩阵。
     * @param mesh 计算网格
     * @param basis 基函数工具
     */
    PositiveLimiter(const ComputingMesh& mesh, Scalar gamma = 1.4)
        : mesh_(mesh), param_gamma(gamma) {
            per_cell_max.resize(mesh.m_cells.size());
            per_cell_min.resize(mesh.m_cells.size());
            for(auto&b : per_cell_max.blocks) b=0.0;
            for(auto&b : per_cell_min.blocks) b=0.0;
            constexpr auto Qpoints = QuadType::get_points();
            constexpr auto Qweights = QuadType::get_weights();
            for (uInt xgi = 0; xgi < QuadType::num_points; ++xgi) {
                basis_table[xgi] = Basis::eval_all(Qpoints[xgi][0], Qpoints[xgi][1], Qpoints[xgi][2]);
            }
         }
    /**
     * @brief 应用限制器到解向量。
     * @param current_coeffs 当前时间步的解系数（coef_new）
     * @param previous_coeffs 上一时间步的解系数（coef_old）
     */
    void apply(LongVector<5 * Basis::NumBasis>& current_coeffs, 
               const LongVector<5 * Basis::NumBasis>& previous_coeffs) {
        apply_1(current_coeffs);
        apply_2(current_coeffs);
    }
    void apply(LongVector<5 * Basis::NumBasis>& current_coeffs) {
        apply_1(current_coeffs);
        apply_2(current_coeffs);
    }

    void apply_1(LongVector<5 * Basis::NumBasis>& current_coeffs,
                const LongVector<5 * Basis::NumBasis>& previous_coeffs){
        #pragma omp parallel for schedule(dynamic)
        for (uInt cellId = 0; cellId < mesh_.m_cells.size(); ++cellId) {
            apply_1_limitCell(cellId, current_coeffs);
        }
    }
    void apply_1(LongVector<5 * Basis::NumBasis>& current_coeffs){
        #pragma omp parallel for schedule(dynamic)
        for (uInt cellId = 0; cellId < mesh_.m_cells.size(); ++cellId) {
            apply_1_limitCell(cellId, current_coeffs);
        }
    }

    void apply_2(LongVector<5 * Basis::NumBasis>& current_coeffs,
                const LongVector<5 * Basis::NumBasis>& previous_coeffs){
        #pragma omp parallel for schedule(dynamic)
        for(uInt cellId = 0; cellId < mesh_.m_cells.size(); cellId++){
            apply_21_limitCell(cellId, current_coeffs);
            apply_22_limitCell(cellId, current_coeffs);
        }
    }
    void apply_2(LongVector<5 * Basis::NumBasis>& current_coeffs){
        #pragma omp parallel for schedule(dynamic)
        for(uInt cellId = 0; cellId < mesh_.m_cells.size(); cellId++){
            apply_21_limitCell(cellId, current_coeffs);
            apply_22_limitCell(cellId, current_coeffs);
        }
    }

    void constructMinMax(LongVector<5 * Basis::NumBasis>& current_coeffs,
                const LongVector<5 * Basis::NumBasis>& previous_coeffs){
        constructMinMax(previous_coeffs);
    }
    void constructMinMax(const LongVector<5 * Basis::NumBasis>& previous_coeffs){
        Scalar max_scalar = std::numeric_limits<Scalar>::max();
        // print(per_cell_max.size());
        for(auto&b : per_cell_max.blocks) b=-max_scalar;
        for(auto&b : per_cell_min.blocks) b=max_scalar;
        // print(per_cell_min.size());
        
        constexpr auto Qpoints = QuadType::get_points();
        constexpr auto Qweights = QuadType::get_weights();
        #pragma omp parallel for schedule(dynamic)
        for(uInt fid = 0; fid < mesh_.m_faces.size(); fid++){
            uInt neig_1 = mesh_.m_faces[fid].m_neighbor_cells[0];
            uInt neig_2 = mesh_.m_faces[fid].m_neighbor_cells[1];
            const auto& coef_1 = previous_coeffs[neig_1]; 
            // 判断是否另一单元（边界）
            const auto& coef_2 = (neig_2==uInt(-1)) ? previous_coeffs[neig_1] : previous_coeffs[neig_2]; 
            DenseMatrix<5,1> max_U = DenseMatrix<5,1>::Ones() * -max_scalar;
            DenseMatrix<5,1> min_U = DenseMatrix<5,1>::Ones() * max_scalar;
            if constexpr (OnlyNeigbAvg){
                // Scalar val1 = 0.0, val2 = 0.0;
                for (int k = 0; k < 5; ++k) {
                    max_U[k] = std::max(coef_1[k + 5*0], coef_2[k + 5*0]);
                    min_U[k] = std::min(coef_1[k + 5*0], coef_2[k + 5*0]);
                }
            }
            else{
                for (uInt xgi = 0; xgi < QuadType::num_points; ++xgi) {
                    // const auto& xg = Qpoints[xgi];
                    // const auto& basis = Basis::eval_all(xg[0],xg[1],xg[2]);
                    const auto& basis = basis_table[xgi];
                    // 计算积分点 xg 处，所有基函数组合的完整的解
                    for (int k = 0; k < 5; ++k) {
                        Scalar val1 = 0.0, val2 = 0.0;
                        for (uInt l = 0; l < Basis::NumBasis; ++l) {
                            val1 += basis[l] * coef_1[k + 5*l];
                            val2 += basis[l] * coef_2[k + 5*l];
                        }
                        // 计算 neig_1,2 单元上，5个守恒量的最大值
                        max_U[k] = std::max(max_U[k], std::max(val1,val2));
                        min_U[k] = std::min(max_U[k], std::min(val1,val2));
                    }
                }
            }
            
            // 或者，只用均值？
            // 将最值 写入到两个单元
            #pragma omp critical
            for (int k = 0; k < 5; ++k) {
                per_cell_max[neig_1][k] = std::max(per_cell_max[neig_1][k],max_U[k]);
                per_cell_min[neig_1][k] = std::min(per_cell_min[neig_1][k],min_U[k]);
                if (neig_2!=uInt(-1)){
                    per_cell_max[neig_2][k] = std::max(per_cell_max[neig_2][k],max_U[k]);
                    per_cell_min[neig_2][k] = std::min(per_cell_min[neig_2][k],min_U[k]);
                }
            }
        }
    }
private:
    Scalar compute_pressure(const DenseMatrix<5,1>& U) {
        Scalar rho = std::max(1e-16,U[0]);
        Scalar rhou = U[1];
        Scalar rhov = U[2];
        Scalar rhow = U[3];
        Scalar rhoE = U[4];

        // if (rho <= 1e-16) return -1.0; // 密度为零或负值，压强也应为负

        Scalar ke = (rhou * rhou + rhov * rhov + rhow * rhow) / (2.0 * rho);
        return (param_gamma - 1.0) * (rhoE - ke);
    };


    void apply_1_limitCell(uInt cellId, 
                LongVector<5 * Basis::NumBasis>& current_coeffs) {
        // 获取当前单元的 max/min 范围
        auto global_min = per_cell_min[cellId];
        auto global_max = per_cell_max[cellId];
        constexpr auto Qpoints = QuadType::get_points();
        constexpr auto Qweights = QuadType::get_weights();

        for (int k = 0; k < 5; ++k) {  // 遍历守恒变量
            Scalar avg = current_coeffs[cellId][k];  // 单元平均值

            // 构造当前变量的积分点值分布
            std::array<Scalar, QuadType::num_points> qp_vals;
            qp_vals.fill(0.0);
            for (uInt xgi = 0; xgi < QuadType::num_points; ++xgi) {
                // const auto& basis = Basis::eval_all(Qpoints[xgi][0], Qpoints[xgi][1], Qpoints[xgi][2]);
                const auto& basis = basis_table[xgi];
                Scalar val = 0.0;
                for (uInt l = 0; l < Basis::NumBasis; ++l) {
                    val += basis[l] * current_coeffs[cellId][k + 5*l];
                }
                qp_vals[xgi] = val;
            }

            Scalar val_min = *std::min_element(qp_vals.begin(), qp_vals.end());
            Scalar val_max = *std::max_element(qp_vals.begin(), qp_vals.end());

            if (val_min < global_min[k] || val_max > global_max[k]) {
                Scalar theta_min = (avg != val_min) ? (avg - global_min[k]) / (avg - val_min + 1e-32) : 0.0;
                Scalar theta_max = (avg != val_max) ? (global_max[k] - avg) / (val_max - avg + 1e-32) : 0.0;
                Scalar theta = std::min({theta_min, theta_max, 1.0});

                // 应用于高阶模态系数
                for (uInt l = 1; l < Basis::NumBasis; ++l) {
                    current_coeffs[cellId][k + 5*l] *= theta;
                }
            }
        }
    }
    void apply_21_limitCell(uInt cellId, 
                LongVector<5 * Basis::NumBasis>& current_coeffs) {
        
        constexpr auto Qpoints = QuadType::get_points();
        constexpr auto Qweights = QuadType::get_weights();
        Scalar eps = 1e-16;
        std::array<Scalar, QuadType::num_points> rho_qp;
        rho_qp.fill(0.0);
        for (uInt xgi = 0; xgi < QuadType::num_points; xgi++) {
            // const auto& xg = Qpoints[xgi];
            // const auto& basis = Basis::eval_all(xg[0],xg[1],xg[2]);
            const auto& basis = basis_table[xgi];
            for (uInt k = 0; k < Basis::NumBasis; k++) {
                rho_qp[xgi] += basis[k] * current_coeffs[cellId][0 + 5*k];  // Density
            }
        }
        Scalar rho_min = *std::min_element(rho_qp.begin(), rho_qp.end());
        Scalar rho_avg = current_coeffs[cellId][0];
        if (rho_min < eps) {
            Scalar effective_rho_avg = std::max(rho_avg, eps);
            Scalar numerator = effective_rho_avg - eps;
            Scalar denominator = effective_rho_avg - rho_min;

            Scalar theta = (denominator < 1e-32) ? 0.0 : std::min(1.0, numerator / denominator);

            // Apply to high-order modes of density
            for (uInt k = 1; k < Basis::NumBasis; k++) {
                current_coeffs[cellId][0 + 5*k] *= theta;
            }
        }
    }
    void apply_22_limitCell(uInt cellId, 
                LongVector<5 * Basis::NumBasis>& current_coeffs) {
        // 对每个单元 cellId 进行 压强的 保正
        // 获取单元平均值
        DenseMatrix<5,1> avg_U;
        for (int k = 0; k < 5; ++k) {
            avg_U[k] = current_coeffs[cellId][k]; 
        }

        Scalar theta_p = 1.0;

        // 对每一个积分点
        constexpr auto Qpoints = QuadType::get_points();
        constexpr auto Qweights = QuadType::get_weights();
        for (uInt xgi = 0; xgi < QuadType::num_points; ++xgi) {
            // const auto& basis = Basis::eval_all(Qpoints[xgi][0], Qpoints[xgi][1], Qpoints[xgi][2]);
            const auto& basis = basis_table[xgi];

            // 计算积分点 xg 处，所有基函数组合的完整的解
            DenseMatrix<5,1> point_U;
            for (int k = 0; k < 5; ++k) {
                Scalar val = 0.0;
                for (uInt l = 0; l < Basis::NumBasis; ++l) {
                    val += basis[l] * current_coeffs[cellId][k + 5*l];
                }
                point_U[k] = val;
            }

            // 检查当前点压强是否为负，飞醋才需要走下一步
            Scalar p_gl = compute_pressure(point_U);
            if (p_gl >= 0) continue;

            // 二分法
            Scalar t_low = 0.0, t_high = 1.0;
            Scalar t_mid = 0.5 * (t_low + t_high);
            for (int iter = 0; iter < 100; ++iter) {
                t_mid = 0.5 * (t_low + t_high);
                DenseMatrix<5,1> U_mid;
                for (int k = 0; k < 5; ++k) {
                    U_mid[k] = (1.0 - t_mid) * avg_U[k] + t_mid * point_U[k];
                }

                Scalar p_mid = compute_pressure(U_mid);

                if (p_mid < 0) {
                    t_high = t_mid;
                } else {
                    t_low = t_mid;
                }
                
            }

            // 全局最小 theta_p
            // if(cellId==0) print(vector2f{theta_p,t_mid});
            theta_p = std::min(theta_p, t_mid);
        }
        // if(cellId==0) print(theta_p);
        for (int k = 0; k < 5; ++k) {          // 遍历每个守恒变量
            for (uInt l = 1; l < Basis::NumBasis; ++l) {  // 只修改高阶项
                current_coeffs[cellId][k + 5*l] *= theta_p;
            }
        }
        
    }

};
