#pragma once
#include "base/type.h"
#include "dg/dg_basis/dg_basis.h"
#include "mesh/mesh.h"
#include "matrix/matrix.h"

/**
 * @brief 严格遵循文献的PWENO限制器，利用正交基特性优化
 * @tparam PolyOrder 多项式阶数
 * @tparam QuadRule 积分规则类型
 */
template <uInt PolyOrder, typename QuadRule>
class OrthoPWENOLimiter {
private:
    const ComputingMesh& mesh_;
    using Basis = DGBasisEvaluator<PolyOrder>;
    static constexpr uInt kNumVars = 5;
    static constexpr uInt kNumLinearModes = 3; // p100, p010, p001

    // 预计算线性基函数的积分数据
    struct {
        std::array<std::array<Scalar, 3>, QuadRule::num_points> points;
        std::array<Scalar, QuadRule::num_points> weights;
        std::array<std::array<Scalar, Basis::NumBasis>, QuadRule::num_points> basis_values;
        std::array<Scalar, kNumLinearModes> linear_mode_norms; // 线性基的L2范数平方
    } quad_;

public:
    explicit OrthoPWENOLimiter(const ComputingMesh& mesh) : mesh_(mesh) {
        // 预计算积分点和基函数值
        for (uInt q = 0; q < QuadRule::num_points; ++q) {
            quad_.points[q] = QuadRule::points[q];
            quad_.weights[q] = QuadRule::weights[q];
            quad_.basis_values[q] = Basis::eval_all(quad_.points[q][0], 
                                                  quad_.points[q][1], 
                                                  quad_.points[q][2]);
        }

        // 预计算线性基的L2范数平方（利用正交性）
        for (uInt m = 0; m < kNumLinearModes; ++m) {
            quad_.linear_mode_norms[m] = 0.0;
            for (uInt q = 0; q < QuadRule::num_points; ++q) {
                Scalar phi = quad_.basis_values[q][m + 1]; // 跳过常数项
                quad_.linear_mode_norms[m] += phi * phi * quad_.weights[q];
            }
        }
    }

    void apply(LongVector<kNumVars * Basis::NumBasis>& current_solution,
               const LongVector<kNumVars * Basis::NumBasis>& previous_solution) {
        for (uInt cell_id = 0; cell_id < mesh_.m_cells.size(); ++cell_id) {
            processCell(cell_id, current_solution, previous_solution);
        }
    }

private:
    void processCell(uInt cell_id,
                    LongVector<kNumVars * Basis::NumBasis>& current_solution,
                    const LongVector<kNumVars * Basis::NumBasis>& previous_solution) {
        const auto& cell = mesh_.m_cells[cell_id];
        std::array<uInt, 4> neighbors;
        uInt num_neighbors = 0;

        // 收集有效邻居
        for (uInt face_id = 0; face_id < 4; ++face_id) {
            if (cell.m_neighbors[face_id] != uInt(-1)) {
                neighbors[num_neighbors++] = cell.m_neighbors[face_id];
            }
        }

        for (uInt var_idx = 0; var_idx < kNumVars; ++var_idx) {
            limitVariable(cell_id, var_idx, neighbors, num_neighbors, 
                         current_solution, previous_solution);
        }
    }

    void limitVariable(uInt cell_id, uInt var_idx, 
                      const std::array<uInt, 4>& neighbors, uInt num_neighbors,
                      LongVector<kNumVars * Basis::NumBasis>& current_solution,
                      const LongVector<kNumVars * Basis::NumBasis>& previous_solution) {
        // --- 阶段1：模态分解 ---
        const Scalar mean = previous_solution[cell_id][var_idx];
        std::array<Scalar, Basis::NumBasis - 1> modal_coeffs;
        for (uInt m = 1; m < Basis::NumBasis; ++m) {
            modal_coeffs[m - 1] = previous_solution[cell_id][var_idx + kNumVars * m];
        }

        // --- 阶段2：邻居线性投影（利用正交性直接求解）---
        std::array<DenseMatrix<1, QuadRule::num_points>, 4> neighbor_projs;
        std::array<Scalar, 4> neighbor_betas = {0};

        for (uInt n = 0; n < num_neighbors; ++n) {
            uInt neighbor_id = neighbors[n];
            std::array<Scalar, kNumLinearModes> coeffs = {0};

            // 在邻居单元上计算线性系数（式(13)的最小二乘解）
            for (uInt m = 0; m < kNumLinearModes; ++m) {
                Scalar numerator = 0.0;
                for (uInt q = 0; q < QuadRule::num_points; ++q) {
                    const auto& xi = quad_.points[q];
                    // 计算p_L(xi) - p̄_L
                    Scalar delta_p = evaluatePoly(neighbor_id, var_idx, xi, previous_solution) 
                                   - previous_solution[neighbor_id][var_idx];
                    numerator += delta_p * quad_.basis_values[q][m + 1] * quad_.weights[q];
                }
                coeffs[m] = numerator / quad_.linear_mode_norms[m];
            }

            // 构建投影多项式 p̃_{L,1}(x) = p̄₀ + a*p100 + b*p010 + c*p001
            for (uInt q = 0; q < QuadRule::num_points; ++q) {
                neighbor_projs[n][q] = mean;
                for (uInt m = 0; m < kNumLinearModes; ++m) {
                    neighbor_projs[n][q] += coeffs[m] * quad_.basis_values[q][m + 1];
                }
                neighbor_betas[n] += neighbor_projs[n][q] * neighbor_projs[n][q] * quad_.weights[q];
            }
        }

        // --- 阶段3：计算光滑度指标（式(17)）---
        std::array<Scalar, 5> betas = {0}; // 自身 + 最多4个邻居
        Scalar h = mesh_.m_cells[cell_id].m_h;

        // 自身高阶模态的光滑度
        for (uInt s = 2; s <= PolyOrder; ++s) {
            Scalar factor = std::pow(h, s) / (std::tgamma(s) * std::pow(2.0, s));
            for (uInt m : getModesForOrder(s)) {
                betas[0] += factor * factor * modal_coeffs[m - 1] * modal_coeffs[m - 1];
            }
        }

        // 邻居投影的光滑度
        for (uInt n = 0; n < num_neighbors; ++n) {
            betas[n + 1] = neighbor_betas[n];
        }

        // --- 阶段4：WENO加权组合（式(15)-(18)）---
        auto recon = combineSolution(mean, modal_coeffs, neighbor_projs, num_neighbors, betas);

        // --- 阶段5：正交投影更新解 ---
        updateSolution(cell_id, var_idx, mean, recon, current_solution);
    }

    // 获取属于特定阶次s的所有模态索引
    std::vector<uInt> getModesForOrder(uInt s) {
        std::vector<uInt> modes;
        uInt start = (s * (s - 1) * (s - 2)) / 6 + 1;
        uInt end = ((s + 1) * s * (s - 1)) / 6;
        for (uInt m = start; m <= end; ++m) {
            modes.push_back(m);
        }
        return modes;
    }

    // 在参考坐标xi处计算多项式值
    Scalar evaluatePoly(uInt cell_id, uInt var_idx, const std::array<Scalar, 3>& xi,
                       const LongVector<kNumVars * Basis::NumBasis>& solution) {
        const auto& basis = Basis::eval_all(xi[0], xi[1], xi[2]);
        Scalar val = 0.0;
        for (uInt m = 0; m < Basis::NumBasis; ++m) {
            val += basis[m] * solution[cell_id][var_idx + kNumVars * m];
        }
        return val;
    }

    // WENO加权重构（式(15)-(18)）
    DenseMatrix<1, QuadRule::num_points> 
    combineSolution(Scalar mean, const std::array<Scalar, Basis::NumBasis - 1>& modal_coeffs,
                   const std::array<DenseMatrix<1, QuadRule::num_points>, 4>& neighbor_projs,
                   uInt num_neighbors, const std::array<Scalar, 5>& betas) {
        // 计算线性权重γ_l（式(18)）
        std::array<Scalar, 5> gammas;
        gammas[0] = 1.0; // 邻居权重γ_L/R
        gammas[1] = 10.0; // 一阶模态γ_1
        for (uInt s = 2; s <= PolyOrder; ++s) {
            gammas[s] = 10;//std::pow(10.0, s + 1); // γ_s = 10^{s+1}    
        }

        // 计算ε_l（式(18)）
        std::array<Scalar, 5> epsilons;
        Scalar eps_base = 0.01 * (betas[1] + betas[2]) / 2.0 + 1e-16;
        epsilons[0] = eps_base; // ε_L/R
        epsilons[1] = eps_base; // ε_1
        for (uInt s = 2; s <= PolyOrder; ++s) {
            epsilons[s] = 1e-16; // ε_s
        }

        // 计算非线性权重ω_l（式(16)）
        std::array<Scalar, 5> omegas;
        Scalar sum_omegas = 0.0;
        for (uInt l = 0; l <= PolyOrder; ++l) {
            omegas[l] = gammas[l] / (epsilons[l] + betas[l] * betas[l]);
            sum_omegas += omegas[l];
        }
        for (uInt l = 0; l <= PolyOrder; ++l) {
            omegas[l] /= sum_omegas;
        }

        // 加权组合（式(15)）
        DenseMatrix<1, QuadRule::num_points> recon;
        for (uInt q = 0; q < QuadRule::num_points; ++q) {
            // 自身模态贡献
            recon[q] = mean;
            for (uInt m = 1; m < Basis::NumBasis; ++m) {
                uInt s = getOrderForMode(m);
                recon[q] += omegas[s] * modal_coeffs[m - 1] * quad_.basis_values[q][m];
            }

            // 邻居贡献（共享ω_0）
            for (uInt n = 0; n < num_neighbors; ++n) {
                recon[q] += omegas[0] * neighbor_projs[n][q];
            }
        }
        return recon;
    }

    // 获取模态对应的阶次
    uInt getOrderForMode(uInt m) {
        uInt s = 1;
        while (m > (s * (s + 1) * (s + 2) / 6)) {
            s++;
        }
        return s;
    }

    // 正交投影更新解
    void updateSolution(uInt cell_id, uInt var_idx, Scalar mean,
                       const DenseMatrix<1, QuadRule::num_points>& recon,
                       LongVector<kNumVars * Basis::NumBasis>& solution) {
                // 计算原始解值
        std::array<Scalar, QuadRule::num_points> original;
        for (uInt q = 0; q < QuadRule::num_points; ++q) {
            original[q] = solution[cell_id][var_idx]; // 均值
            for (uInt m = 1; m < Basis::NumBasis; ++m) {
                original[q] += solution[cell_id][var_idx + kNumVars * m] 
                             * quad_.basis_values[q][m];
            }
        }

        // 计算安全缩放因子
        Scalar theta = 1.0;
        for (uInt q = 0; q < QuadRule::num_points; ++q) {
            Scalar delta = recon[q] - original[q];
            if (std::abs(delta) > 1e-8 * std::abs(original[q])) {
                theta = std::min(theta, 0.8 * std::abs(original[q]) / std::abs(delta));
            }
        }

        // 应用缩放
        for (uInt m = 1; m < Basis::NumBasis; ++m) {
            solution[cell_id][var_idx + kNumVars * m] *= theta;
        }
    }
};