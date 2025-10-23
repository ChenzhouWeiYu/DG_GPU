// include/dg/dg_schemes/positive_limiters/sampling_points.h
#pragma once
#include "base/type.h"
#include "dg/dg_basis/dg_basis.h"

/// 采样点集合（编译期生成）
template<uInt Order, typename QuadC, typename QuadF, uInt Level>
struct SamplingPoints {
    static constexpr uInt NumBasis = DGBasisEvaluator<Order>::NumBasis;
    static constexpr uInt num_samples = []() {
        if constexpr (Level == 0) return QuadC::num_points;
        else if constexpr (Level == 1) return QuadC::num_points + 4;
        else return QuadC::num_points + 4 + 4*3*QuadF::num_points;
    }();
    
    // 编译期生成所有采样点的基函数值表
    static constexpr auto generateBasisTable() {
        std::array<std::array<Scalar, NumBasis>, num_samples> table{};
        uInt idx = 0;

        // Level 0: 体积分点
        if constexpr (Level >= 0) {
            constexpr auto vol_points = QuadC::get_points();
            for (uInt i = 0; i < QuadC::num_points; ++i) {
                table[idx++] = DGBasisEvaluator<Order>::eval_all(
                    vol_points[i][0], vol_points[i][1], vol_points[i][2]);
            }
        }

        // Level 1: 4 个顶点 (v0, v1, v2, v3)
        if constexpr (Level >= 1) {
            // v0 = (0,0,0)
            table[idx++] = DGBasisEvaluator<Order>::eval_all(0.0, 0.0, 0.0);
            // v1 = (1,0,0)
            table[idx++] = DGBasisEvaluator<Order>::eval_all(1.0, 0.0, 0.0);
            // v2 = (0,1,0)
            table[idx++] = DGBasisEvaluator<Order>::eval_all(0.0, 1.0, 0.0);
            // v3 = (0,0,1)
            table[idx++] = DGBasisEvaluator<Order>::eval_all(0.0, 0.0, 1.0);
        }

        // Level 2: 面积分点（4 面 × 3 轮换）
        if constexpr (Level >= 2) {
            constexpr auto face_points = QuadF::get_points();

            // 面 0: opposite v0 → (v1, v2, v3)
            for (uInt i = 0; i < QuadF::num_points; ++i) {
                const auto& uv = face_points[i];
                // (v1, v2, v3)
                table[idx++] = DGBasisEvaluator<Order>::eval_all(uv[0], uv[1], 1 - uv[0] - uv[1]);
                // (v2, v3, v1)
                table[idx++] = DGBasisEvaluator<Order>::eval_all(1 - uv[0] - uv[1], uv[0], uv[1]);
                // (v3, v1, v2)
                table[idx++] = DGBasisEvaluator<Order>::eval_all(uv[1], 1 - uv[0] - uv[1], uv[0]);
            }

            // 面 1: opposite v1 → (v0, v3, v2)
            for (uInt i = 0; i < QuadF::num_points; ++i) {
                const auto& uv = face_points[i];
                // (v0, v3, v2)
                table[idx++] = DGBasisEvaluator<Order>::eval_all(0.0, 1 - uv[0] - uv[1], uv[0]);
                // (v3, v2, v0)
                table[idx++] = DGBasisEvaluator<Order>::eval_all(0.0, uv[0], 1 - uv[0] - uv[1]);
                // (v2, v0, v3)
                table[idx++] = DGBasisEvaluator<Order>::eval_all(0.0, uv[1], uv[0]);
            }

            // 面 2: opposite v2 → (v0, v1, v3)
            for (uInt i = 0; i < QuadF::num_points; ++i) {
                const auto& uv = face_points[i];
                // (v0, v1, v3)
                table[idx++] = DGBasisEvaluator<Order>::eval_all(uv[0], 0.0, 1 - uv[0] - uv[1]);
                // (v1, v3, v0)
                table[idx++] = DGBasisEvaluator<Order>::eval_all(1 - uv[0] - uv[1], 0.0, uv[0]);
                // (v3, v0, v1)
                table[idx++] = DGBasisEvaluator<Order>::eval_all(uv[1], 0.0, uv[0]);
            }

            // 面 3: opposite v3 → (v0, v2, v1)
            for (uInt i = 0; i < QuadF::num_points; ++i) {
                const auto& uv = face_points[i];
                // (v0, v2, v1)
                table[idx++] = DGBasisEvaluator<Order>::eval_all(uv[1], uv[0], 0.0);
                // (v2, v1, v0)
                table[idx++] = DGBasisEvaluator<Order>::eval_all(1 - uv[0] - uv[1], uv[0], 0.0);
                // (v1, v0, v2)
                table[idx++] = DGBasisEvaluator<Order>::eval_all(uv[0], 1 - uv[0] - uv[1], 0.0);
            }
        }

        return table;
    }

    static constexpr auto basis_table = generateBasisTable();
};