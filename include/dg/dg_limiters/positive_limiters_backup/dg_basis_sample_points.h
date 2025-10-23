#pragma once
#include "base/type.h"
#include "dg/dg_basis/dg_basis.h"

template<uInt Order, typename QuadC, typename QuadF, uInt Level>
struct DGBasisSamplePoints {
    static constexpr uInt NumBasis = DGBasisEvaluator<Order>::NumBasis;
    static constexpr uInt NumVolPoints = QuadC::num_points;
    static constexpr uInt NumFacePoints = QuadF::num_points;

    // 编译期计算采样点数量
    static constexpr uInt computeNumSamples() {
        if constexpr (Level == 0) {
            return NumVolPoints;
        } else if constexpr (Level == 1) {
            return NumVolPoints + 4; // 4 vertices
        } else { // Level == 2
            return NumVolPoints + 4 + 4 * NumFacePoints; // 4 faces
        }
    }

    static constexpr uInt NumSamples = computeNumSamples();

    // 编译期生成采样点坐标
    static constexpr auto generateSamplePoints() {
        std::array<vector3f, NumSamples> points{};
        uInt idx = 0;

        // 1. 体积高斯点
        constexpr auto vol_points = QuadC::get_points();
        for (uInt i = 0; i < NumVolPoints; ++i) {
            points[idx++] = vol_points[i];
        }

        if constexpr (Level >= 1) {
            // 2. 4 个顶点 (0,0,0), (1,0,0), (0,1,0), (0,0,1)
            points[idx++] = {0.0, 0.0, 0.0};
            points[idx++] = {1.0, 0.0, 0.0};
            points[idx++] = {0.0, 1.0, 0.0};
            points[idx++] = {0.0, 0.0, 1.0};
        }

        if constexpr (Level == 2) {
            // 3. 4 个面的高斯点
            constexpr auto face_points = QuadF::get_points();
            // 面 0: (0,0,0), (1,0,0), (0,1,0) -> z=0
            for (uInt i = 0; i < NumFacePoints; ++i) {
                Scalar u = face_points[i][0], v = face_points[i][1];
                points[idx++] = {u, v, 0.0};
            }
            // 面 1: (0,0,0), (0,1,0), (0,0,1) -> x=0
            for (uInt i = 0; i < NumFacePoints; ++i) {
                Scalar u = face_points[i][0], v = face_points[i][1];
                points[idx++] = {0.0, u, v};
            }
            // 面 2: (0,0,0), (0,0,1), (1,0,0) -> y=0
            for (uInt i = 0; i < NumFacePoints; ++i) {
                Scalar u = face_points[i][0], v = face_points[i][1];
                points[idx++] = {u, 0.0, v};
            }
            // 面 3: (1,0,0), (0,1,0), (0,0,1) -> x+y+z=1
            for (uInt i = 0; i < NumFacePoints; ++i) {
                Scalar u = face_points[i][0], v = face_points[i][1];
                points[idx++] = {1.0 - u - v, u, v};
            }
        }
        return points;
    }

    // 编译期计算基函数值表
    static constexpr auto generateBasisTable() {
        constexpr auto points = generateSamplePoints();
        std::array<std::array<Scalar, NumBasis>, NumSamples> table{};
        for (uInt i = 0; i < NumSamples; ++i) {
            table[i] = DGBasisEvaluator<Order>::eval_all(points[i][0], points[i][1], points[i][2]);
        }
        return table;
    }

    // 公共接口
    static constexpr auto getBasisTable() {
        return generateBasisTable();
    }

    static constexpr uInt getNumSamples() {
        return NumSamples;
    }
};