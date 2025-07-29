#pragma once
#include "base/type.h"
#include "dg/dg_basis/dg_basis.h"
#include "mesh/mesh.h"
#include "matrix/matrix.h"
#include "base/exact.h"
#include "dg/dg_flux/euler_physical_flux.h"

template<uInt OrderU=3, uInt OrderP=OrderU-1, typename Flux = AirFluxC, 
        typename GaussQuadCell = GaussLegendreTet::Auto, 
        typename GaussQuadFace = GaussLegendreTri::Auto>
class InCompressibleImplicitConvection {
private:
    // 基函数空间定义
    using BasisU = DGBasisEvaluator<OrderU>;  // 速度基ϕ ∈ V^p
    using BasisP = DGBasisEvaluator<OrderP>;  // 压力基ψ ∈ V^{p-1}

    // 积分规则选择
    using QuadC = typename std::conditional_t<
        std::is_same_v<GaussQuadCell, GaussLegendreTet::Auto>,
        typename AutoQuadSelector<OrderU, GaussLegendreTet::Auto>::type,
        GaussQuadCell
    >;
    using QuadF = typename std::conditional_t<
        std::is_same_v<GaussQuadFace, GaussLegendreTri::Auto>,
        typename AutoQuadSelector<OrderU, GaussLegendreTri::Auto>::type,
        GaussQuadFace
    >;

    // 维度常量
    static constexpr uInt NU = BasisU::NumBasis;  // 速度基数量
    static constexpr uInt NP = BasisP::NumBasis;  // 压力基数量
    static constexpr uInt DoFs = 3 * NU + NP;     // 总自由度

    // 坐标转换工具函数
    vector3f transform_to_cell(const CompTriangleFace& face, 
                             const vector2f& uv, 
                             uInt side) const;

    /* 矩阵分块组装工具函数
       block: 目标矩阵块
       row: 测试函数索引j (对应矩阵行)
       col: 试探函数索引i (对应矩阵列) */
    void assemble_Auu(DenseMatrix<DoFs,DoFs>& block, 
                 const DenseMatrix<3,3>& Auu,
                 uInt row, uInt col);
    void assemble_Auu(DenseMatrix<DoFs,DoFs>& block, 
                 const Scalar& Auu,
                 uInt row, uInt col);
    void assemble_Auu(DenseMatrix<DoFs,DoFs>& block, 
                 const DenseMatrix<3,1>& Auu,
                 uInt row, uInt col);
    void assemble_Aup(DenseMatrix<DoFs,DoFs>& block, 
                 const DenseMatrix<3,1>& Aup,
                 uInt row, uInt col);
    void assemble_Apu(DenseMatrix<DoFs,DoFs>& block, 
                 const DenseMatrix<1,3>& Apu,
                 uInt row, uInt col);
    void assemble_App(DenseMatrix<DoFs,DoFs>& block, 
                 const DenseMatrix<1,1>& App,
                 uInt row, uInt col);
    void assemble_App(DenseMatrix<DoFs,DoFs>& block, 
                 const Scalar& App,
                 uInt row, uInt col);
    void assemble_bu(DenseMatrix<DoFs,1>& block_rhs, 
                 const DenseMatrix<3,1>& bu, uInt row);
    void assemble_bp(DenseMatrix<DoFs,1>& block_rhs, 
                 const DenseMatrix<1,1>& bu, uInt row);
    void assemble_bp(DenseMatrix<DoFs,1>& block_rhs, 
                 const Scalar& bu, uInt row);

public:
    void assemble(const ComputingMesh& mesh, 
                 const LongVector<DoFs>& old_solution,
                 const Scalar curr_time,
                 BlockSparseMatrix<DoFs,DoFs>& sparse_mat,
                 LongVector<DoFs>& sparse_rhs);
private:
    // 单元局部矩阵组装
    void assemble_cells(const ComputingMesh& mesh, 
                      const LongVector<DoFs>& old_solution,
                      BlockSparseMatrix<DoFs,DoFs>& sparse_mat);
    
    // 内部面矩阵组装
    void assemble_internal_faces(const ComputingMesh& mesh,
                                const LongVector<DoFs>& old_solution,
                                BlockSparseMatrix<DoFs,DoFs>& sparse_mat);

    // 边界面矩阵组装
    void assemble_boundary_faces(const ComputingMesh& mesh, 
                 const LongVector<DoFs>& old_solution,
                 const Scalar curr_time,
                 BlockSparseMatrix<DoFs,DoFs>& sparse_mat,
                 LongVector<DoFs>& sparse_rhs);
};

