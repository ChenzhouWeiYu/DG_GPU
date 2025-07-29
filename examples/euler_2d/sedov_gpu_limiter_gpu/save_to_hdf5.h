#pragma once

#include "base/type.h"
#include "dg/dg_basis/dg_basis.h"
#include "mesh/mesh.h"
#include "matrix/matrix.h"
#include <H5Cpp.h>


/**
 * @brief 将 DG 解系数、单元信息、基函数查找表等写入 HDF5 文件
 * @tparam Basis 多项式基函数计算器（如 DGBasisEvaluator<Order>）
 * @tparam QuadC 作图点类（如 GaussLegendreTet::Degree5Points）
 */
template<typename QuadC, typename Basis, uInt N_var = 5>
void save_DG_solution_to_hdf5(
    const ComputingMesh& cmesh,
    const LongVector<N_var * Basis::NumBasis>& coef,
    const std::string& filename,
    Scalar time = 0.0,
    int step = 0)
{
    constexpr uInt N_basis = Basis::NumBasis;
    constexpr uInt N_plot = QuadC::num_points;
    uInt N_cell = cmesh.m_cells.size();

    // 构造标准单元上作图点和基函数查找表
    std::array<std::array<Scalar, 3>,N_plot> plot_points;
    std::array<std::array<Scalar,N_basis>,N_plot> basis_table;

    for (uInt xg = 0; xg < N_plot; ++xg) {
        const auto& xg_coord = QuadC::get_points()[xg];
        plot_points[xg] = {xg_coord[0], xg_coord[1], xg_coord[2]};
        auto phi_vals = Basis::eval_all(xg_coord[0], xg_coord[1], xg_coord[2]);
        for (uInt l = 0; l < N_basis; ++l) {
            basis_table[xg][l] = phi_vals[l];
        }
    }

    // 每个单元顶点 (4 点 * 3 维)
    std::vector<std::array<std::array<Scalar, 3>, 4>> vertices(N_cell);
    for (uInt cid = 0; cid < N_cell; ++cid) {
        for (int v = 0; v < 4; ++v) {
            const auto& pid = cmesh.m_cells[cid].m_nodes[v];
            const auto& p = cmesh.m_points[pid];
            vertices[cid][v] = {p[0], p[1], p[2]};
        }
    }

    // 每个单元 5*N_basis 个系数
    std::vector<std::array<std::array<Scalar, N_basis>, N_var>> coeffs(N_cell);
    for (uInt cid = 0; cid < N_cell; ++cid) {
        for (uInt k = 0; k < N_var; ++k) {
            for (uInt l = 0; l < N_basis; ++l) {
                coeffs[cid][k][l] = coef[cid][5*l + k];
            }
        }
    }

    // 写入 HDF5
    H5::H5File file(filename, H5F_ACC_TRUNC);

    // 保存标量 time 和 step
    H5::DataSpace scalar_space(H5S_SCALAR);
    H5::Attribute attr_time = file.createAttribute("time", H5::PredType::NATIVE_DOUBLE, scalar_space);
    attr_time.write(H5::PredType::NATIVE_DOUBLE, &time);
    H5::Attribute attr_step = file.createAttribute("step", H5::PredType::NATIVE_INT, scalar_space);
    attr_step.write(H5::PredType::NATIVE_INT, &step);

    // 写 basis_table: [N_plot][N_basis]
    hsize_t dims_basis[2] = {N_plot, N_basis};
    H5::DataSpace basis_space(2, dims_basis);
    H5::DataSet basis_dset = file.createDataSet("basis_table", H5::PredType::NATIVE_DOUBLE, basis_space);
    basis_dset.write(basis_table[0].data(), H5::PredType::NATIVE_DOUBLE);

    // 写 plot_points: [N_plot][3]
    hsize_t dims_plot[2] = {N_plot, 3};
    H5::DataSpace plot_space(2, dims_plot);
    H5::DataSet plot_dset = file.createDataSet("plot_points", H5::PredType::NATIVE_DOUBLE, plot_space);
    plot_dset.write(plot_points[0].data(), H5::PredType::NATIVE_DOUBLE);

    // 写 vertices: [N_cell][4][3]
    hsize_t dims_verts[3] = {N_cell, 4, 3};
    H5::DataSpace vert_space(3, dims_verts);
    H5::DataSet vert_dset = file.createDataSet("vertices", H5::PredType::NATIVE_DOUBLE, vert_space);
    vert_dset.write(vertices[0][0].data(), H5::PredType::NATIVE_DOUBLE);

    // 写 coeffs: [N_cell][5][N_basis]
    hsize_t dims_coef[3] = {N_cell, N_var, N_basis};
    H5::DataSpace coef_space(3, dims_coef);
    H5::DataSet coef_dset = file.createDataSet("coeffs", H5::PredType::NATIVE_DOUBLE, coef_space);
    coef_dset.write(coeffs[0][0].data(), H5::PredType::NATIVE_DOUBLE);

    file.close();

    std::cout << "Saved HDF5 to " << filename << " with " << N_cell << " cells.\n";
}
