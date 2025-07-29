#include "base/io.h"
#include "base/exact.h"
#include <H5Cpp.h>

/**
 * @brief 将 DG 解系数、单元信息、基函数查找表等写入 HDF5 文件
 * @tparam Basis 多项式基函数计算器（如 DGBasisEvaluator<Order>）
 * @tparam QuadC 作图点类（如 GaussLegendreTet::Degree5Points）
 */
template<typename QuadC, typename Basis, uInt N_var>
void save_DG_solution_to_hdf5(
    const ComputingMesh& cmesh,
    const LongVector<N_var * Basis::NumBasis>& coef,
    const std::string filename,
    Scalar time,
    uInt step)
{
    constexpr uInt N_basis = Basis::NumBasis;
    constexpr uInt N_plot = QuadC::num_points;
    uInt N_cell = cmesh.m_cells.size();
    constexpr auto Qpoints = QuadC::get_points();
    constexpr auto Qweights = QuadC::get_weights();

    // ---- 构造标准单元上的查找表 ---- //
    std::vector<Scalar> basis_table(N_plot * N_basis);
    std::vector<Scalar> plot_points(N_plot * 3);

    for (std::size_t xg = 0; xg < N_plot; ++xg) {
        const auto& xg_coord = Qpoints[xg];
        plot_points[3*xg + 0] = xg_coord[0];
        plot_points[3*xg + 1] = xg_coord[1];
        plot_points[3*xg + 2] = xg_coord[2];
        auto phi_vals = Basis::eval_all(xg_coord[0], xg_coord[1], xg_coord[2]);
        for (std::size_t l = 0; l < N_basis; ++l) {
            basis_table[xg * N_basis + l] = phi_vals[l];
        }
    }

    // ---- 每个单元顶点 ---- //
    std::vector<Scalar> vertices(N_cell * 4 * 3);
    for (std::size_t cid = 0; cid < N_cell; ++cid) {
        for (int v = 0; v < 4; ++v) {
            const auto& pid = cmesh.m_cells[cid].m_nodes[v];
            const auto& p = cmesh.m_points[pid];
            vertices[cid * 12 + 3*v + 0] = p[0];
            vertices[cid * 12 + 3*v + 1] = p[1];
            vertices[cid * 12 + 3*v + 2] = p[2];
        }
    }

    // ---- 系数数组 ---- //
    std::vector<Scalar> coeffs(N_cell * N_var * N_basis);
    for (std::size_t cid = 0; cid < N_cell; ++cid) {
        for (std::size_t k = 0; k < N_var; ++k) {
            for (std::size_t l = 0; l < N_basis; ++l) {
                coeffs[(cid * N_var + k) * N_basis + l] = coef[cid](5*l + k, 0);
            }
        }
    }

    // ---- 写入 HDF5 文件 ---- //
    H5::H5File file(filename, H5F_ACC_TRUNC);

    // 元数据 time/step
    H5::DataSpace scalar_space(H5S_SCALAR);
    file.createAttribute("time", H5::PredType::NATIVE_DOUBLE, scalar_space).write(H5::PredType::NATIVE_DOUBLE, &time);
    hsize_t step64 = static_cast<hsize_t>(step);
    file.createAttribute("step", H5::PredType::NATIVE_HSIZE, scalar_space).write(H5::PredType::NATIVE_HSIZE, &step64);

    // basis_table
    hsize_t dims_basis[2] = {N_plot, N_basis};
    H5::DataSpace basis_space(2, dims_basis);
    file.createDataSet("basis_table", H5::PredType::NATIVE_DOUBLE, basis_space).write(basis_table.data(), H5::PredType::NATIVE_DOUBLE);

    // plot_points
    hsize_t dims_plot[2] = {N_plot, 3};
    H5::DataSpace plot_space(2, dims_plot);
    file.createDataSet("plot_points", H5::PredType::NATIVE_DOUBLE, plot_space).write(plot_points.data(), H5::PredType::NATIVE_DOUBLE);

    // vertices
    hsize_t dims_vert[3] = {N_cell, 4, 3};
    H5::DataSpace vert_space(3, dims_vert);
    file.createDataSet("vertices", H5::PredType::NATIVE_DOUBLE, vert_space).write(vertices.data(), H5::PredType::NATIVE_DOUBLE);

    // coeffs
    hsize_t dims_coef[3] = {N_cell, N_var, N_basis};
    H5::DataSpace coef_space(3, dims_coef);
    file.createDataSet("coeffs", H5::PredType::NATIVE_DOUBLE, coef_space).write(coeffs.data(), H5::PredType::NATIVE_DOUBLE);

    file.close();

    std::cout << "Saved HDF5 to " << filename << " with " << N_cell << " cells.\n";
}



template<typename X3QuadC, typename S2QuadC, typename X3Basis, typename S2Basis, typename S2Mesh>
void save_RTE_solution_to_hdf5(
    const ComputingMesh& cmesh,
    const LongVector<S2Mesh::num_cells * X3Basis::NumBasis * S2Basis::NumBasis>& coef,
    const std::string& filename,
    Scalar time,
    uInt step)
{
    constexpr uInt X3DoFs = X3Basis::NumBasis;
    constexpr uInt S2DoFs = S2Basis::NumBasis;
    constexpr uInt S2Cells = S2Mesh::num_cells;
    constexpr uInt DoFs = X3DoFs * S2DoFs * S2Cells;

    const uInt N_cell = cmesh.m_cells.size();
    constexpr uInt N_plot_x3 = X3QuadC::num_points;
    constexpr uInt N_plot_s2 = S2QuadC::num_points;

    const auto& Qx = X3QuadC::get_points();
    const auto& Qs = S2QuadC::get_points();

    // ------------------- 标准单元绘图点 & 基函数表 -------------------- //
    std::vector<Scalar> plot_points_x3(N_plot_x3 * 3);
    std::vector<Scalar> plot_points_s2(N_plot_s2 * 2);
    std::vector<Scalar> basis_table_x3(N_plot_x3 * X3DoFs);
    std::vector<Scalar> basis_table_s2(N_plot_s2 * S2DoFs);

    for (uInt g = 0; g < N_plot_x3; ++g) {
        const auto& p = Qx[g];
        plot_points_x3[3*g+0] = p[0];
        plot_points_x3[3*g+1] = p[1];
        plot_points_x3[3*g+2] = p[2];

        auto phi = X3Basis::eval_all(p[0], p[1], p[2]);
        for (uInt i = 0; i < X3DoFs; ++i)
            basis_table_x3[g * X3DoFs + i] = phi[i];
    }

    for (uInt g = 0; g < N_plot_s2; ++g) {
        const auto& uv = Qs[g];
        plot_points_s2[2*g+0] = uv[0];
        plot_points_s2[2*g+1] = uv[1];

        auto theta = S2Basis::eval_all(uv[0], uv[1]);
        for (uInt j = 0; j < S2DoFs; ++j)
            basis_table_s2[g * S2DoFs + j] = theta[j];
    }

    // ------------------- 空间单元顶点 -------------------- //
    std::vector<Scalar> vertices_x3(N_cell * 4 * 3);
    for (uInt cid = 0; cid < N_cell; ++cid) {
        for (uInt v = 0; v < 4; ++v) {
            const auto& pid = cmesh.m_cells[cid].m_nodes[v];
            const auto& p = cmesh.m_points[pid];
            vertices_x3[cid * 12 + 3*v + 0] = p[0];
            vertices_x3[cid * 12 + 3*v + 1] = p[1];
            vertices_x3[cid * 12 + 3*v + 2] = p[2];
        }
    }

    // ------------------- 角度单元顶点 -------------------- //
    constexpr auto s2_cells = S2Mesh::s2_cells();
    std::vector<Scalar> vertices_s2(S2Cells * 3 * 2);
    for (uInt aid = 0; aid < S2Cells; ++aid) {
        const auto& cell = s2_cells[aid];
        for (uInt v = 0; v < 3; ++v) {
            vertices_s2[aid * 6 + 2*v + 0] = cell.vertices[v][0]; // φ
            vertices_s2[aid * 6 + 2*v + 1] = cell.vertices[v][1]; // μ
        }
    }

    // ------------------- 系数数组 -------------------- //
    std::vector<Scalar> coeffs(N_cell * S2Cells * X3DoFs * S2DoFs);
    for (uInt cid = 0; cid < N_cell; ++cid) {
        for (uInt aid = 0; aid < S2Cells; ++aid) {
            for (uInt i = 0; i < X3DoFs; ++i) {
                for (uInt j = 0; j < S2DoFs; ++j) {
                    uInt local_idx = aid * X3DoFs * S2DoFs + i * S2DoFs + j;
                    coeffs[(((cid * S2Cells) + aid) * X3DoFs + i) * S2DoFs + j] =
                        coef[cid](local_idx, 0);
                }
            }
        }
    }

    // ------------------- 写入 HDF5 -------------------- //
    H5::H5File file(filename, H5F_ACC_TRUNC);
    H5::DataSpace scalar_space(H5S_SCALAR);
    file.createAttribute("time", H5::PredType::NATIVE_DOUBLE, scalar_space).write(H5::PredType::NATIVE_DOUBLE, &time);
    hsize_t step64 = static_cast<hsize_t>(step);
    file.createAttribute("step", H5::PredType::NATIVE_HSIZE, scalar_space).write(H5::PredType::NATIVE_HSIZE, &step64);

    // plot_points + basis_table
    
    hsize_t dims_plot_points_x3[2] = {N_plot_x3, 3};
    H5::DataSpace plot_points_x3_space(2, dims_plot_points_x3);
    hsize_t dims_basis_table_x3[2] = {N_plot_x3, X3DoFs};
    H5::DataSpace basis_table_x3space(2, dims_basis_table_x3);
    file.createDataSet("plot_points_x3", H5::PredType::NATIVE_DOUBLE, plot_points_x3_space).write(plot_points_x3.data(), H5::PredType::NATIVE_DOUBLE);
    file.createDataSet("basis_table_x3", H5::PredType::NATIVE_DOUBLE, basis_table_x3space).write(basis_table_x3.data(), H5::PredType::NATIVE_DOUBLE);
    
    
    hsize_t dims_plot_points_s2[2] = {N_plot_s2, 2};
    H5::DataSpace plot_points_s2_space(2, dims_plot_points_s2);
    hsize_t dims_basis_table_s2[2] = {N_plot_s2, S2DoFs};
    H5::DataSpace basis_table_s2space(2, dims_basis_table_s2);
    file.createDataSet("plot_points_s2", H5::PredType::NATIVE_DOUBLE, plot_points_s2_space).write(plot_points_s2.data(), H5::PredType::NATIVE_DOUBLE);
    file.createDataSet("basis_table_s2", H5::PredType::NATIVE_DOUBLE, basis_table_s2space).write(basis_table_s2.data(), H5::PredType::NATIVE_DOUBLE);

    // vertices
    hsize_t dims_vertices_x3[3] = {N_cell, 4, 3};
    H5::DataSpace vertices_x3_space(3, dims_vertices_x3);
    hsize_t dims_vertices_s2[3] = {S2Cells, 3, 2};
    H5::DataSpace vertices_s2_space(3, dims_vertices_s2);
    file.createDataSet("vertices_x3", H5::PredType::NATIVE_DOUBLE, vertices_x3_space).write(vertices_x3.data(), H5::PredType::NATIVE_DOUBLE);
    file.createDataSet("vertices_s2", H5::PredType::NATIVE_DOUBLE, vertices_s2_space).write(vertices_s2.data(), H5::PredType::NATIVE_DOUBLE);

    // coeffs
    hsize_t dims_coeffs[4] = {N_cell, S2Cells, X3DoFs, S2DoFs};
    H5::DataSpace coeffs_space(4, dims_coeffs);
    file.createDataSet("coeffs", H5::PredType::NATIVE_DOUBLE, coeffs_space).write(coeffs.data(), H5::PredType::NATIVE_DOUBLE);

    file.close();

    std::cout << "RTE solution saved to: " << filename
              << " with " << N_cell << " space cells and " << S2Cells << " angular cells.\n";
}



// template<typename QuadC, typename Basis>
// std::tuple<LongVector<5*QuadC::num_points>, 
//             LongVector<5*QuadC::num_points>, 
//             LongVector<5*Basis::NumBasis>> 
// reconstruct_solution(const ComputingMesh& cmesh, const LongVector<5*Basis::NumBasis>& coef, Scalar curr_time){
//     LongVector<5*QuadC::num_points> U_h(cmesh.m_cells.size());
//     LongVector<5*QuadC::num_points> U_s(cmesh.m_cells.size());
//     LongVector<5*Basis::NumBasis> error(cmesh.m_cells.size());

//     #pragma omp parallel for schedule(dynamic)
//     for(uInt cellId=0;cellId<cmesh.m_cells.size();cellId++){
//         const auto& cell = cmesh.m_cells[cellId];
//         const auto& U_func = [&](vector3f Xi)->DenseMatrix<5,1>{
//             return U_Xi(cell,Xi,curr_time);
//         };
//         const auto& U_coef = Basis::func2coef(U_func);
//         for(uInt k=0;k<Basis::NumBasis;k++){
//             const auto& block_coef = coef[cellId].template SubMat<5,1>(5*k,0) - U_coef[k];
//             MatrixView<5*Basis::NumBasis,1,5,1>(error[cellId],5*k,0) = block_coef;
//         }
//         constexpr auto Qpoints = QuadC::get_points();
//         constexpr auto Qweights = QuadC::get_weights();
//         for(uInt xgId=0; xgId<QuadC::num_points; ++xgId) {
//             const auto& p = Qpoints[xgId];
//             const auto& pos = cell.transform_to_physical(p);
//             const auto& value = Basis::eval_all(p[0],p[1],p[2]);
//             const auto& U = Basis::template coef2filed<5,Scalar>(coef[cellId],p);
//             MatrixView<5*QuadC::num_points,1,5,1> block_U_h(U_h[cellId],5*xgId,0);
//             MatrixView<5*QuadC::num_points,1,5,1> block_U_s(U_s[cellId],5*xgId,0);
            
//             block_U_h = U;
//             block_U_h[0] = U[0];
//             block_U_s = U_func(p);
//         }
//     }
//     return {U_h, U_s, error};

// }

template<typename QuadC, typename Basis>
std::tuple<LongVector<5*QuadC::num_points>, 
            LongVector<5*QuadC::num_points>, 
            LongVector<5*Basis::NumBasis>> 
reconstruct_solution(const ComputingMesh& cmesh, const LongVector<5*Basis::NumBasis>& coef, Scalar curr_time) {
    
    LongVector<5*QuadC::num_points> U_h(cmesh.m_cells.size());
    LongVector<5*QuadC::num_points> U_s(cmesh.m_cells.size());
    LongVector<5*Basis::NumBasis> error(cmesh.m_cells.size());

    #pragma omp parallel for schedule(dynamic)
    for (uInt cellId = 0; cellId < cmesh.m_cells.size(); cellId++) {
        const auto& cell = cmesh.m_cells[cellId];
        const auto& U_func = [&](vector3f Xi) -> DenseMatrix<5,1> {
            return U_Xi(cell, Xi, curr_time);
        };
        const auto& U_coef = Basis::func2coef(U_func);
        for (uInt k = 0; k < Basis::NumBasis; k++) {
            // 手动访问 block_coef = coef[cellId].SubMat<5,1>(5*k,0) - U_coef[k];
            for (uInt i = 0; i < 5; i++) {
                error[cellId](5*k + i, 0) = coef[cellId](5*k + i, 0) - U_coef[k](i, 0);
            }
        }
        constexpr auto Qpoints = QuadC::get_points();
        for (uInt xgId = 0; xgId < QuadC::num_points; ++xgId) {
            const auto& p = Qpoints[xgId];
            const auto U = Basis::template coef2filed<5, Scalar>(coef[cellId], p);
            const auto U_ref = U_func(p);
            // 手动写入 U_h, U_s
            for (uInt i = 0; i < 5; i++) {
                U_h[cellId](5*xgId + i, 0) = U(i, 0);
                U_s[cellId](5*xgId + i, 0) = U_ref(i, 0);
            }
        }
    }
    return {U_h, U_s, error};
}


template<typename QuadC, typename Basis>
LongVector<5*QuadC::num_points>
reconstruct_solution(const ComputingMesh& cmesh, const LongVector<5*Basis::NumBasis>& coef) {
    
    LongVector<5*QuadC::num_points> U_h(cmesh.m_cells.size());

    #pragma omp parallel for schedule(dynamic)
    for (uInt cellId = 0; cellId < cmesh.m_cells.size(); cellId++) {
        const auto& cell = cmesh.m_cells[cellId];
        constexpr auto Qpoints = QuadC::get_points();
        for (uInt xgId = 0; xgId < QuadC::num_points; ++xgId) {
            const auto& p = Qpoints[xgId];
            const auto U = Basis::template coef2filed<5, Scalar>(coef[cellId], p);
            // 手动写入 U_h, U_s
            for (uInt i = 0; i < 5; i++) {
                U_h[cellId](5*xgId + i, 0) = U(i, 0);
            }
        }
    }
    return U_h;
}



template<typename QuadC, typename Basis>
void save_Uh(const ComputingMesh& cmesh, const LongVector<5*Basis::NumBasis>& U_n, Scalar total_time, const std::string filename) {
    std::ofstream fp(filename);
    const auto& U_h= reconstruct_solution<QuadC,Basis>(cmesh, U_n);
    constexpr auto Qpoints = QuadC::get_points();
    for (uInt cellId = 0; cellId < cmesh.m_cells.size(); cellId++) {
        const auto& cell = cmesh.m_cells[cellId];
        for (uInt xgId = 0; xgId < QuadC::num_points; ++xgId) {
            const auto& p = Qpoints[xgId];
            const auto& pos = cell.transform_to_physical(p);
            fp << std::setprecision(16) << pos[0] << " " 
               << std::setprecision(16) << pos[1] << " " 
               << std::setprecision(16) << pos[2];
            for (uInt i = 0; i < 5; i++) {
                fp << " " << std::setprecision(16) << U_h[cellId](5*xgId + i, 0);
            }
            fp << std::endl;
        }
    }
    fp.close();
}

template<typename QuadC, typename Basis>
void save_Uh_Us(const ComputingMesh& cmesh, const LongVector<5*Basis::NumBasis>& U_n, Scalar total_time, const std::string filename) {
    std::ofstream fp(filename);
    auto [U_h, U_s, error] = reconstruct_solution<QuadC,Basis>(cmesh, U_n, total_time);
    constexpr auto Qpoints = QuadC::get_points();
    for (uInt cellId = 0; cellId < cmesh.m_cells.size(); cellId++) {
        const auto& cell = cmesh.m_cells[cellId];
        for (uInt xgId = 0; xgId < QuadC::num_points; ++xgId) {
            const auto& p = Qpoints[xgId];
            const auto& pos = cell.transform_to_physical(p);
            fp << std::setprecision(16) << pos[0] << " " 
               << std::setprecision(16) << pos[1] << " " 
               << std::setprecision(16) << pos[2];
            for (uInt i = 0; i < 5; i++) {
                fp << " " << std::setprecision(16) << U_h[cellId](5*xgId + i, 0)
                   << " " << std::setprecision(16) << U_s[cellId](5*xgId + i, 0);
            }
            fp << std::endl;
        }
    }
    fp.close();
}




// template<typename QuadC, typename Basis>
// void save_Uh(const ComputingMesh& cmesh, const LongVector<5*Basis::NumBasis>& U_n, Scalar total_time, const std::string filename){
//     std::ofstream fp(filename);
//     auto [U_h, U_s, error] = reconstruct_solution<QuadC,Basis>(cmesh, U_n, total_time);
//     constexpr auto Qpoints = QuadC::get_points();
//     constexpr auto Qweights = QuadC::get_weights();
//     for(uInt cellId=0;cellId<cmesh.m_cells.size();cellId++){
//         const auto& cell = cmesh.m_cells[cellId];
//         for(uInt xgId=0; xgId<QuadC::num_points; ++xgId) {
//             const auto& p = Qpoints[xgId];
//             const auto& pos = cell.transform_to_physical(p);
//             const auto& bUh = U_h[cellId].template SubMat<5,1>(5*xgId,0);
//             const auto& bUs = U_s[cellId].template SubMat<5,1>(5*xgId,0);
//             fp <<std::setprecision(16)<< pos[0] << "  " <<std::setprecision(16)<< pos[1] << "  " <<std::setprecision(16)<< pos[2]
//                 << "  " <<std::setprecision(16)<<  bUh[0]
//                 << "  " <<std::setprecision(16)<<  bUh[1]
//                 << "  " <<std::setprecision(16)<<  bUh[2]
//                 << "  " <<std::setprecision(16)<<  bUh[3]
//                 << "  " <<std::setprecision(16)<<  bUh[4] << std::endl;
//         }
//     }
//     fp.close();
// }

// template<typename QuadC, typename Basis>
// void save_Uh_Us(const ComputingMesh& cmesh, const LongVector<5*Basis::NumBasis>& U_n, Scalar total_time, const std::string filename){
//     std::ofstream fp(filename);
//     auto [U_h, U_s, error] = reconstruct_solution<QuadC,Basis>(cmesh, U_n, total_time);
//     constexpr auto Qpoints = QuadC::get_points();
//     constexpr auto Qweights = QuadC::get_weights();
//     for(uInt cellId=0;cellId<cmesh.m_cells.size();cellId++){
//         const auto& cell = cmesh.m_cells[cellId];
//         for(uInt xgId=0; xgId<QuadC::num_points; ++xgId) {
//             const auto& p = Qpoints[xgId];
//             const auto& pos = cell.transform_to_physical(p);
//             const auto& bUh = U_h[cellId].template SubMat<5,1>(5*xgId,0);
//             const auto& bUs = U_s[cellId].template SubMat<5,1>(5*xgId,0);
//             fp <<std::setprecision(16)<< pos[0] << "  " <<std::setprecision(16)<< pos[1] << "  " <<std::setprecision(16)<< pos[2]
//                 << "  " <<std::setprecision(16)<<  bUh[0] << "  " <<std::setprecision(16)<<  bUs[0] 
//                 << "  " <<std::setprecision(16)<<  bUh[1] << "  " <<std::setprecision(16)<<  bUs[1] 
//                 << "  " <<std::setprecision(16)<<  bUh[2] << "  " <<std::setprecision(16)<<  bUs[2]
//                 << "  " <<std::setprecision(16)<<  bUh[3] << "  " <<std::setprecision(16)<<  bUs[3] 
//                 << "  " <<std::setprecision(16)<<  bUh[4] << "  " <<std::setprecision(16)<<  bUs[4] << std::endl;
//         }
//     }
//     fp.close();
// }




template<typename QuadC, typename Basis>
void save_Uh(const ComputingMesh& cmesh, const LongVector<5*Basis::NumBasis>& U_n, const std::string filename){
    save_Uh<QuadC,Basis>(cmesh, U_n, 0.0, filename);
}


template void save_RTE_solution_to_hdf5<
    typename AutoQuadSelector<1, GaussLegendreTet::Auto>::type,
    typename AutoQuadSelector<1, GaussLegendreTri::Auto>::type,
    DGBasisEvaluator<1>, DGBasisEvaluator2D<1>, S2MeshIcosahedral>(
        const ComputingMesh&, 
        const LongVector<S2MeshIcosahedral::num_cells * DGBasisEvaluator<1>::NumBasis * DGBasisEvaluator2D<1>::NumBasis>&, 
        const std::string&, Scalar, uInt); 
template void save_RTE_solution_to_hdf5<
    typename AutoQuadSelector<2, GaussLegendreTet::Auto>::type,
    typename AutoQuadSelector<2, GaussLegendreTri::Auto>::type,
    DGBasisEvaluator<2>, DGBasisEvaluator2D<2>, S2MeshIcosahedral>(
        const ComputingMesh&, 
        const LongVector<S2MeshIcosahedral::num_cells * DGBasisEvaluator<2>::NumBasis * DGBasisEvaluator2D<2>::NumBasis>&, 
        const std::string&, Scalar, uInt); 

// #define explict_template_instantiation(QuadC, Basis) \
// template void save_Uh_Us<QuadC, Basis>(const ComputingMesh&, const LongVector<5 * Basis::NumBasis>&, Scalar, const std::string); \
// template void save_Uh<QuadC, Basis>(const ComputingMesh&, const LongVector<5 * Basis::NumBasis>&, Scalar, const std::string); \
// template void save_Uh<QuadC, Basis>(const ComputingMesh&, const LongVector<5 * Basis::NumBasis>&, const std::string); \
// template void save_DG_solution_to_hdf5<QuadC, Basis, 5>(const ComputingMesh&, const LongVector<5 * Basis::NumBasis>&, const std::string, Scalar, uInt); \
// template void save_DG_solution_to_hdf5<QuadC, Basis, 60>(const ComputingMesh&, const LongVector<60 * Basis::NumBasis>&, const std::string, Scalar, uInt); \
// template void save_DG_solution_to_hdf5<QuadC, Basis, 120>(const ComputingMesh&, const LongVector<120 * Basis::NumBasis>&, const std::string, Scalar, uInt); \
// template void save_DG_solution_to_hdf5<QuadC, Basis, 200>(const ComputingMesh&, const LongVector<200 * Basis::NumBasis>&, const std::string, Scalar, uInt); \
// template std::tuple<LongVector<5 * QuadC::num_points>, LongVector<5 * QuadC::num_points>, LongVector<5 * Basis::NumBasis>> \
// reconstruct_solution<QuadC, Basis>(const ComputingMesh&, const LongVector<5 * Basis::NumBasis>&, Scalar); \
// template LongVector<5 * QuadC::num_points> reconstruct_solution<QuadC, Basis>(const ComputingMesh&, const LongVector<5 * Basis::NumBasis>&);

#define explict_template_instantiation(QuadC, Basis) \
template void save_Uh_Us<QuadC, Basis>(const ComputingMesh&, const LongVector<5 * Basis::NumBasis>&, Scalar, const std::string); \
template void save_Uh<QuadC, Basis>(const ComputingMesh&, const LongVector<5 * Basis::NumBasis>&, Scalar, const std::string); \
template void save_Uh<QuadC, Basis>(const ComputingMesh&, const LongVector<5 * Basis::NumBasis>&, const std::string); \
template void save_DG_solution_to_hdf5<QuadC, Basis, 5>(const ComputingMesh&, const LongVector<5 * Basis::NumBasis>&, const std::string, Scalar, uInt); \
template std::tuple<LongVector<5 * QuadC::num_points>, LongVector<5 * QuadC::num_points>, LongVector<5 * Basis::NumBasis>> \
reconstruct_solution<QuadC, Basis>(const ComputingMesh&, const LongVector<5 * Basis::NumBasis>&, Scalar); \
template LongVector<5 * QuadC::num_points> reconstruct_solution<QuadC, Basis>(const ComputingMesh&, const LongVector<5 * Basis::NumBasis>&);


explict_template_instantiation(AutoQuadHelper<1>::QuadC, AutoQuadHelper<1>::Basis)
explict_template_instantiation(AutoQuadHelper<2>::QuadC, AutoQuadHelper<2>::Basis)
explict_template_instantiation(AutoQuadHelper<3>::QuadC, AutoQuadHelper<3>::Basis)
explict_template_instantiation(AutoQuadHelper<4>::QuadC, AutoQuadHelper<4>::Basis)
explict_template_instantiation(AutoQuadHelper<5>::QuadC, AutoQuadHelper<5>::Basis)
#undef explict_template_instantiation






