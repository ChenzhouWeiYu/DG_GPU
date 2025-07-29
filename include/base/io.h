#pragma once
#include "base/type.h"
#include "mesh/mesh.h"
#include "dg/dg_basis/dg_basis.h"
#include "dg/dg_schemes/explicit_rte_gpu/s2_mesh_icosahedral.h"

template<typename QuadC, typename Basis>
std::tuple<LongVector<5*QuadC::num_points>, 
            LongVector<5*QuadC::num_points>, 
            LongVector<5*Basis::NumBasis>> 
reconstruct_solution(const ComputingMesh& cmesh, const LongVector<5*Basis::NumBasis>& coef, Scalar curr_time);

template<typename QuadC, typename Basis>
LongVector<5*QuadC::num_points>
reconstruct_solution(const ComputingMesh& cmesh, const LongVector<5*Basis::NumBasis>& coef);


template<typename QuadC, typename Basis>
void save_Uh(const ComputingMesh& cmesh, const LongVector<5*Basis::NumBasis>& U_n, Scalar total_time, const std::string filename);

template<typename QuadC, typename Basis>
void save_Uh_Us(const ComputingMesh& cmesh, const LongVector<5*Basis::NumBasis>& U_n, Scalar total_time, const std::string filename);

template<typename QuadC, typename Basis>
void save_Uh(const ComputingMesh& cmesh, const LongVector<5*Basis::NumBasis>& U_n, const std::string filename);


template<int Order>
struct AutoQuadHelper {
    using Basis = DGBasisEvaluator<Order>;
    using QuadC = typename AutoQuadSelector<Basis::OrderBasis, GaussLegendreTet::Auto>::type;
};

template<typename QuadC, typename Basis, uInt N_var = 5>
void save_DG_solution_to_hdf5(const ComputingMesh& cmesh,
    const LongVector<N_var * Basis::NumBasis>& coef,
    const std::string filename,
    Scalar time = 0.0, uInt step = 0);

template<typename X3QuadC, typename S2QuadC, typename X3Basis, typename S2Basis, typename S2Mesh>
void save_RTE_solution_to_hdf5(
    const ComputingMesh& cmesh,
    const LongVector<S2Mesh::num_cells * X3Basis::NumBasis * S2Basis::NumBasis>& coef,
    const std::string& filename,
    Scalar time = 0.0,
    uInt step = 0);


extern template void save_RTE_solution_to_hdf5<
    typename AutoQuadSelector<1, GaussLegendreTet::Auto>::type,
    typename AutoQuadSelector<1, GaussLegendreTri::Auto>::type,
    DGBasisEvaluator<1>, DGBasisEvaluator2D<1>, S2MeshIcosahedral>(
        const ComputingMesh&, 
        const LongVector<S2MeshIcosahedral::num_cells * DGBasisEvaluator<1>::NumBasis * DGBasisEvaluator2D<1>::NumBasis>&, 
        const std::string&, Scalar, uInt); 

extern template void save_RTE_solution_to_hdf5<
    typename AutoQuadSelector<2, GaussLegendreTet::Auto>::type,
    typename AutoQuadSelector<2, GaussLegendreTri::Auto>::type,
    DGBasisEvaluator<2>, DGBasisEvaluator2D<2>, S2MeshIcosahedral>(
        const ComputingMesh&, 
        const LongVector<S2MeshIcosahedral::num_cells * DGBasisEvaluator<2>::NumBasis * DGBasisEvaluator2D<2>::NumBasis>&, 
        const std::string&, Scalar, uInt); 

// #define explict_template_instantiation(QuadC, Basis) \
// extern template void save_Uh_Us<QuadC, Basis>(const ComputingMesh&, const LongVector<5 * Basis::NumBasis>&, Scalar, const std::string); \
// extern template void save_Uh<QuadC, Basis>(const ComputingMesh&, const LongVector<5 * Basis::NumBasis>&, Scalar, const std::string); \
// extern template void save_Uh<QuadC, Basis>(const ComputingMesh&, const LongVector<5 * Basis::NumBasis>&, const std::string); \
// extern template void save_DG_solution_to_hdf5<QuadC, Basis, 5>(const ComputingMesh&, const LongVector<5 * Basis::NumBasis>&, const std::string, Scalar, uInt); \
// extern template void save_DG_solution_to_hdf5<QuadC, Basis, 60>(const ComputingMesh&, const LongVector<60 * Basis::NumBasis>&, const std::string, Scalar, uInt); \
// extern template void save_DG_solution_to_hdf5<QuadC, Basis, 120>(const ComputingMesh&, const LongVector<120 * Basis::NumBasis>&, const std::string, Scalar, uInt); \
// extern template void save_DG_solution_to_hdf5<QuadC, Basis, 200>(const ComputingMesh&, const LongVector<200 * Basis::NumBasis>&, const std::string, Scalar, uInt); \
// extern template std::tuple<LongVector<5 * QuadC::num_points>, LongVector<5 * QuadC::num_points>, LongVector<5 * Basis::NumBasis>> \
// reconstruct_solution<QuadC, Basis>(const ComputingMesh&, const LongVector<5 * Basis::NumBasis>&, Scalar); \
// extern template LongVector<5 * QuadC::num_points> reconstruct_solution<QuadC, Basis>(const ComputingMesh&, const LongVector<5 * Basis::NumBasis>&);

#define explict_template_instantiation(QuadC, Basis) \
extern template void save_Uh_Us<QuadC, Basis>(const ComputingMesh&, const LongVector<5 * Basis::NumBasis>&, Scalar, const std::string); \
extern template void save_Uh<QuadC, Basis>(const ComputingMesh&, const LongVector<5 * Basis::NumBasis>&, Scalar, const std::string); \
extern template void save_Uh<QuadC, Basis>(const ComputingMesh&, const LongVector<5 * Basis::NumBasis>&, const std::string); \
extern template void save_DG_solution_to_hdf5<QuadC, Basis, 5>(const ComputingMesh&, const LongVector<5 * Basis::NumBasis>&, const std::string, Scalar, uInt); \
extern template std::tuple<LongVector<5 * QuadC::num_points>, LongVector<5 * QuadC::num_points>, LongVector<5 * Basis::NumBasis>> \
reconstruct_solution<QuadC, Basis>(const ComputingMesh&, const LongVector<5 * Basis::NumBasis>&, Scalar); \
extern template LongVector<5 * QuadC::num_points> reconstruct_solution<QuadC, Basis>(const ComputingMesh&, const LongVector<5 * Basis::NumBasis>&);


explict_template_instantiation(AutoQuadHelper<1>::QuadC, AutoQuadHelper<1>::Basis)
explict_template_instantiation(AutoQuadHelper<2>::QuadC, AutoQuadHelper<2>::Basis)
explict_template_instantiation(AutoQuadHelper<3>::QuadC, AutoQuadHelper<3>::Basis)
explict_template_instantiation(AutoQuadHelper<4>::QuadC, AutoQuadHelper<4>::Basis)
explict_template_instantiation(AutoQuadHelper<5>::QuadC, AutoQuadHelper<5>::Basis)
#undef explict_template_instantiation

// Note: The above macro is used to explicitly instantiate the templates for different QuadC and Basis types.
// This is necessary to ensure that the compiler generates the code for these specific combinations,
// which can then be linked correctly when used in different translation units.

#

// Explicit template instantiation declarations
// These are necessary to avoid multiple definitions in different translation units
// when the functions are used in multiple source files.
// extern template void save_Uh_Us<QuadC1, Basis1>(const ComputingMesh&, const LongVector<5 * Basis1::NumBasis>&, Scalar, const std::string);
// extern template void save_Uh<QuadC1, Basis1>(const ComputingMesh&, const LongVector<5 * Basis1::NumBasis>&, Scalar, const std::string);
// extern template void save_Uh<QuadC1, Basis1>(const ComputingMesh&, const LongVector<5 * Basis1::NumBasis>&, const std::string);
// extern template std::tuple<LongVector<5 * QuadC1::num_points>, LongVector<5 * QuadC1::num_points>, LongVector<5 * Basis1::NumBasis>>
// reconstruct_solution<QuadC1, Basis1>(const ComputingMesh&, const LongVector<5 * Basis1::NumBasis>&, Scalar);
// extern template LongVector<5 * QuadC1::num_points> reconstruct_solution<QuadC1, Basis1>(const ComputingMesh&, const LongVector<5 * Basis1::NumBasis>&);


// extern template void save_Uh_Us<QuadC2, Basis2>(const ComputingMesh&, const LongVector<5 * Basis2::NumBasis>&, Scalar, const std::string);
// extern template void save_Uh<QuadC2, Basis2>(const ComputingMesh&, const LongVector<5 * Basis2::NumBasis>&, Scalar, const std::string);
// extern template void save_Uh<QuadC2, Basis2>(const ComputingMesh&, const LongVector<5 * Basis2::NumBasis>&, const std::string);
// extern template std::tuple<LongVector<5 * QuadC2::num_points>, LongVector<5 * QuadC2::num_points>, LongVector<5 * Basis2::NumBasis>>
// reconstruct_solution<QuadC2, Basis2>(const ComputingMesh&, const LongVector<5 * Basis2::NumBasis>&, Scalar);
// extern template LongVector<5 * QuadC2::num_points> reconstruct_solution<QuadC2, Basis2>(const ComputingMesh&, const LongVector<5 * Basis2::NumBasis>&);

// extern template void save_Uh_Us<QuadC3, Basis3>(const ComputingMesh&, const LongVector<5 * Basis3::NumBasis>&, Scalar, const std::string);
// extern template void save_Uh<QuadC3, Basis3>(const ComputingMesh&, const LongVector<5 * Basis3::NumBasis>&, Scalar, const std::string);
// extern template void save_Uh<QuadC3, Basis3>(const ComputingMesh&, const LongVector<5 * Basis3::NumBasis>&, const std::string);
// extern template std::tuple<LongVector<5 * QuadC3::num_points>, LongVector<5 * QuadC3::num_points>, LongVector<5 * Basis3::NumBasis>>
// reconstruct_solution<QuadC3, Basis3>(const ComputingMesh&, const LongVector<5 * Basis3::NumBasis>&, Scalar);
// extern template LongVector<5 * QuadC1::num_points> reconstruct_solution<QuadC1, Basis1>(const ComputingMesh&, const LongVector<5 * Basis1::NumBasis>&);

// extern template void save_Uh_Us<QuadC4, Basis4>(const ComputingMesh&, const LongVector<5 * Basis4::NumBasis>&, Scalar, const std::string);
// extern template void save_Uh<QuadC4, Basis4>(const ComputingMesh&, const LongVector<5 * Basis4::NumBasis>&, Scalar, const std::string);
// extern template void save_Uh<QuadC4, Basis4>(const ComputingMesh&, const LongVector<5 * Basis4::NumBasis>&, const std::string);
// extern template std::tuple<LongVector<5 * QuadC4::num_points>, LongVector<5 * QuadC4::num_points>, LongVector<5 * Basis4::NumBasis>>
// reconstruct_solution<QuadC4, Basis4>(const ComputingMesh&, const LongVector<5 * Basis4::NumBasis>&, Scalar);
// extern template LongVector<5 * QuadC1::num_points> reconstruct_solution<QuadC1, Basis1>(const ComputingMesh&, const LongVector<5 * Basis1::NumBasis>&);

// extern template void save_Uh_Us<QuadC5, Basis5>(const ComputingMesh&, const LongVector<5 * Basis5::NumBasis>&, Scalar, const std::string);
// extern template void save_Uh<QuadC5, Basis5>(const ComputingMesh&, const LongVector<5 * Basis5::NumBasis>&, Scalar, const std::string);
// extern template void save_Uh<QuadC5, Basis5>(const ComputingMesh&, const LongVector<5 * Basis5::NumBasis>&, const std::string);
// extern template std::tuple<LongVector<5 * QuadC5::num_points>, LongVector<5 * QuadC5::num_points>, LongVector<5 * Basis5::NumBasis>>
// reconstruct_solution<QuadC5, Basis5>(const ComputingMesh&, const LongVector<5 * Basis5::NumBasis>&, Scalar);
// extern template LongVector<5 * QuadC1::num_points> reconstruct_solution<QuadC1, Basis1>(const ComputingMesh&, const LongVector<5 * Basis1::NumBasis>&);


// #include <fstream>
// #include <string>
// #include <iomanip>

// // 你已有的头文件
// #include "ComputingMesh.h"
// #include "DenseMatrix.h"
// #include "LongVector.h"

// 写入VTU文件

template<uInt Order, typename QuadC, typename Basis>
void export_vtu(const ComputingMesh& mesh, 
                const LongVector<5*Basis::NumBasis>& coef, 
                Scalar curr_time,
                const std::string& filename) 
{
    std::ofstream fout(filename);
    fout << "<?xml version=\"1.0\"?>\n";
    fout << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    fout << "  <UnstructuredGrid>\n";
    fout << "    <Piece NumberOfPoints=\"" << mesh.m_points.size() << "\" NumberOfCells=\"" << mesh.m_cells.size() << "\">\n";

    // 写入Points
    fout << "      <Points>\n";
    fout << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    fout << std::setprecision(16);
    for (const auto& p : mesh.m_points) {
        fout << p[0] << " " << p[1] << " " << p[2] << " ";
    }
    fout << "\n";
    fout << "        </DataArray>\n";
    fout << "      </Points>\n";

    // 写入Cells
    fout << "      <Cells>\n";
    fout << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    for (const auto& cell : mesh.m_cells) {
        fout << cell.m_nodes[0] << " " << cell.m_nodes[1] << " " << cell.m_nodes[2] << " " << cell.m_nodes[3] << " ";
    }
    fout << "\n";
    fout << "        </DataArray>\n";

    fout << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    for (uInt i = 1; i <= mesh.m_cells.size(); ++i) {
        fout << i * 4 << " ";
    }
    fout << "\n";
    fout << "        </DataArray>\n";

    fout << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (uInt i = 0; i < mesh.m_cells.size(); ++i) {
        fout << "10 "; // VTK_TETRA = 10
    }
    fout << "\n";
    fout << "        </DataArray>\n";
    fout << "      </Cells>\n";

    // 写入CellData (以每个单元中心点数据来做第一版)
    fout << "      <CellData Scalars=\"Solution\">\n";

    fout << "        <DataArray type=\"Float64\" Name=\"rho\" format=\"ascii\">\n";
    for (uInt cid = 0; cid < mesh.m_cells.size(); ++cid) {
        Scalar rho_sum = 0.0;
        for (uInt i = 0; i < Basis::NumBasis; ++i) {
            rho_sum += coef[cid](5*i+0, 0); // 取密度部分做简单均值
        }
        fout << rho_sum / Basis::NumBasis << " ";
    }
    fout << "\n        </DataArray>\n";

    fout << "        <DataArray type=\"Float64\" Name=\"E\" format=\"ascii\">\n";
    for (uInt cid = 0; cid < mesh.m_cells.size(); ++cid) {
        Scalar E_sum = 0.0;
        for (uInt i = 0; i < Basis::NumBasis; ++i) {
            E_sum += coef[cid](5*i+4, 0);
        }
        fout << E_sum / Basis::NumBasis << " ";
    }
    fout << "\n        </DataArray>\n";

    fout << "      </CellData>\n";
    fout << "    </Piece>\n";
    fout << "  </UnstructuredGrid>\n";
    fout << "</VTKFile>\n";
    fout.close();
}