#include "base/io.h"
#include "base/exact.h"

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
void save_Uh(const ComputingMesh& cmesh, const LongVector<5*Basis::NumBasis>& U_n, Scalar total_time, const std::string filename) {
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




using Basis1 = DGBasisEvaluator<1>;
using QuadC1 = typename AutoQuadSelector<Basis1::OrderBasis, GaussLegendreTet::Auto>::type;
using Basis2 = DGBasisEvaluator<2>;
using QuadC2 = typename AutoQuadSelector<Basis2::OrderBasis, GaussLegendreTet::Auto>::type;
using Basis3 = DGBasisEvaluator<3>;
using QuadC3 = typename AutoQuadSelector<Basis3::OrderBasis, GaussLegendreTet::Auto>::type;
using Basis4 = DGBasisEvaluator<4>;
using QuadC4 = typename AutoQuadSelector<Basis4::OrderBasis, GaussLegendreTet::Auto>::type;
using Basis5 = DGBasisEvaluator<5>;
using QuadC5 = typename AutoQuadSelector<Basis5::OrderBasis, GaussLegendreTet::Auto>::type;

template void save_Uh_Us<QuadC1, Basis1>(const ComputingMesh&, const LongVector<5 * Basis1::NumBasis>&, Scalar, const std::string);
template void save_Uh<QuadC1, Basis1>(const ComputingMesh&, const LongVector<5 * Basis1::NumBasis>&, Scalar, const std::string);
template void save_Uh<QuadC1, Basis1>(const ComputingMesh&, const LongVector<5 * Basis1::NumBasis>&, const std::string);
template std::tuple<LongVector<5 * QuadC1::num_points>, LongVector<5 * QuadC1::num_points>, LongVector<5 * Basis1::NumBasis>>
reconstruct_solution<QuadC1, Basis1>(const ComputingMesh&, const LongVector<5 * Basis1::NumBasis>&, Scalar);

template void save_Uh_Us<QuadC2, Basis2>(const ComputingMesh&, const LongVector<5 * Basis2::NumBasis>&, Scalar, const std::string);
template void save_Uh<QuadC2, Basis2>(const ComputingMesh&, const LongVector<5 * Basis2::NumBasis>&, Scalar, const std::string);
template void save_Uh<QuadC2, Basis2>(const ComputingMesh&, const LongVector<5 * Basis2::NumBasis>&, const std::string);
template std::tuple<LongVector<5 * QuadC2::num_points>, LongVector<5 * QuadC2::num_points>, LongVector<5 * Basis2::NumBasis>>
reconstruct_solution<QuadC2, Basis2>(const ComputingMesh&, const LongVector<5 * Basis2::NumBasis>&, Scalar);

template void save_Uh_Us<QuadC3, Basis3>(const ComputingMesh&, const LongVector<5 * Basis3::NumBasis>&, Scalar, const std::string);
template void save_Uh<QuadC3, Basis3>(const ComputingMesh&, const LongVector<5 * Basis3::NumBasis>&, Scalar, const std::string);
template void save_Uh<QuadC3, Basis3>(const ComputingMesh&, const LongVector<5 * Basis3::NumBasis>&, const std::string);
template std::tuple<LongVector<5 * QuadC3::num_points>, LongVector<5 * QuadC3::num_points>, LongVector<5 * Basis3::NumBasis>>
reconstruct_solution<QuadC3, Basis3>(const ComputingMesh&, const LongVector<5 * Basis3::NumBasis>&, Scalar);

template void save_Uh_Us<QuadC4, Basis4>(const ComputingMesh&, const LongVector<5 * Basis4::NumBasis>&, Scalar, const std::string);
template void save_Uh<QuadC4, Basis4>(const ComputingMesh&, const LongVector<5 * Basis4::NumBasis>&, Scalar, const std::string);
template void save_Uh<QuadC4, Basis4>(const ComputingMesh&, const LongVector<5 * Basis4::NumBasis>&, const std::string);
template std::tuple<LongVector<5 * QuadC4::num_points>, LongVector<5 * QuadC4::num_points>, LongVector<5 * Basis4::NumBasis>>
reconstruct_solution<QuadC4, Basis4>(const ComputingMesh&, const LongVector<5 * Basis4::NumBasis>&, Scalar);

template void save_Uh_Us<QuadC5, Basis5>(const ComputingMesh&, const LongVector<5 * Basis5::NumBasis>&, Scalar, const std::string);
template void save_Uh<QuadC5, Basis5>(const ComputingMesh&, const LongVector<5 * Basis5::NumBasis>&, Scalar, const std::string);
template void save_Uh<QuadC5, Basis5>(const ComputingMesh&, const LongVector<5 * Basis5::NumBasis>&, const std::string);
template std::tuple<LongVector<5 * QuadC5::num_points>, LongVector<5 * QuadC5::num_points>, LongVector<5 * Basis5::NumBasis>>
reconstruct_solution<QuadC5, Basis5>(const ComputingMesh&, const LongVector<5 * Basis5::NumBasis>&, Scalar);


