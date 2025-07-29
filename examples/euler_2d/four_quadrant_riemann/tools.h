#pragma once
#include "base/type.h"
#include "mesh/mesh.h"
#include "mesh/cgal_mesh.h"
#include "matrix/matrix.h"
#include "dg/dg_basis/dg_basis.h"

inline ComputingMesh create_mesh(uInt N){
    Scalar h = 1.0/(7*N);

    // uInt Nx = 7*N;
    // uInt Ny = 7*N;
    CGALMesh generator(0.20, h, 0.707*h); // 设置长宽比为0.2，网格大小为h，厚度为0.5*h
    
    constexpr vector2f lb = {0,0};
    constexpr vector2f ub = {1,1};

    std::vector<std::array<double, 2>> points = {{lb[0], lb[1]}, {ub[0], ub[1]}, {lb[0], ub[1]}};
    std::vector<std::array<double, 2>> internal_points;
    // constexpr auto Qpoints = GaussLegendreTri::Degree25Points117::get_points();
    // for(const auto p : Qpoints){
    //     uInt NN = (N+1)/2;
    //     Scalar hh = 1.0/NN;
    //     for(uInt kk = 0; kk < NN; kk++){
    //         internal_points.push_back({hh * kk + hh * p[0], hh * kk + hh * (1-p[1])});
    //     }
    //     for(uInt kk = 0; kk < NN-1; kk++){
    //         internal_points.push_back({hh * kk + hh * (1-p[0]) + hh, hh * kk + hh * p[1]});
    //     }
    // }
    uInt NN = 1;
    for(NN = 16; NN < (7*N)<<1; NN<<=1);
    Scalar hh = 1.0/NN;
    // for(uInt kk = 4; kk < NN-3; kk++){
    //     internal_points.push_back({hh * kk, hh * kk});
    //     internal_points.push_back({hh * kk - hh, hh * kk + hh});
    //     internal_points.push_back({hh * kk - hh * 2, hh * kk + hh * 2});
    //     internal_points.push_back({hh * kk - hh * 3, hh * kk + hh * 3});
    // }
    Scalar dx1 = hh * M_SQRT2 * std::cos(105.0/180.0 * M_PI);
    Scalar dy1 = hh * M_SQRT2 * std::sin(105.0/180.0 * M_PI);
    Scalar dx2 = hh * std::cos(-120.0/180.0 * M_PI);
    Scalar dy2 = hh * std::sin(-120.0/180.0 * M_PI);
    Scalar dx3 = hh * std::cos(30.0/180.0 * M_PI);
    Scalar dy3 = hh * std::sin(30.0/180.0 * M_PI);
    for(uInt kk = 1; kk < NN; kk++){
        for(uInt kkk = 0; kkk < uInt(2.0/hh); kkk++){
            Scalar x = hh * kk + dx1 * kkk;
            Scalar y = hh * kk + dy1 * kkk;
            if((x > 0 + hh*3 && y < 1 - hh*3) || kkk == 0){
                internal_points.push_back({x,y});
            }
        }
        for(uInt kkk = 0; kkk < 3; kkk++){
            Scalar x = hh * kk + dx2 * kkk;
            Scalar y = 1.0     + dy2 * kkk;
            if((y-x>hh && x+y>1+hh/2) || kkk == 0){
                internal_points.push_back({x,y});
            }
        }
        for(uInt kkk = 0; kkk < 3; kkk++){
            Scalar x = 0.0     + dx3 * kkk;
            Scalar y = hh * kk + dy3 * kkk;
            if((y-x>hh && x+y<1-hh/2) || kkk == 0){
                internal_points.push_back({x,y});
            }
        }
    }

    // std::vector<std::array<double, 2>> points = {{lb[0], lb[1]}, {ub[0], lb[1]}, {ub[0], ub[1]}, {lb[0], ub[1]}};
    // std::vector<std::array<double, 2>> internal_points;
    // for(uInt nx = 1; nx < Nx; nx++){
    //     {
    //         uInt ny = nx;
    //         Scalar x = lb[0] + (ub[0] - lb[0]) * nx / Nx;
    //         Scalar y = lb[1] + (ub[1] - lb[1]) * ny / Ny;
    //         internal_points.push_back({x, y});
    //     }
    //     {
    //         uInt ny = Nx-nx;
    //         if (ny == nx) continue;
    //         Scalar x = lb[0] + (ub[0] - lb[0]) * nx / Nx;
    //         Scalar y = lb[1] + (ub[1] - lb[1]) * ny / Ny;
    //         internal_points.push_back({x, y});
    //     }
    // }

    // 边界点
    // for(uInt nx = 0; nx < Nx; nx++){
    //     Scalar x = lb[0] + (ub[0] - lb[0]) * nx / Nx;
    //     points.push_back({x, lb[1]});
    // }
    // for(uInt ny = 0; ny < Ny; ny++){
    //     Scalar y = lb[1] + (ub[1] - lb[1]) * ny / Ny;
    //     points.push_back({ub[0], y});
    // }
    // for(uInt nx = Nx; nx > 0; nx--){
    //     Scalar x = lb[0] + (ub[0] - lb[0]) * nx / Nx;
    //     points.push_back({x, ub[1]});
    // }
    // for(uInt ny = Ny; ny > 0; ny--){
    //     Scalar y = lb[1] + (ub[1] - lb[1]) * ny / Ny;
    //     points.push_back({lb[0], y});
    // }
    
    // 内部点
    // for(uInt nx = 1; nx < Nx; nx++){
    //     for(uInt ny = 1; ny < Ny; ny++){
    //         Scalar x = lb[0] + (ub[0] - lb[0]) * nx / Nx;
    //         Scalar y = lb[1] + (ub[1] - lb[1]) * ny / Ny;
    //         internal_points.push_back({x, y});
    //     }
    // }
    // for(uInt nx = 0; nx < Nx; nx++){
    //     // for(uInt ny = 0; ny < Ny; ny++){
    //         // 偏移半格
    //         uInt ny = nx;
    //         if ((nx - ny)%2==0){
    //             Scalar x = lb[0] + (ub[0] - lb[0]) * (nx + 0.5) / Nx;
    //             Scalar y = lb[1] + (ub[1] - lb[1]) * (ny + 0.5) / Ny;
    //             internal_points.push_back({x, y});
    //         }
    //     // }
    // }


    // // 偏移半格局部加密
    // auto dist2 = [&](Scalar x, Scalar y, Scalar x0, Scalar y0) -> Scalar {
    //     // 这里可以根据需要定义哪些点需要局部加密
    //     return (x - x0) * (x - x0) + (y - y0) * (y - y0);
    // };
    // auto near_curve = [&](Scalar x, Scalar y) -> Scalar {
    //     Scalar val = -0.2034*x*x + 0.0597*x*y + -0.7398*y*y + 0.5771*x + -0.2638*y + -0.0718;
    //     return std::abs(val);
    // };
    // auto near_init = [&](Scalar x, Scalar y) -> Scalar {
    //     Scalar val = y - 1.732*(x-0.1667);
    //     return std::abs(val);
    // };
    // auto is_refined_1 = [&](Scalar x, Scalar y) -> bool {
    //     // 这里可以根据需要定义哪些点需要局部加密
    //     return (x>0.45 && y>0.45 && x<0.9 && y<0.9);
    // };
    // auto is_refined_2 = [&](Scalar x, Scalar y) -> bool {
    //     // 这里可以根据需要定义哪些点需要局部加密
    //     // return false;
    //     return (x>0.575 && y>0.575 && x<0.875 && y<0.875);
    // };
    // auto is_refined_3 = [&](Scalar x, Scalar y) -> bool {
    //     // 这里可以根据需要定义哪些点需要局部加密
    //     // return false;
    //     return (x>0.675 && y>0.675 && x<0.850 && y<0.850);
    // };
    // auto is_refined_4 = [&](Scalar x, Scalar y) -> bool {
    //     // 这里可以根据需要定义哪些点需要局部加密
    //     // return false;
    //     return (x>0.75 && y>0.75 && x<0.825 && y<0.825);
    // };
    // for(uInt nx = 0; nx < Nx; nx++){
    //     for(uInt ny = 0; ny < Ny; ny++){
    //         // 偏移半格
    //         Scalar x = lb[0] + (ub[0] - lb[0]) * (nx + 0.5) / Nx;
    //         Scalar y = lb[1] + (ub[1] - lb[1]) * (ny + 0.5) / Ny;
    //         Scalar hx = (ub[0] - lb[0]) / Nx;
    //         Scalar hy = (ub[1] - lb[1]) / Ny;
    //         if((is_refined_4(x, y))){
    //             for(uInt nnx = 0; nnx < 5; nnx ++){
    //                 for(uInt nny = 0; nny < 5; nny ++){
    //                     if ( (nnx==0 || nnx==5) && (nny==0 || nny==5) ) continue;
    //                     internal_points.push_back({x - hx/2.0 + hx/5.0 * nnx, y - hy/2.0 + hy/5.0 * nny});
    //                 }
    //             }
    //         }
    //         else if((is_refined_3(x, y))){
    //             for(uInt nnx = 0; nnx < 4; nnx ++){
    //                 for(uInt nny = 0; nny < 4; nny ++){
    //                     if ( (nnx==0 || nnx==4) && (nny==0 || nny==4) ) continue;
    //                     internal_points.push_back({x - hx/2.0 + hx/4.0 * nnx, y - hy/2.0 + hy/4.0 * nny});
    //                 }
    //             }
    //         }
    //         else if (is_refined_2(x, y)) {
    //             internal_points.push_back({x - hx/6.0, y - hy/6.0});
    //             internal_points.push_back({x + hx/6.0, y - hy/6.0});
    //             internal_points.push_back({x - hx/6.0, y + hy/6.0});
    //             internal_points.push_back({x + hx/6.0, y + hy/6.0});
    //             // 如果是二次加密
    //             if(!is_refined_3(x - hx, y) && x!= lb[0]){
    //                 internal_points.push_back({x - hx/2.0, y - hy/6.0});
    //                 internal_points.push_back({x - hx/2.0, y + hy/6.0});
    //             }
    //             if(!is_refined_3(x, y - hy) && y!= lb[0]){
    //                 internal_points.push_back({x - hx/6.0, y - hy/2.0});
    //                 internal_points.push_back({x + hx/6.0, y - hy/2.0});
    //             }
    //         }else if (is_refined_1(x, y)) {
    //             internal_points.push_back({x, y});
    //             // 只考虑左侧和下侧，不然重复了
    //             if(is_refined_2(x - hx, y) && x!= lb[0]){
    //                 internal_points.push_back({x - hx/2.0, y - hy/6.0});
    //                 internal_points.push_back({x - hx/2.0, y + hy/6.0});
    //             }
    //             else{
    //                 internal_points.push_back({x - hx/2.0, y});
    //             }
                
    //             if(is_refined_2(x, y - hy) && y!= lb[0]){
    //                 internal_points.push_back({x - hx/6.0, y - hy/2.0});
    //                 internal_points.push_back({x + hx/6.0, y - hy/2.0});
    //             }
    //             else{
    //                 internal_points.push_back({x, y - hy/2.0});
    //             }
                
    //         }
    //         else
    //         {
    //             if(is_refined_2(x - hx, y) && x!= lb[0]){
    //                 internal_points.push_back({x - hx/2.0, y - hy/6.0});
    //                 internal_points.push_back({x - hx/2.0, y + hy/6.0});
    //             }else if(is_refined_1(x - hx, y) && x!= lb[0]){
    //                 internal_points.push_back({x - hx/2.0, y});
    //             }
                
    //             if(is_refined_2(x, y - hy) && y!= lb[0]){
    //                 internal_points.push_back({x - hx/6.0, y - hy/2.0});
    //                 internal_points.push_back({x + hx/6.0, y - hy/2.0});
    //             }else if(is_refined_1(x, y - hy) && y!= lb[0]){
    //                 internal_points.push_back({x, y - hy/2.0});
    //             }
                
    //         }
    //     }
    // }


    generator.generate_2d_mesh(points,internal_points);

    DGMesh dg_mesh = generator.get_dg_mesh();
    generator.export_dgmesh_to_vtk("dg_mesh.vtk");
    std::cout << "Total Points: " << dg_mesh.points.size() << std::endl;
    std::cout << "Total Faces: " << dg_mesh.faces.size() << std::endl;
    std::cout << "Total Cells: " << dg_mesh.cells.size() << std::endl;

    ComputingMesh cmesh(dg_mesh);                                
    cmesh.m_boundaryTypes.resize(cmesh.m_faces.size());                   
    for(uInt faceId=0;faceId<cmesh.m_faces.size();faceId++){           
        if(cmesh.m_faces[faceId].m_neighbor_cells[1]==uInt(-1)){ 
            const auto& face = cmesh.m_faces[faceId];           
            if(std::abs(face.m_normal[2])>0.9 )          
                cmesh.m_boundaryTypes[faceId] = BoundaryType::Pseudo3DZ;
            else if(std::abs(face.m_normal[2])<1e-4){
                if(std::abs(face.m_normal[0]+face.m_normal[1])<1e-4){
                    cmesh.m_boundaryTypes[faceId] = BoundaryType::Symmetry;
                }
                else{
                    cmesh.m_boundaryTypes[faceId] = BoundaryType::Dirichlet;
                    // cmesh.m_boundaryTypes[faceId] = BoundaryType::Neumann;
                }
                
                // cmesh.m_boundaryTypes[faceId] = BoundaryType::Neumann;
                // const auto& nodes = face.m_nodes;
                // const auto& p0 = cmesh.m_points[nodes[0]];
                // const auto& p1 = cmesh.m_points[nodes[1]];
                // const auto& p2 = cmesh.m_points[nodes[2]];
                // const Scalar x = (p0[0] + p1[0] + p2[0]) / 3.0;
                // const Scalar y = (p0[1] + p1[1] + p2[1]) / 3.0;
                // // print(centor);
                // if( face.m_normal[0]<-0.99 ){
                //     cmesh.m_boundaryTypes[faceId] = BoundaryType::Dirichlet;
                // }
                // else
                // {
                //     cmesh.m_boundaryTypes[faceId] = BoundaryType::Neumann;
                // }
            }
                
        }
    }
    return cmesh;
}









// template<typename QuadC, typename Basis>
// std::tuple<LongVector<5*QuadC::num_points>, 
//             LongVector<5*QuadC::num_points>, 
//             LongVector<5*Basis::NumBasis>> 
// reconstruct_solution(const ComputingMesh& cmesh, const LongVector<5*Basis::NumBasis>& coef, Scalar curr_time);

// template<typename QuadC, typename Basis>
// void save_Uh(const ComputingMesh& cmesh, const LongVector<5*Basis::NumBasis>& U_n, Scalar total_time, const std::string filename);

// template<typename QuadC, typename Basis>
// void save_Uh_Us(const ComputingMesh& cmesh, const LongVector<5*Basis::NumBasis>& U_n, Scalar total_time, const std::string filename);

// template<typename QuadC, typename Basis>
// void save_Uh(const ComputingMesh& cmesh, const LongVector<5*Basis::NumBasis>& U_n, const std::string filename);


// using Basis1 = DGBasisEvaluator<1>;
// using QuadC1 = typename AutoQuadSelector<Basis1::OrderBasis, GaussLegendreTet::Auto>::type;
// using Basis2 = DGBasisEvaluator<2>;
// using QuadC2 = typename AutoQuadSelector<Basis2::OrderBasis, GaussLegendreTet::Auto>::type;
// using Basis3 = DGBasisEvaluator<3>;
// using QuadC3 = typename AutoQuadSelector<Basis3::OrderBasis, GaussLegendreTet::Auto>::type;
// using Basis4 = DGBasisEvaluator<4>;
// using QuadC4 = typename AutoQuadSelector<Basis4::OrderBasis, GaussLegendreTet::Auto>::type;
// using Basis5 = DGBasisEvaluator<5>;
// using QuadC5 = typename AutoQuadSelector<Basis5::OrderBasis, GaussLegendreTet::Auto>::type;

// extern template void save_Uh_Us<QuadC1, Basis1>(const ComputingMesh&, const LongVector<5 * Basis1::NumBasis>&, Scalar, const std::string);
// extern template void save_Uh<QuadC1, Basis1>(const ComputingMesh&, const LongVector<5 * Basis1::NumBasis>&, Scalar, const std::string);
// extern template void save_Uh<QuadC1, Basis1>(const ComputingMesh&, const LongVector<5 * Basis1::NumBasis>&, const std::string);
// extern template std::tuple<LongVector<5 * QuadC1::num_points>, LongVector<5 * QuadC1::num_points>, LongVector<5 * Basis1::NumBasis>>
// reconstruct_solution<QuadC1, Basis1>(const ComputingMesh&, const LongVector<5 * Basis1::NumBasis>&, Scalar);

// extern template void save_Uh_Us<QuadC2, Basis2>(const ComputingMesh&, const LongVector<5 * Basis2::NumBasis>&, Scalar, const std::string);
// extern template void save_Uh<QuadC2, Basis2>(const ComputingMesh&, const LongVector<5 * Basis2::NumBasis>&, Scalar, const std::string);
// extern template void save_Uh<QuadC2, Basis2>(const ComputingMesh&, const LongVector<5 * Basis2::NumBasis>&, const std::string);
// extern template std::tuple<LongVector<5 * QuadC2::num_points>, LongVector<5 * QuadC2::num_points>, LongVector<5 * Basis2::NumBasis>>
// reconstruct_solution<QuadC2, Basis2>(const ComputingMesh&, const LongVector<5 * Basis2::NumBasis>&, Scalar);

// extern template void save_Uh_Us<QuadC3, Basis3>(const ComputingMesh&, const LongVector<5 * Basis3::NumBasis>&, Scalar, const std::string);
// extern template void save_Uh<QuadC3, Basis3>(const ComputingMesh&, const LongVector<5 * Basis3::NumBasis>&, Scalar, const std::string);
// extern template void save_Uh<QuadC3, Basis3>(const ComputingMesh&, const LongVector<5 * Basis3::NumBasis>&, const std::string);
// extern template std::tuple<LongVector<5 * QuadC3::num_points>, LongVector<5 * QuadC3::num_points>, LongVector<5 * Basis3::NumBasis>>
// reconstruct_solution<QuadC3, Basis3>(const ComputingMesh&, const LongVector<5 * Basis3::NumBasis>&, Scalar);

// extern template void save_Uh_Us<QuadC4, Basis4>(const ComputingMesh&, const LongVector<5 * Basis4::NumBasis>&, Scalar, const std::string);
// extern template void save_Uh<QuadC4, Basis4>(const ComputingMesh&, const LongVector<5 * Basis4::NumBasis>&, Scalar, const std::string);
// extern template void save_Uh<QuadC4, Basis4>(const ComputingMesh&, const LongVector<5 * Basis4::NumBasis>&, const std::string);
// extern template std::tuple<LongVector<5 * QuadC4::num_points>, LongVector<5 * QuadC4::num_points>, LongVector<5 * Basis4::NumBasis>>
// reconstruct_solution<QuadC4, Basis4>(const ComputingMesh&, const LongVector<5 * Basis4::NumBasis>&, Scalar);

// extern template void save_Uh_Us<QuadC5, Basis5>(const ComputingMesh&, const LongVector<5 * Basis5::NumBasis>&, Scalar, const std::string);
// extern template void save_Uh<QuadC5, Basis5>(const ComputingMesh&, const LongVector<5 * Basis5::NumBasis>&, Scalar, const std::string);
// extern template void save_Uh<QuadC5, Basis5>(const ComputingMesh&, const LongVector<5 * Basis5::NumBasis>&, const std::string);
// extern template std::tuple<LongVector<5 * QuadC5::num_points>, LongVector<5 * QuadC5::num_points>, LongVector<5 * Basis5::NumBasis>>
// reconstruct_solution<QuadC5, Basis5>(const ComputingMesh&, const LongVector<5 * Basis5::NumBasis>&, Scalar);