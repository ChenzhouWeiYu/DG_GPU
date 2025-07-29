#include "base/type.h"
#include "mesh/mesh.h"
#include "mesh/cgal_mesh.h"
#include "matrix/matrix.h"
#include "dg/dg_basis/dg_basis.h"

inline ComputingMesh create_mesh(uInt N){
    // GeneralMesh mesh = OrthHexMesh({-1.1, -1.1, -1.1/N},{1.1, 1.1, 1.1/N},{N,N,1});
    // mesh.split_hex5_scan();                                   
    // mesh.rebuild_cell_topology();                             
    // mesh.validate_mesh();                                     
    // ComputingMesh cmesh(mesh);     

    Scalar h = 2.2/N;
    for(uInt ss = 2; ss < 10; ss++){
        for(uInt s = 4*ss; s < 4*ss+4; s++){
            Scalar r = 0.15 * h + 0.15 * 0.985 * h * (3-1) + 0.15 * std::pow(1.05,s-3) * 0.985 * h * (s-3); // 半径
            if (r > 1.055) {
                h = h * 1.05/r;
            }
        }
        if(h < 2.2/N ) break;
    }
    CGALMesh generator(0.3, h, h*0.5); // 设置长宽比为0.2，网格大小为h，厚度为0.5*h
    
    // 边界点
    std::vector<std::array<double, 2>> points = {{-1.1,-1.1},{1.1,-1.1},{1.1,1.1},{-1.1,1.1}};
    // 内部点
    std::vector<std::array<double, 2>> internal_points = {{0.0, 0.0}};

    // for(uInt s = 1; s < 10; s++){
    //     Scalar r = 0.3 * h + 0.3 * 0.866 * h * (s-1 < 4 ? s-1 : 3-1 + (s - 3) * (s - 3)); // 半径
    //     if (r > 1.0) continue; // 超出边界
    //     uInt num_points = (s < 4 ? 3 << s : (3 << 3) * (s - 3)); // 每个圆周上的点数
    //     // 生成内部点
    //     for(uInt k = 0; k < num_points; k++){
    //         Scalar x = 0.0 + r * std::cos(2.0 * M_PI * k / num_points);
    //         Scalar y = 0.0 + r * std::sin(2.0 * M_PI * k / num_points);
    //         internal_points.push_back({x,y});
    //         print(vector2f{{x,y}}); 
    //     }
    // }

    for(uInt s = 1; s < 4; s++){
        Scalar r = 0.15 * h + 0.15 * 0.985 * h * (s-1); // 半径
        // print(r);
        if (r > 1.0) continue; // 超出边界
        uInt num_points =  3 << s;  // 每个圆周上的点数
        // 生成内部点
        for(uInt k = 0; k < num_points; k++){
            Scalar x = 0.0 + r * std::cos(2.0 * M_PI * k / num_points);
            Scalar y = 0.0 + r * std::sin(2.0 * M_PI * k / num_points);
            internal_points.push_back({x,y});
            // print(vector2f{{x,y}}); 
        }
    }
    for(uInt s = 4; s < 8; s++){
        Scalar r = 0.15 * h + 0.15 * 0.985 * h * (3-1) + 0.15 * std::pow(1.05,s-3) * 0.985 * h * (s-3); // 半径
        // print(r);
        if (r > 1.0) continue; // 超出边界
        uInt num_points =  3 << 3;  // 每个圆周上的点数
        // 生成内部点
        for(uInt k = 0; k < num_points; k++){
            Scalar x = 0.0 + r * std::cos(2.0 * M_PI * (k+0.5*((s+1)%2)) / num_points);
            Scalar y = 0.0 + r * std::sin(2.0 * M_PI * (k+0.5*((s+1)%2)) / num_points);
            internal_points.push_back({x,y});
            // print(vector2f{{x,y}}); 
        }
    }

    for(uInt ss = 2; ss < 10; ss++){
        for(uInt s = 4*ss; s < 4*ss+4; s++){
            Scalar r = 0.15 * h + 0.15 * 0.985 * h * (3-1) + 0.15 * std::pow(1.05,s-3) * 0.985 * h * (s-3); // 半径
            // print(r);
            if (r > 1.055) continue; // 超出边界
            uInt num_points =  6 * 2 *(ss+1);  // 每个圆周上的点数
            // 生成内部点
            for(uInt k = 0; k < num_points; k++){
                Scalar x = 0.0 + r * std::cos(2.0 * M_PI * (k+0.5*((s+1)%2)) / num_points);
                Scalar y = 0.0 + r * std::sin(2.0 * M_PI * (k+0.5*((s+1)%2)) / num_points);
                internal_points.push_back({x,y});
                // print(vector2f{{x,y}}); 
            }
        }
    }

    internal_points.push_back({-1.1 + 0.5 * h, -1.1 + 0.5 * h});
    internal_points.push_back({ 1.1 - 0.5 * h, -1.1 + 0.5 * h});
    internal_points.push_back({ 1.1 - 0.5 * h,  1.1 - 0.5 * h});
    internal_points.push_back({-1.1 + 0.5 * h,  1.1 - 0.5 * h});
    
    

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
            if(std::abs(face.m_normal[2])>0.5 )          
                cmesh.m_boundaryTypes[faceId] = BoundaryType::Pseudo3DZ;
            else
                cmesh.m_boundaryTypes[faceId] = BoundaryType::Dirichlet;
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