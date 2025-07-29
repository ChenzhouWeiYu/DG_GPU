#include "base/type.h"
#include "mesh/mesh.h"
#include "mesh/cgal_mesh.h"
#include "matrix/matrix.h"
#include "dg/dg_basis/dg_basis.h"

ComputingMesh create_mesh(uInt N){
    // GeneralMesh mesh = OrthHexMesh({-1.1, -1.1, -1.1/N},{1.1, 1.1, 1.1/N},{N,N,1});
    // mesh.split_hex5_scan();                                   
    // mesh.rebuild_cell_topology();                             
    // mesh.validate_mesh();                                     
    // ComputingMesh cmesh(mesh);     

    Scalar h = 2.2/N;
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
        print(r);
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
        print(r);
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
            print(r);
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
// reconstruct_solution(const ComputingMesh& cmesh, LongVector<5*Basis::NumBasis>& coef, Scalar curr_time){
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
//         for(uInt xgId=0; xgId<QuadC::num_points; ++xgId) {
//             const auto& p = QuadC::points[xgId];
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