#include "base/type.h"
#include "mesh/mesh.h"
#include "matrix/matrix.h"
#include "dg/dg_basis/dg_basis.h"

ComputingMesh create_mesh(uInt N){
    MeshGenerator generator;
    Scalar h = 4.0/N;
    // 1. 设置自定义细化标准
    generator.set_refinement_criteria(0.125, h);
    
    auto custom_size_field = [&](Scalar x,Scalar y)->Scalar{return 0.025*(x*x+y*2);};
    // 2. 设置尺寸场函数
    generator.set_size_field(custom_size_field);
    
    // // 3. 生成L型区域
    // generator.generate_l_shape();
    // std::vector<std::array<double, 2>> points = {{-1,0},{1,0},{0,-4}};
    
    std::vector<std::array<double, 2>> points;
    std::vector<std::array<double, 2>> internal_points;
    for(uInt k = 0;k<3*N;k++){
        Scalar x = 1.0 - 0.25*h/3 * k;
        Scalar y = -h/3 * k;
        x += 1e-12 * (1 - (y/2+1) * (y/2+1));
        points.push_back({x,y});
        // std::cout << "(x,y) = (\t" << x << ",\t" << y << ")" << std::endl;
    }
    points.push_back({0,-4});
    // points.push_back({-1,0});
    for(uInt k = 0;k<3*N;k++){
        Scalar x =  - 1.0 + 0.25*h/3 * (3*N-1-k);
        Scalar y = -h/3 * (3*N-1-k);
        x -= 1e-12 * (1 - (y/2+1) * (y/2+1));
        points.push_back({x,y});
    }
    uInt NN = (3*N)/2;
    Scalar hh = 2.0/NN;
    for(uInt k = 1;k<NN;k++){
        Scalar x =  -1 + hh * k;
        Scalar y = 0;
        y += 1e-16 * (1 - x*x);
        points.push_back({x,y});
        internal_points.push_back({x,y-0.75*hh});
        if(k==1){
            for(uInt k = 0;k<(13*N)/5;k++){
                Scalar xx = x + 0.25*h/3 * k;
                Scalar yy = y-0.75*hh - h/3 * k;
                internal_points.push_back({xx,yy});
            }
        }
        if(k==NN-1){
            for(uInt k = 0;k<(13*N)/5;k++){
                Scalar xx = x - 0.25*h/3 * k;
                Scalar yy = y-0.75*hh - h/3 * k;
                internal_points.push_back({xx,yy});
            }
        }
    }


    
    generator.generate_convex_polygon(points,internal_points);
    // std::cout << "111" << std::endl;
    // generator.generate_domain(
    //     std::make_unique<CylinderGenerator>(vector2f{-4,-4},vector2f{6,4}));
    // generator.generate_domain(
    //     std::make_unique<Case046>());
    
    // 4. 拉伸为3D
    generator.extrude_to_3d(h*0.5);
    generator.tetrahedralize();
    
    
    // 5. 获取网格数据
    auto mesh_data = generator.get_mesh_data();

    std::cout << "生成网格完成!" << std::endl;
    std::cout << "顶点数: " << mesh_data.vertices.size() << std::endl;
    std::cout << "面数: " << mesh_data.faces.size() << std::endl;
    std::cout << "单元数: " << mesh_data.tetrahedra.size() << std::endl;
    std::cout << "体邻居: " << mesh_data.cell_adjacency.size() << std::endl;
    std::cout << "面邻居: " << mesh_data.face_adjacency.size() << std::endl;

    DGMesh dg_mesh = build_dg_mesh(mesh_data);
    std::cout << "Total Points: " << dg_mesh.points.size() << std::endl;
    std::cout << "Total Faces: " << dg_mesh.faces.size() << std::endl;
    std::cout << "Total Cells: " << dg_mesh.cells.size() << std::endl;

    // std::vector<uInt> cell_count(dg_mesh.cells.size(),0);
    // uInt boundary_count = 0, internal_count = 0, bool_count = 0;
    // // 清空计数器
    // std::fill(cell_count.begin(), cell_count.end(), 0);
    // boundary_count = 0;
    // internal_count = 0;
    // bool_count = 0;
    // // 统计每个单元的邻接面数量
    // for (const auto& face : dg_mesh.face_cells) {
    //     size_t c1 = face[0];
    //     size_t c2 = face[1];

    //     if (c2 == size_t(-1)) {
    //         // 边界面：仅统计 c1 的邻接面数量
    //         cell_count[c1] += 1;
    //         boundary_count += 1;
    //     } else if (c1 == size_t(-1)) {
    //         // 边界面：仅统计 c2 的邻接面数量
    //         cell_count[c2] += 1;
    //         boundary_count += 1;
    //     } else {
    //         // 内部面：c1 和 c2 各统计一次
    //         cell_count[c1] += 1;
    //         cell_count[c2] += 1;
    //         internal_count += 1;
    //     }
    // }
    // for (const auto& b : dg_mesh.is_boundary_face){
    //     bool_count += b;
    // }
    // for(const auto& cc : cell_count) print(cc);
    // print(boundary_count);
    // print(internal_count);
    // print(bool_count);
    generator.export_to_vtk("mesh.vtk");
    generator.export_to_file("mesh.txt");
    export_dgmesh_to_vtk(dg_mesh,"dgmesh.vtk");


    ComputingMesh cmesh(dg_mesh);                                
    cmesh.m_boundaryTypes.resize(cmesh.m_faces.size());                   
    for(uInt faceId=0;faceId<cmesh.m_faces.size();faceId++){           
        if(cmesh.m_faces[faceId].m_neighbor_cells[1]==uInt(-1)){ 
            const auto& face = cmesh.m_faces[faceId];           
            if(std::abs(face.m_normal[2])>0.5 )          
                cmesh.m_boundaryTypes[faceId] = BoundaryType::Pseudo3DZ;
            else
                cmesh.m_boundaryTypes[faceId] = BoundaryType::WallTN;
        }
    }
    return cmesh;
}

template<typename QuadC, typename Basis>
std::tuple<LongVector<5*QuadC::num_points>, 
            LongVector<5*QuadC::num_points>, 
            LongVector<5*Basis::NumBasis>> 
reconstruct_solution(const ComputingMesh& cmesh, LongVector<5*Basis::NumBasis>& coef, Scalar curr_time){
    LongVector<5*QuadC::num_points> U_h(cmesh.m_cells.size());
    LongVector<5*QuadC::num_points> U_s(cmesh.m_cells.size());
    LongVector<5*Basis::NumBasis> error(cmesh.m_cells.size());

    #pragma omp parallel for schedule(dynamic)
    for(uInt cellId=0;cellId<cmesh.m_cells.size();cellId++){
        const auto& cell = cmesh.m_cells[cellId];
        const auto& U_func = [&](vector3f Xi)->DenseMatrix<5,1>{
            return U_Xi(cell,Xi,curr_time);
        };
        const auto& U_coef = Basis::func2coef(U_func);
        for(uInt k=0;k<Basis::NumBasis;k++){
            const auto& block_coef = coef[cellId].template SubMat<5,1>(5*k,0) - U_coef[k];
            MatrixView<5*Basis::NumBasis,1,5,1>(error[cellId],5*k,0) = block_coef;
        }
        for(uInt xgId=0; xgId<QuadC::num_points; ++xgId) {
            const auto& p = QuadC::points[xgId];
            const auto& pos = cell.transform_to_physical(p);
            const auto& value = Basis::eval_all(p[0],p[1],p[2]);
            const auto& U = Basis::template coef2filed<5,Scalar>(coef[cellId],p);
            MatrixView<5*QuadC::num_points,1,5,1> block_U_h(U_h[cellId],5*xgId,0);
            MatrixView<5*QuadC::num_points,1,5,1> block_U_s(U_s[cellId],5*xgId,0);
            
            block_U_h = U;
            block_U_h[0] = U[0];
            block_U_s = U_func(p);
        }
    }
    return {U_h, U_s, error};

}