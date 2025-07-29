#include "base/type.h"
#include "mesh/mesh.h"
#include "matrix/matrix.h"
#include "dg/dg_basis/dg_basis.h"
#include "mesh/mesh_generator.h"
#include "cylinder_generator.h"

ComputingMesh create_mesh(uInt N){
    MeshGenerator generator;
    Scalar h = 4.0/N;
    // 1. 设置自定义细化标准
    generator.set_refinement_criteria(0.2, h);
    
    auto custom_size_field = [&](Scalar x,Scalar y)->Scalar{return 0.025*(x*x+y*2);};
    // 2. 设置尺寸场函数
    generator.set_size_field(custom_size_field);

    // 3. 生成圆柱区域
    generator.generate_domain(
        std::make_unique<CylinderGenerator>(vector2f{-5,-8},vector2f{15,8},N));
    
    // 4. 拉伸为3D
    generator.extrude_to_3d(h*0.5);
    generator.tetrahedralize();
    
    
    // 5. 获取网格数据
    auto mesh_data = generator.get_mesh_data();

    // std::cout << "生成网格完成!" << std::endl;
    // std::cout << "顶点数: " << mesh_data.vertices.size() << std::endl;
    // std::cout << "面数: " << mesh_data.faces.size() << std::endl;
    // std::cout << "单元数: " << mesh_data.tetrahedra.size() << std::endl;
    // std::cout << "体邻居: " << mesh_data.cell_adjacency.size() << std::endl;
    // std::cout << "面邻居: " << mesh_data.face_adjacency.size() << std::endl;

    DGMesh dg_mesh = build_dg_mesh(mesh_data);
    std::cout << "Total Points: " << dg_mesh.points.size() << std::endl;
    std::cout << "Total Faces: " << dg_mesh.faces.size() << std::endl;
    std::cout << "Total Cells: " << dg_mesh.cells.size() << std::endl;

    // generator.export_to_vtk("mesh.vtk");
    // generator.export_to_file("mesh.txt");
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


template<typename QuadCInput, typename BasisInput,typename QuadCOutput, typename BasisOutput>
LongVector<5*BasisOutput::NumBasis> read_solution_file(const ComputingMesh& cmesh, const std::string& filename) {
    LongVector<5*QuadCInput::num_points> U_h(cmesh.m_cells.size());
    LongVector<5*QuadCInput::num_points> U_s(cmesh.m_cells.size());
    std::ifstream file(filename);
    std::string line;

    uInt cellId = 0, xgId = 0;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        Scalar x, y, z;
        iss >> x >> y >> z;
        std::cout << "Reading cell " << cellId << " at point (" << x << ", " << y << ", " << z << ")\n";
        // 这个位置读出来的数据，就是 单元 cellId 上，QuadC 的各个积分点处的值，共5个分量
        for (int i = 0; i < 5; ++i) iss >> U_h[cellId][5*xgId+i] >> U_s[cellId][5*xgId+i];
        xgId++;
        if (xgId >= QuadCInput::num_points) {
            xgId = 0;
            cellId++;
        }
    }
    file.close();
    print(U_h);
    std::array<std::array<Scalar,BasisInput::NumBasis>,QuadCInput::num_points> basis;
    for(uInt xgId=0; xgId<QuadCInput::num_points; ++xgId) {
        const auto& p = QuadCInput::points[xgId];
        basis[xgId] = BasisInput::eval_all(p[0],p[1],p[2]);
    }
    LongVector<5*BasisOutput::NumBasis> coef(cmesh.m_cells.size());
    for(uInt cellId=0; cellId<cmesh.m_cells.size(); cellId++){
        // 单元 cellId 上，QuadC 的各个积分点处的值，共5个分量
        const auto& value = U_h[cellId];
        // 考虑最小二乘的时候，sum_i (basis_i, basis_j) c_i = (value, basis_j) for all j
        // 再考虑到 正交性， (basis_i, basis_j) = 0 for all i neq j
        // 所以只剩下了，(basis_j, basis_j) c_j = (value, basis_j) for all j
        // (basis_j, basis_j) = sum_{xg} basis_j(xg) * basis_j(x_g)
        // (value,   basis_j) = sum_{xg} value(xg)   * basis_j(x_g)
        for(uInt k=0; k<BasisInput::NumBasis; k++){
            Scalar sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0,
                   sum4 = 0.0, sum = 0.0;
            for(uInt xgId=0; xgId<QuadCInput::num_points; ++xgId) {
                sum += basis[xgId][k] * basis[xgId][k] * QuadCInput::weights[xgId];
                sum0 += value[5*xgId+0] * basis[xgId][k] * QuadCInput::weights[xgId];
                sum1 += value[5*xgId+1] * basis[xgId][k] * QuadCInput::weights[xgId];
                sum2 += value[5*xgId+2] * basis[xgId][k] * QuadCInput::weights[xgId];
                sum3 += value[5*xgId+3] * basis[xgId][k] * QuadCInput::weights[xgId];
                sum4 += value[5*xgId+4] * basis[xgId][k] * QuadCInput::weights[xgId];
            }
            coef[cellId][5*k + 0] = sum0 / sum;
            coef[cellId][5*k + 1] = sum1 / sum;
            coef[cellId][5*k + 2] = sum2 / sum;
            coef[cellId][5*k + 3] = sum3 / sum;
            coef[cellId][5*k + 4] = sum4 / sum;
        }
    }
    return coef;
}
