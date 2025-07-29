#include "base/type.h"
#include "mesh/mesh.h"
#include "mesh/mesh_generator.h"
#include "matrix/matrix.h"
#include "dg/dg_basis/dg_basis.h"
// #include "dg/dg_schemes/explicit_convection.h"
// #include "dg/dg_schemes/explicit_diffusion.h"
// #include "dg/dg_schemes/implicit_convection.h"
// #include "dg/dg_schemes/implicit_diffusion.h"
// #include "EigenSolver/EigenSparseSolver.h"
// #include "implicit_convection.h"

#include "problem.h"
#include "cylinder_generator.h"


Scalar uniform(){
    return (Scalar)(std::rand())/RAND_MAX;
}







int main(int argc,char** argv){
    // DenseMatrix<5,1> U{uniform(),uniform(),uniform(),uniform(),uniform()};
    // const auto& flux = AirFlux::computeFlux(U);
    // const auto& jaco = AirFlux::computeJacobian(U);

    // OldImplicitConvection<3> conv;

    // const auto& flux_old = conv.assemble_FU(U);
    // const auto& jaco_old = conv.assemble_jacobian(U);
    // // debug(flux);
    // // debug(flux_old);
    // debug(flux-flux_old);
    // // debug(jaco);
    // // debug(jaco_old);
    // debug(jaco[0]-jaco_old[0]);
    // debug(jaco[1]-jaco_old[1]);
    // debug(jaco[2]-jaco_old[2]);


    // DenseMatrix<5,1> U1{uniform(),uniform(),uniform(),uniform(),uniform()};
    // DenseMatrix<5,1> U2{uniform(),uniform(),uniform(),uniform(),uniform()};
    // const auto& wave = AirFlux::computeWaveSpeed(U1,U2);
    // const auto& wave_old = conv.compute_max_wave_speed(U1,U2);
    // debug(wave);
    // debug(wave_old);
    // debug(wave-wave_old);



    // GeneralMesh mesh = OrthHexMesh({0.0, 0.0, 0.0},{1.0, 1.0, 1.0},{10,10,10});
    // mesh.split_hex5_scan();                                   
    // mesh.rebuild_cell_topology();                             
    // mesh.validate_mesh();                                     
    // ComputingMesh cmesh(mesh);

    // for(const auto& cell : cmesh.m_cells){
    //     const auto& J1 = DenseMatrix<3,3>(cell.compute_jacobian_mat()).inverse();
    //     const auto& J2 = cell.get_invJacMat();
    //     print((J1-J2).transpose().multiply(J1-J2).trace());
    // }




    MeshGenerator generator;
    
    // 1. 设置自定义细化标准
    generator.set_refinement_criteria(0.125, 0.1);
    

    auto custom_size_field = [&](Scalar x,Scalar y)->Scalar{return 0.025*(x*x+y*2);};
    // 2. 设置尺寸场函数
    generator.set_size_field(custom_size_field);
    
    // // 3. 生成L型区域
    // generator.generate_l_shape();

    
    generator.generate_domain(
        std::make_unique<CylinderGenerator>(vector2f{-4,-4},vector2f{6,4}));
    
    // 4. 拉伸为3D
    generator.extrude_to_3d(0.0316);

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



    // std::vector<uInt> cell_count(mesh_data.tetrahedra.size(),0);
    // uInt boundary_count = 0, internal_count = 0, bool_count = 0;
    // // 清空计数器
    // std::fill(cell_count.begin(), cell_count.end(), 0);
    // boundary_count = 0;
    // internal_count = 0;
    // bool_count = 0;
    // // 统计每个单元的邻接面数量
    // for (const auto& face : mesh_data.face_adjacency) {
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
    // for (const auto& b : mesh_data.is_boundary_face){
    //     bool_count += b;
    // }
    // for(const auto& cc : cell_count) print(cc);
    // print(boundary_count);
    // print(internal_count);
    // print(bool_count);


    std::vector<uInt> cell_count(dg_mesh.cells.size(),0);
    uInt boundary_count = 0, internal_count = 0, bool_count = 0;
    // 清空计数器
    std::fill(cell_count.begin(), cell_count.end(), 0);
    boundary_count = 0;
    internal_count = 0;
    bool_count = 0;
    // 统计每个单元的邻接面数量
    for (const auto& face : dg_mesh.face_cells) {
        size_t c1 = face[0];
        size_t c2 = face[1];

        if (c2 == size_t(-1)) {
            // 边界面：仅统计 c1 的邻接面数量
            cell_count[c1] += 1;
            boundary_count += 1;
        } else if (c1 == size_t(-1)) {
            // 边界面：仅统计 c2 的邻接面数量
            cell_count[c2] += 1;
            boundary_count += 1;
        } else {
            // 内部面：c1 和 c2 各统计一次
            cell_count[c1] += 1;
            cell_count[c2] += 1;
            internal_count += 1;
        }
    }
    for (const auto& b : dg_mesh.is_boundary_face){
        bool_count += b;
    }
    for(const auto& cc : cell_count) print(cc);
    print(boundary_count);
    print(internal_count);
    print(bool_count);

    // // bool_count = std::count(mesh_data.is_boundary_face.begin(), mesh_data.is_boundary_face.end(), true);
    // // print(bool_count);


    // // std::vector<uInt> face_adjacency_encoder(mesh_data.face_adjacency.size());
    // // for(uInt k = 0; k < face_adjacency_encoder.size(); k++){
    // //     uInt c1 = mesh_data.face_adjacency[k][0];
    // //     uInt c2 = mesh_data.face_adjacency[k][1];
    // //     face_adjacency_encoder[k] = 0xFFFFFFFF * ((c1<c2)?c1:c2) + ((c1<c2)?c2:c1);
    // // }

    // // std::sort(face_adjacency_encoder.begin(),face_adjacency_encoder.end());
    // // for(uInt k = 0; k < face_adjacency_encoder.size()-1; k++){
    // //     if(face_adjacency_encoder[k] == face_adjacency_encoder[k+1]){
    // //         print("有重复  " + std::to_string(face_adjacency_encoder[k]/0xFFFFFFFF)  +  "  " + std::to_string(face_adjacency_encoder[k]%0xFFFFFFFF));
    // //     }
    // // }

    // std::vector<uInt> faces_encoder(mesh_data.faces.size());
    // for(uInt k = 0; k < faces_encoder.size(); k++){
    //     std::sort(mesh_data.faces[k].begin(),mesh_data.faces[k].end());
    //     uInt v1 = mesh_data.faces[k][0];
    //     uInt v2 = mesh_data.faces[k][1];
    //     uInt v3 = mesh_data.faces[k][2];
    //     faces_encoder[k] = 0xFFFFF * (0xFFFFF * v1 + v2) + v3;
    // }

    // std::sort(faces_encoder.begin(),faces_encoder.end());
    // for(uInt k = 0; k < faces_encoder.size()-1; k++){
    //     if(faces_encoder[k] == faces_encoder[k+1]){
    //         print("有重复  " + std::to_string(faces_encoder[k]/0xFFFFFFFF)  +  "  " + std::to_string(faces_encoder[k]%0xFFFFFFFF/0xFFFF)  +  "  " + std::to_string(faces_encoder[k]%0xFFFF));
    //     }
    // }


    // // generator.check_adjacency();

    
    // generator.export_to_file("mesh.txt");
    // generator.export_to_vtk("mesh.vtk");

    return 0;
}