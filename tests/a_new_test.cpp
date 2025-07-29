#include <fstream>
#include <sstream>
#include <memory>
#include "Type.h"
#include <chrono>
#include "mesh/mesh.h"
#include "matrix/matrix.h"
#include "DG_Schemes/ExplicitConvection.h"
#include "DG_Schemes/ImplicitConvection.h"

int main(int argc, char** argv){
    uInt N=5;
    GeneralMesh new_mesh = OrthHexMesh({0,0,0},{1,1,1},{N,N,N});
    
    // HexMesh mesh = HexMesh({0,0,0},{1,1,1},{N,N,N});
    // GmshMesh mesh = GmshMesh("./case11.msh");

    // new_mesh.m_points.resize(mesh.points.size());
    // new_mesh.m_faces.resize(mesh.faces.size());
    // new_mesh.m_cells.resize(mesh.cells.size());

    // for(uInt p_id=0;p_id<mesh.points.size();p_id++){
    //     new_mesh.m_points[p_id] = mesh.points[p_id];
    // }

    // for(uInt f_id=0;f_id<mesh.faces.size();f_id++){
    //     switch (mesh.face2node[f_id].size())
    //     {
    //     case 3:{
    //         auto& nodes = mesh.face2node[f_id];
    //         TriangleFace Tface;
    //         Tface.nodes = {nodes[0],nodes[1],nodes[2]};
    //         Tface.neighbor_cells = {mesh.face2cell[f_id][0],mesh.face2cell[f_id][1]?mesh.face2cell[f_id][1]:uInt(-1)};
    //         new_mesh.m_faces[f_id] = Tface;
    //         break;
    //     }
    //     case 4:{
    //         auto& nodes = mesh.face2node[f_id];
    //         QuadFace Qface;
    //         Qface.nodes = {nodes[0],nodes[1],nodes[2],nodes[3]};
    //         Qface.neighbor_cells = {mesh.face2cell[f_id][0],mesh.face2cell[f_id][1]?mesh.face2cell[f_id][1]:uInt(-1)};
    //         new_mesh.m_faces[f_id] = Qface;
    //         break;
    //     }
    //     default:
    //         break;
    //     }
        
    // }

    // for(uInt c_id=0;c_id<mesh.cells.size();c_id++){
    //     switch (mesh.cells[c_id])
    //     {
    //     case 4:{
    //         Hexahedron hex;
    //         new_mesh.m_cells[c_id] = hex;
    //         break;
    //     }
    //     case 6:{
    //         Prism prism;
    //         new_mesh.m_cells[c_id] = prism;
    //     }
    //     default:
    //         break;
    //     }
        
    // }
    // // std::cout<<"11"<<std::endl;
    // // for(auto&fc:mesh.face2cell){
    // //     for(auto&c:fc){
    // //         std::cout<<c<<"\t";
    // //     }
    // //     std::cout<<std::endl;
    // // }
    // // for(auto&f:new_mesh.m_faces){
    // //     std::visit([&](auto&& face) {
    // //         for (uInt cell_id : face.neighbor_cells) {
    // //             std::cout<<cell_id<<"\t";
    // //         }
    // //         std::cout<<std::endl;
    // //     }, f);
    // // }
    // new_mesh.rebuild_cell_topology();
    // // std::cout<<"1111"<<std::endl;
    // // std::cout<<"11"<<std::endl;
    // new_mesh.validate_mesh();
    // // std::cout<<"22"<<std::endl;

    // for(auto&c:new_mesh.m_cells){
    //     std::visit([&](auto&& cell) {for(auto n:cell.nodes)std::cout<<n<<"\t";std::cout<<std::endl;}, c);
    // }

    
    std::cout << "sizeof TriangleFace  \t" << sizeof(TriangleFace) << std::endl;
    std::cout << "sizeof QuadFace      \t" << sizeof(QuadFace) << std::endl;
    std::cout << "sizeof GeneralFace   \t" << sizeof(GeneralFace) << std::endl;

    std::cout << "sizeof Hexahedron   \t" << sizeof(Hexahedron) << std::endl;
    std::cout << "sizeof Prism        \t" << sizeof(Prism) << std::endl;
    std::cout << "sizeof Pyramid      \t" << sizeof(Pyramid) << std::endl;
    std::cout << "sizeof Tetrahedron  \t" << sizeof(Tetrahedron) << std::endl;
    std::cout << "sizeof GeneralCell  \t" << sizeof(GeneralCell) << std::endl;

    
    // std::cout << "sizeof HexMesh  \t" << sizeof(mesh) << std::endl;
    // std::cout << "sizeof GeneralMesh  \t" << sizeof(new_mesh) << std::endl;

    std::cout << std::get<QuadFace>(new_mesh.m_faces[17]).nodes[0] <<  std::endl;


    std::cout << "points num: \t" << new_mesh.m_points.size()
            << "\t faces num: \t" << new_mesh.m_faces.size()
            << "\t cells num: \t" << new_mesh.m_cells.size()<<std::endl;
    new_mesh.split_hex6_scan();
//     for(auto&f:new_mesh.m_faces){
//         auto&face=std::get<TriangleFace>(f);
//         debug(face.neighbor_cells);
//     }
    new_mesh.rebuild_cell_topology();
    new_mesh.validate_mesh();
    std::cout << "points num: \t" << new_mesh.m_points.size()
            << "\t faces num: \t" << new_mesh.m_faces.size()
            << "\t cells num: \t" << new_mesh.m_cells.size()<<std::endl;



    std::cout << "sizeof TriangleFace  \t" << sizeof(TriangleFace) << std::endl;
    std::cout << "sizeof QuadFace      \t" << sizeof(QuadFace) << std::endl;
    std::cout << "sizeof GeneralFace   \t" << sizeof(GeneralFace) << std::endl;

    std::cout << "sizeof Hexahedron   \t" << sizeof(Hexahedron) << std::endl;
    std::cout << "sizeof Prism        \t" << sizeof(Prism) << std::endl;
    std::cout << "sizeof Pyramid      \t" << sizeof(Pyramid) << std::endl;
    std::cout << "sizeof Tetrahedron  \t" << sizeof(Tetrahedron) << std::endl;
    std::cout << "sizeof GeneralCell  \t" << sizeof(GeneralCell) << std::endl;

    
    // std::cout << "sizeof HexMesh  \t" << sizeof(mesh) << std::endl;
    // std::cout << "sizeof GeneralMesh  \t" << sizeof(new_mesh) << std::endl;

    std::cout << std::get<TriangleFace>(new_mesh.m_faces[17]).nodes[0] <<  std::endl;

    // std::unordered_set<uInt> flat_diags;
    // uInt count1=0,count2=0;
    // for(auto&f:new_mesh.m_faces){
    //     if(std::get_if<TriangleFace>(&f)){
    //         const auto& face = std::get<TriangleFace>(f);
    //         uInt  d1 = face.neighbor_cells[0];
    //         uInt  d2 = (face.neighbor_cells[1]==uInt(-1)?new_mesh.m_faces.size()+flat_diags.size():face.neighbor_cells[1]);
    //         flat_diags.insert(d1*100000+d2);
    //         if(face.neighbor_cells[1]!=uInt(-1)) count1++; else count2++;
    //         if(face.neighbor_cells[1]!=uInt(-1))debug(face.nodes);
    //         // debug(d1*100000+d2);
    //     }
    // }
    // debug(flat_diags.size());
    // debug(count2);
    
    



    // for(auto&f:new_mesh.m_faces){
    //     if(std::get_if<QuadFace>(&f)){
    //         const auto& face = std::get<QuadFace>(f);
    //         debug(face.diagonal_nodes);
    //     }
    // }

    // for(auto&c:new_mesh.m_cells){
    //     if(std::get_if<Tetrahedron>(&c)){
    //         const auto& cell = std::get<Tetrahedron>(c);
    //         const auto& points = new_mesh.m_points;
    //         const auto& p0 = points[cell.nodes[0]];
    //         const auto& p1 = points[cell.nodes[1]];
    //         const auto& p2 = points[cell.nodes[2]];
    //         const auto& p3 = points[cell.nodes[3]];
    //         const auto& vol = vec_dot(vec_cross(p1-p0,p2-p0),p3-p0);
    //         if(vol<1e-8){
    //             debug(cell.nodes);
    //             // debug(p0);
    //             // debug(p1);
    //             // debug(p2);
    //             // debug(p3);
    //             debug(p1-p0);
    //             debug(p2-p0);
    //             debug(p3-p0);
    //         }
    //     }
    // }


    ComputingMesh cmesh(new_mesh);
    std::cout << "points num: \t" << cmesh.m_points.size()
            << "\t faces num: \t" << new_mesh.m_faces.size()
            << "\t cells num: \t" << cmesh.m_cells.size()<<std::endl;
    auto f=[&](const vector3f& xyz)->Scalar{return xyz[0]*xyz[0]+xyz[1]*xyz[1]+xyz[2]*xyz[2];};

    // debug(cmesh.m_cells[0].interpolate_phy({1,0,0,0}));
    
    // for(const auto& face:cmesh.m_faces){
    //     const auto& face_integrate = face.integrate<GaussLegendreTri::Degree4Points7>(f);
    //     std::cout<<face.m_neighbor_cells[0]<<"\t"<<face_integrate<<std::endl;
    // }

    // std::cout << "sizeof TriangleFace  \t" << sizeof(TriangleFace) << std::endl;
    // std::cout << "sizeof QuadFace      \t" << sizeof(QuadFace) << std::endl;
    // std::cout << "sizeof GeneralFace   \t" << sizeof(GeneralFace) << std::endl;

    // std::cout << "sizeof Hexahedron   \t" << sizeof(Hexahedron) << std::endl;
    // std::cout << "sizeof Prism        \t" << sizeof(Prism) << std::endl;
    // std::cout << "sizeof Pyramid      \t" << sizeof(Pyramid) << std::endl;
    // std::cout << "sizeof Tetrahedron  \t" << sizeof(Tetrahedron) << std::endl;
    // std::cout << "sizeof GeneralCell  \t" << sizeof(GeneralCell) << std::endl;

    
    // std::cout << "ComputingMesh: Num of cells \t" << cmesh.m_cells.size() << std::endl;
    // std::cout << cmesh.m_cells[17].m_dual_faces[0].m_area <<  std::endl;



    

    return 0;
}