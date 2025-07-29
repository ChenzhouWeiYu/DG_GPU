#include "base/type.h"
#include "mesh/mesh.h"
#include "mesh/cgal_mesh.h"
#include "runner/run_compressible_euler/run_compressible_euler_interface.h"

ComputingMesh create_mesh(uInt N){
    Scalar h = 1.0/N;
    vector3f lb = {0,       0,        0        };
    vector3f ub = {1,       h*0.866,  h*0.866  };

    uInt Nx = 1*N;
    uInt Ny = 1;
    CGALMesh generator(0.125, h, ub[2]-lb[2]); // 设置长宽比为0.2，网格大小为h，厚度为0.5*h
    

    std::vector<std::array<double, 2>> points;// = {{0.0, -0.25}, {1.0, -0.25}, {1.0, 0.25}, {0.0, 0.25}};
    std::vector<std::array<double, 2>> internal_points;

    // 边界点
    for(uInt nx = 0; nx < Nx; nx++){
        Scalar x = lb[0] + (ub[0] - lb[0]) * nx / Nx;
        points.push_back({x, lb[1]});
    }
    for(uInt ny = 0; ny < Ny; ny++){
        Scalar y = lb[1] + (ub[1] - lb[1]) * ny / Ny;
        points.push_back({ub[0], y});
    }
    for(uInt nx = Nx; nx > 0; nx--){
        Scalar x = lb[0] + (ub[0] - lb[0]) * nx / Nx;
        points.push_back({x, ub[1]});
    }
    for(uInt ny = Ny; ny > 0; ny--){
        Scalar y = lb[1] + (ub[1] - lb[1]) * ny / Ny;
        points.push_back({lb[0], y});
    }
    



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
            else 
            if(std::abs(face.m_normal[1])>0.9 )          
                cmesh.m_boundaryTypes[faceId] = BoundaryType::Pseudo3DY;
            else{
                cmesh.m_boundaryTypes[faceId] = BoundaryType::Dirichlet;
                // cmesh.m_boundaryTypes[faceId] = BoundaryType::Neumann;
            }
                
        }
    }
    return cmesh;
}



