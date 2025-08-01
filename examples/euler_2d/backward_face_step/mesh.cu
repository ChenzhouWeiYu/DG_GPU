#include "base/type.h"
#include "mesh/mesh.h"
#include "mesh/cgal_mesh.h"

ComputingMesh create_mesh(uInt N){
    Scalar h = 1.0/(N);
    vector3f lb = {0,       0,       0        };
    vector3f ub = {1,       1,       h*0.433  };

    CGALMesh generator(0.125, h, ub[2]-lb[2]); // 设置长宽比为0.2，网格大小为h，厚度为0.5*h
    

    std::vector<std::array<double, 2>> points = {{lb[0], lb[1]}, {ub[0], lb[1]}, {ub[0], ub[1]}, {lb[0], ub[1]}};
    std::vector<std::array<double, 2>> internal_points;

    internal_points.push_back({0.0,0.5});
    
    for(uInt kk = 1; kk< 16 * N; kk++){
        for(uInt k = 0; k <= 3*kk; k++){
            Scalar x = 0.0 + h/2 * kk * std::cos(60.0/180.0 * M_PI / kk * k - 90.0/180.0 * M_PI);
            Scalar y = 0.5 + h/2 * kk * std::sin(60.0/180.0 * M_PI / kk * k - 90.0/180.0 * M_PI);
            if((y<lb[1]+h-1e-8 || x > ub[0]-h+1e-8 || y > ub[1]-h+1e-8)) continue;
            if(k==0 || k==3*kk) x = 0.0;
            internal_points.push_back({x,y});
        }
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
            if(std::abs(face.m_normal[2])>0.9999 )          
                cmesh.m_boundaryTypes[faceId] = BoundaryType::Symmetry;
            else {
                cmesh.m_boundaryTypes[faceId] = BoundaryType::Symmetry;
                const auto& nodes = face.m_nodes;
                const auto& p0 = cmesh.m_points[nodes[0]];
                const auto& p1 = cmesh.m_points[nodes[1]];
                const auto& p2 = cmesh.m_points[nodes[2]];
                const Scalar x = (p0[0] + p1[0] + p2[0]) / 3.0;
                const Scalar y = (p0[1] + p1[1] + p2[1]) / 3.0;
                if(std::abs(face.m_normal[0])>0.9999 && x < 0 + 1e-4 && y > 0.5)          
                    cmesh.m_boundaryTypes[faceId] = BoundaryType::Dirichlet;
                if(std::abs(face.m_normal[0])>0.9999 && x > 1 - 1e-4)
                    cmesh.m_boundaryTypes[faceId] = BoundaryType::Neumann;
            }
        }
    }
    return cmesh;
}



