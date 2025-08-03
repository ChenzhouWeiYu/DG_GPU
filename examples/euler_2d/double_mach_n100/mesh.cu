#include "base/type.h"
#include "mesh/mesh.h"
#include "mesh/cgal_mesh.h"

ComputingMesh create_mesh(uInt N){
    Scalar h = static_cast<Scalar>(1)/static_cast<Scalar>(5*N) / std::sin(60.0/180.0 * M_PI);
    Scalar dx = h * std::cos(60.0/180.0 * M_PI);
    Scalar dy = h * std::sin(60.0/180.0 * M_PI);

    CGALMesh generator(0.25, h*2, h*0.866); // 设置长宽比为0.2，网格大小为h，厚度为0.5*h
    
    std::vector<std::array<double, 2>> points = {{0.0, 0.0}, {3.2, 0.0}, {3.2, 1.0}, {0.0, 1.0}};
    std::vector<std::array<double, 2>> internal_points;

    for(uInt layer = 0; layer <= 5*N; layer++){
        Scalar y = static_cast<Scalar>(layer)/static_cast<Scalar>(5*N);
        for(uInt nx = 1; nx < 18*N; nx++){
            Scalar x = nx * h + layer * dx;
            if(x>3.2-h) break;
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
            if(std::abs(face.m_normal[2])>0.8 )          
                cmesh.m_boundaryTypes[faceId] = BoundaryType::Pseudo3DZ;
            else{
                const auto& nodes = face.m_nodes;
                const auto& p0 = cmesh.m_points[nodes[0]];
                const auto& p1 = cmesh.m_points[nodes[1]];
                const auto& p2 = cmesh.m_points[nodes[2]];
                const Scalar x = (p0[0] + p1[0] + p2[0]) / 3.0;
                const Scalar y = (p0[1] + p1[1] + p2[1]) / 3.0;
                // print(centor);
                if( x>1.0/6.0 && std::abs(y)<1e-6 && face.m_normal[1]<-0.8 ){
                    cmesh.m_boundaryTypes[faceId] = BoundaryType::Pseudo3DY;
                }
                else
                {
                    cmesh.m_boundaryTypes[faceId] = BoundaryType::Dirichlet;
                }
            }
                
        }
    }
    return cmesh;
}

