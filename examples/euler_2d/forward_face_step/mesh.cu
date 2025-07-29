#include "base/type.h"
#include "mesh/mesh.h"
#include "mesh/cgal_mesh.h"
#include "runner/run_compressible_euler/run_compressible_euler_interface.h"

ComputingMesh create_mesh(uInt N){
    Scalar h = 1.0/(5*N);
    vector3f lb = {0,       0,       0        };
    vector3f ub = {3,       1,       h*0.433  };

    uInt Nx = 15*N;
    uInt Ny = 5*N;
    uInt Nxx = 3*N;
    uInt Nyy = 1*N;
    CGALMesh generator(0.125, h, ub[2]-lb[2]); // 设置长宽比为0.2，网格大小为h，厚度为0.5*h
    

    std::vector<std::array<double, 2>> points;// = {{0.0, -0.25}, {1.0, -0.25}, {1.0, 0.25}, {0.0, 0.25}};
    std::vector<std::array<double, 2>> internal_points;

    // 边界点
    for(uInt nx = 0; nx < Nxx; nx++){
        Scalar x = lb[0] + (ub[0] - lb[0]) * nx / Nx;
        Scalar y = lb[1] + (ub[1] - lb[1]) * 0 / Ny;
        points.push_back({x, y});
    }
    for(uInt ny = 0; ny < Nyy; ny++){
        Scalar x = lb[0] + (ub[0] - lb[0]) * Nxx / Nx;
        Scalar y = lb[1] + (ub[1] - lb[1]) * ny / Ny;
        points.push_back({x, y});
    }
    for(uInt nx = Nxx; nx < Nx; nx++){
        Scalar x = lb[0] + (ub[0] - lb[0]) * nx / Nx;
        Scalar y = lb[1] + (ub[1] - lb[1]) * Nyy / Ny;
        points.push_back({x, y});
    }
    for(uInt ny = Nyy; ny < Ny; ny++){
        Scalar x = lb[0] + (ub[0] - lb[0]) * Nx / Nx;
        Scalar y = lb[1] + (ub[1] - lb[1]) * ny / Ny;
        points.push_back({x, y});
    }
    for(uInt nx = Nx; nx > 0; nx--){
        Scalar x = lb[0] + (ub[0] - lb[0]) * nx / Nx;
        points.push_back({x, ub[1]});
    }
    for(uInt ny = Ny; ny > 0; ny--){
        Scalar y = lb[1] + (ub[1] - lb[1]) * ny / Ny;
        points.push_back({lb[0], y});
    }
    
    for(uInt kk = 1; kk< 16 * N; kk++){
        {
            bool invalid = false;
            // for(uInt k = 0; k <= (5)*(2*kk-1); k++){
            //     Scalar x = 0.6 + (ub[0]-lb[0])/Nx/2 * (2*kk-1) * std::cos(54.0/180.0 * M_PI / (2*kk-1) * k);
            //     Scalar y = 0.2 + (ub[1]-lb[1])/Ny/2 * (2*kk-1) * std::sin(54.0/180.0 * M_PI / (2*kk-1) * k);
            //     if(((x-1.5)*(x-1.5)+y*y > 1.4*1.4)) invalid = true;
            // }
            if(!invalid)
            for(uInt k = 0; k <= (kk<5*N?5:1)*(2*kk-1); k++){
                Scalar x = 0.6 + (ub[0]-lb[0])/Nx/2 * (2*kk-1) * std::cos(54.0/180.0 * M_PI / (2*kk-1) * k);
                Scalar y = 0.2 + (ub[1]-lb[1])/Ny/2 * (2*kk-1) * std::sin(54.0/180.0 * M_PI / (2*kk-1) * k);
                if((y<0.0+1.0/(5*N)-1e-8 || x < 0.0+1.0/(5*N)-1e-8 || x > 3.0-1.0/(5*N)+1e-8 || y > 1.0-1.0/(5*N)+1e-8)) continue;
                // if((x < 1.5) && ((x-1.5)*(x-1.5)+y*y > 1.3*1.3)) continue;
                internal_points.push_back({x,y});
            }
        }
        {
            bool invalid = false;
            // for(uInt k = 1; k < (5)*(2*kk); k++){
            //     Scalar x = 0.6 + (ub[0]-lb[0])/Nx/2 * (2*kk) * std::cos(54.0/180.0 * M_PI / (2*kk) * k);
            //     Scalar y = 0.2 + (ub[1]-lb[1])/Ny/2 * (2*kk) * std::sin(54.0/180.0 * M_PI / (2*kk) * k);
            //     if(((x-1.5)*(x-1.5)+y*y > 1.4*1.4)) invalid = true;
            // }
            if(!invalid)
            for(uInt k = 1; k < (kk<5*N?5:1)*(2*kk); k++){
                Scalar x = 0.6 + (ub[0]-lb[0])/Nx/2 * (2*kk) * std::cos(54.0/180.0 * M_PI / (2*kk) * k);
                Scalar y = 0.2 + (ub[1]-lb[1])/Ny/2 * (2*kk) * std::sin(54.0/180.0 * M_PI / (2*kk) * k);
                if((y<0.0+1.0/(5*N)-1e-8 || x < 0.0+1.0/(5*N)-1e-8 || x > 3.0-1.0/(5*N)+1e-8 || y > 1.0-1.0/(5*N)+1e-8)) continue;
                // if((x < 1.5) && ((x-1.5)*(x-1.5)+y*y > 1.3*1.3)) continue;
                internal_points.push_back({x,y});
            }

        }
        
    }
    



    generator.generate_2d_mesh(points,internal_points,{0.6,0.2});

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
                cmesh.m_boundaryTypes[faceId] = BoundaryType::Pseudo3DZ;
            else {
                cmesh.m_boundaryTypes[faceId] = BoundaryType::Symmetry;
                const auto& nodes = face.m_nodes;
                const auto& p0 = cmesh.m_points[nodes[0]];
                const auto& p1 = cmesh.m_points[nodes[1]];
                const auto& p2 = cmesh.m_points[nodes[2]];
                const Scalar x = (p0[0] + p1[0] + p2[0]) / 3.0;
                const Scalar y = (p0[1] + p1[1] + p2[1]) / 3.0;
                if(std::abs(face.m_normal[0])>0.9999 && x < 0 + 1e-3)          
                    cmesh.m_boundaryTypes[faceId] = BoundaryType::Dirichlet;
                if(std::abs(face.m_normal[0])>0.9999 && x > 3 - 1e-3)
                    cmesh.m_boundaryTypes[faceId] = BoundaryType::Neumann;
            }
        }
    }
    return cmesh;
}



