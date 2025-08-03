#include "base/type.h"
#include "mesh/mesh.h"
#include "mesh/cgal_mesh.h"

ComputingMesh create_mesh(uInt N){

    Scalar h = 1.0/(N);
    vector3f lb = {  0,     0,     0        };
    vector3f ub = {0.6,   0.6,     h*0.866  };

    CGALMesh generator(0.25, h, ub[2]-lb[2]); // 设置长宽比为0.2，网格大小为h，厚度为0.5*h
    

    std::vector<std::array<double, 2>> points = {
        {0.00, 0.00}, 
        {0.24, 0.00}, {0.24, 0.09}, {0.285, 0.09}, {0.285, 0.00},
        {0.60, 0.00}, {0.60, 0.60}, {0.00, 0.60}
    };
    std::vector<std::array<double, 2>> internal_points;
    // for(uInt nx=0;nx<=40;nx++){
    //     for(uInt ny=0;ny<=40;ny++){
    //         Scalar x = 0.00 + nx * 0.015;
    //         Scalar y = 0.00 + ny * 0.015;
    //         if(x > 0.24 && x < 0.285 && y > 0.00 && y < 0.09) continue;
    //         if(std::abs(x-0.00) < 1e-8) x = 0.00;
    //         if(std::abs(y-0.00) < 1e-8) y = 0.00;
    //         if(std::abs(x-0.60) < 1e-8) x = 0.60;
    //         if(std::abs(y-0.60) < 1e-8) y = 0.60;
    //         if(std::abs(x-0.24) < 1e-8) x = 0.24;
    //         if(std::abs(y-0.09) < 1e-8) y = 0.09;
    //         if(std::abs(x-0.285) < 1e-8) x = 0.285;

    //         internal_points.push_back({x,y});
    //     }
    // }

    // 求解域左下角的 corner, 0 to 90 
    Scalar LD_x = 0.0, LD_y = 0.0;
    Scalar h_corner = N < 30 ? (0.6/30) : 0.6/(60*((N+59)/60));
    for(uInt layer = 0; layer < N/3; layer++){
        Scalar radius = h_corner * (1 + (0.866 + layer*0.0025) * layer);
        if(radius >= 0.18) break;
        for(uInt k = 0; k<=layer+2; k++){
            Scalar x = LD_x + radius * std::cos( 0.0/180.0 * M_PI + 90.0/180.0 * M_PI / (layer+2) * k);
            Scalar y = LD_y + radius * std::sin( 0.0/180.0 * M_PI + 90.0/180.0 * M_PI / (layer+2) * k);
            if(std::abs(x-LD_x) < 1e-8) x = LD_x;
            if(std::abs(y-LD_y) < 1e-8) y = LD_y;
            internal_points.push_back({x,y});
        }
    }
    for(uInt layer = N/3; layer < N*2; layer++){
        Scalar radius = h_corner * (1 + (0.866 + (N/3)*0.0025) * (layer));
        if(radius > 0.6 - 1.732*h_corner) break;
        for(uInt k = layer+2 - (N/8); k<=layer+2; k++){
            Scalar x = LD_x + radius * std::cos( 0.0/180.0 * M_PI + 90.0/180.0 * M_PI / (layer+2) * k);
            Scalar y = LD_y + radius * std::sin( 0.0/180.0 * M_PI + 90.0/180.0 * M_PI / (layer+2) * k);
            if(std::abs(x-LD_x) < 1e-8) x = LD_x;
            if(std::abs(y-LD_y) < 1e-8) y = LD_y;
            internal_points.push_back({x,y});
        }
    }

    // 障碍物左下角 90 to 180
    Scalar MLD_x = 0.24, MLD_y = 0.00;
    for(uInt layer = 0; layer < N/6; layer++){
        Scalar radius = h_corner * (1 + 0.866 * layer);
        if(radius >= 0.028) break;
        for(uInt k = 0; k<=layer+2; k++){
            Scalar x = MLD_x + radius * std::cos( 90.0/180.0 * M_PI + 90.0/180.0 * M_PI / (layer+2) * k);
            Scalar y = MLD_y + radius * std::sin( 90.0/180.0 * M_PI + 90.0/180.0 * M_PI / (layer+2) * k);
            if(std::abs(x-MLD_x) < 1e-8) x = MLD_x;
            if(std::abs(y-MLD_y) < 1e-8) y = MLD_y;
            internal_points.push_back({x,y});
        }
    }

    // 障碍物右下角 0 to 90
    Scalar MRD_x = 0.285, MRD_y = 0.00;
    for(uInt layer = 0; layer < N/6; layer++){
        Scalar radius = h_corner * (1 + 0.866 * layer);
        if(radius >= 0.028) break;
        for(uInt k = 0; k<=layer+2; k++){
            Scalar x = MRD_x + radius * std::cos( 0.0/180.0 * M_PI + 90.0/180.0 * M_PI / (layer+2) * k);
            Scalar y = MRD_y + radius * std::sin( 0.0/180.0 * M_PI + 90.0/180.0 * M_PI / (layer+2) * k);
            if(std::abs(x-MRD_x) < 1e-8) x = MRD_x;
            if(std::abs(y-MRD_y) < 1e-8) y = MRD_y;
            internal_points.push_back({x,y});
        }
    }

    // 求解域右下角的 corner, 90 to 180
    Scalar RD_x = 0.60, RD_y = 0.00;
    for(uInt layer = 0; layer < N/6; layer++){
        Scalar radius = h_corner * (1 + (0.866 + layer*0.0025) * layer);
        if(radius >= 0.3) break;
        for(uInt k = 0; k<=layer+2; k++){
            Scalar x = RD_x + radius * std::cos( 90.0/180.0 * M_PI + 90.0/180.0 * M_PI / (layer+2) * k);
            Scalar y = RD_y + radius * std::sin( 90.0/180.0 * M_PI + 90.0/180.0 * M_PI / (layer+2) * k);
            if(std::abs(x-RD_x) < 1e-8) x = RD_x;
            if(std::abs(y-RD_y) < 1e-8) y = RD_y;
            internal_points.push_back({x,y});
        }
    }

    for(uInt layer = N/6; layer < (N*2)/3; layer++){
        Scalar radius = h_corner * (1 + (0.866 + (N/6)*0.0025) * layer);
        for(uInt k = layer+2-(N/24); k<=layer+2; k++){
            Scalar x = RD_x + radius * std::cos( 90.0/180.0 * M_PI + 90.0/180.0 * M_PI / (layer+2) * k);
            Scalar y = RD_y + radius * std::sin( 90.0/180.0 * M_PI + 90.0/180.0 * M_PI / (layer+2) * k);
            if((x-MRD_x)*(x-MRD_x) + (y-MRD_y)*(y-MRD_y) < 0.035*0.035) continue;
            if(std::abs(x-RD_x) < 1e-8) x = RD_x;
            if(std::abs(y-RD_y) < 1e-8) y = RD_y;
            internal_points.push_back({x,y});
        }
    }

    for(uInt layer = N/6; layer < N*2; layer++){
        Scalar radius = h_corner * (1 + (0.866 + (N/6)*0.0025) * layer);
        if(radius > 0.6 - 1.732*h_corner) break;
        for(uInt k = 0; k<=(N/6)+2; k++){
            Scalar x = RD_x + radius * std::cos( 90.0/180.0 * M_PI + 90.0/180.0 * M_PI / (layer+2) * k);
            Scalar y = RD_y + radius * std::sin( 90.0/180.0 * M_PI + 90.0/180.0 * M_PI / (layer+2) * k);
            if((x-MRD_x)*(x-MRD_x) + (y-MRD_y)*(y-MRD_y) < 0.035*0.035) continue;
            if(std::abs(x-RD_x) < 1e-8) x = RD_x;
            if(std::abs(y-RD_y) < 1e-8) y = RD_y;
            internal_points.push_back({x,y});
        }
    }

    // 障碍物左上角 0 to 270, 使用 54 度角
    Scalar MLU_x = 0.24, MLU_y = 0.09;
    for(uInt layer = 0; layer < N/3; layer++){
        Scalar radius = h_corner * (1 + (0.707 + layer*0.005) * layer);
        for(uInt k = 0; k <= 5*(layer+1); k++){
            Scalar x = MLU_x + radius * std::cos( 0.0/180.0 * M_PI + 54.0/180.0 * M_PI / (layer+1) * k);
            Scalar y = MLU_y + radius * std::sin( 0.0/180.0 * M_PI + 54.0/180.0 * M_PI / (layer+1) * k);
            if((x-MLD_x)*(x-MLD_x) + (y-MLD_y)*(y-MLD_y) < 0.035*0.035) continue;
            if((x-LD_x)*(x-LD_x) + (y-LD_y)*(y-LD_y) < 0.16*0.16) continue;
            if(x>0.5*(MLD_x+MRD_x)-0.33*h_corner) continue;
            if(y<0.33*h) continue;
            if(std::abs(x-MLU_x) < 1e-8) x = MLU_x;
            if(std::abs(y-MLU_y) < 1e-8) y = MLU_y;
            internal_points.push_back({x,y});
        }
    }

    // 障碍物右上角 -90 to 180, 使用 54 度角
    Scalar MRU_x = 0.285, MRU_y = 0.09;
    for(uInt layer = 0; layer < N/3; layer++){
        Scalar radius = h_corner * (1 + (0.707 + layer*0.005) * layer);
        for(uInt k = 0; k <= 5*(layer+1); k++){
            Scalar x = MRU_x + radius * std::cos( -90.0/180.0 * M_PI + 54.0/180.0 * M_PI / (layer+1) * k);
            Scalar y = MRU_y + radius * std::sin( -90.0/180.0 * M_PI + 54.0/180.0 * M_PI / (layer+1) * k);
            if((x-MRD_x)*(x-MRD_x) + (y-MRD_y)*(y-MRD_y) < 0.035*0.035) continue;
            if((x-LD_x)*(x-LD_x) + (y-LD_y)*(y-LD_y) < 0.16*0.16) continue;
            if(y<0.032) continue;
            if(x<0.5*(MLD_x+MRD_x)+0.33*h_corner) continue;
            if(std::abs(x-MRU_x) < 1e-8) x = MRU_x;
            if(std::abs(y-MRU_y) < 1e-8) y = MRU_y;
            internal_points.push_back({x,y});
        }
    }

    
    generator.generate_2d_mesh(points,internal_points,RectangularHole({0.24,0.0},{0.285,0.09}));
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
                if(std::abs(face.m_normal[1])>0.9999 && y > 0.6 - 1e-4)          
                    cmesh.m_boundaryTypes[faceId] = BoundaryType::Neumann;
                if(std::abs(face.m_normal[0])>0.9999 && x > 0.6 - 1e-4)
                    cmesh.m_boundaryTypes[faceId] = BoundaryType::Neumann;
                
                // if(std::abs(face.m_normal[1])>0.9999 && y < 0.0 + 1e-4)          
                //     cmesh.m_boundaryTypes[faceId] = BoundaryType::Neumann;
                // if(std::abs(face.m_normal[0])>0.9999 && x < 0.0 + 1e-4)
                //     cmesh.m_boundaryTypes[faceId] = BoundaryType::Neumann;
            }
        }
    }
    return cmesh;
}



