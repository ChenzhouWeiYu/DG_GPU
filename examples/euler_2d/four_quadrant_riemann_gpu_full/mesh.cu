#include "base/type.h"
#include "mesh/mesh.h"
#include "mesh/cgal_mesh.h"

ComputingMesh create_mesh(uInt N){
    Scalar h = 1.0/(7*N);

    // uInt Nx = 7*N;
    // uInt Ny = 7*N;
    CGALMesh generator(0.20, h, 0.707*h); // 设置长宽比为0.2，网格大小为h，厚度为0.5*h
    
    constexpr vector2f lb = {0,0};
    constexpr vector2f ub = {1,1};

    std::vector<std::array<double, 2>> points = {{lb[0], lb[1]}, {ub[0], lb[1]}, {ub[0], ub[1]}, {lb[0], ub[1]}};
    std::vector<std::array<double, 2>> internal_points;
    // constexpr auto Qpoints = GaussLegendreTri::Degree25Points117::get_points();
    // for(const auto p : Qpoints){
    //     uInt NN = (N+1)/2;
    //     Scalar hh = 1.0/NN;
    //     for(uInt kk = 0; kk < NN; kk++){
    //         internal_points.push_back({hh * kk + hh * p[0], hh * kk + hh * (1-p[1])});
    //     }
    //     for(uInt kk = 0; kk < NN-1; kk++){
    //         internal_points.push_back({hh * kk + hh * (1-p[0]) + hh, hh * kk + hh * p[1]});
    //     }
    // }
    uInt NN = 1;
    for(NN = 16; NN < (7*N)<<1; NN<<=1);
    Scalar hh = 1.0/NN;
    // for(uInt kk = 4; kk < NN-3; kk++){
    //     internal_points.push_back({hh * kk, hh * kk});
    //     internal_points.push_back({hh * kk - hh, hh * kk + hh});
    //     internal_points.push_back({hh * kk - hh * 2, hh * kk + hh * 2});
    //     internal_points.push_back({hh * kk - hh * 3, hh * kk + hh * 3});
    // }
    Scalar dx1 = hh * M_SQRT2 * std::cos(105.0/180.0 * M_PI);
    Scalar dy1 = hh * M_SQRT2 * std::sin(105.0/180.0 * M_PI);
    Scalar dx2 = hh * std::cos(-120.0/180.0 * M_PI);
    Scalar dy2 = hh * std::sin(-120.0/180.0 * M_PI);
    Scalar dx3 = hh * std::cos(30.0/180.0 * M_PI);
    Scalar dy3 = hh * std::sin(30.0/180.0 * M_PI);
    for(uInt kk = 1; kk < NN; kk++){
        for(uInt kkk = 0; kkk < uInt(2.0/hh); kkk++){
            Scalar x = hh * kk + dx1 * kkk;
            Scalar y = hh * kk + dy1 * kkk;
            if((x > 0 + hh*3 && y < 1 - hh*3) || kkk == 0){
                internal_points.push_back({x,y});
                if (kkk>0){
                    internal_points.push_back({y,x});
                }
            }
        }
        for(uInt kkk = 0; kkk < 3; kkk++){
            Scalar x = hh * kk + dx2 * kkk;
            Scalar y = 1.0     + dy2 * kkk;
            if((y-x>hh && x+y>1+hh/2) || kkk == 0){
                internal_points.push_back({x,y});
                if (kkk>0){
                    internal_points.push_back({y,x});
                }
            }
        }
        for(uInt kkk = 0; kkk < 3; kkk++){
            Scalar x = 0.0     + dx3 * kkk;
            Scalar y = hh * kk + dy3 * kkk;
            if((y-x>hh && x+y<1-hh/2) || kkk == 0){
                internal_points.push_back({x,y});
                if (kkk>0){
                    internal_points.push_back({y,x});
                }
            }
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
            if(std::abs(face.m_normal[2])>0.9 )          
                cmesh.m_boundaryTypes[faceId] = BoundaryType::Symmetry;
            else 
                cmesh.m_boundaryTypes[faceId] = BoundaryType::Neumann;
            /*
            if(std::abs(face.m_normal[2])<1e-4){
                if(std::abs(face.m_normal[0]+face.m_normal[1])<1e-4){
                    cmesh.m_boundaryTypes[faceId] = BoundaryType::Symmetry;
                }
                else{
                    // cmesh.m_boundaryTypes[faceId] = BoundaryType::Dirichlet;
                    cmesh.m_boundaryTypes[faceId] = BoundaryType::Neumann;
                }
            }
            */
                
        }
    }
    return cmesh;
}



