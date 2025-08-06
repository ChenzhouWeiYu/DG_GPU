#include "base/type.h"
#include "mesh/mesh.h"
#include "matrix/matrix.h"
#include "dg/dg_basis/dg_basis.h"

ComputingMesh create_mesh(uInt N){
    Scalar h = 4.0/N;
    CGALMesh generator(0.20, h, 0.707*h);
    
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

    // internal_points.push_back({0,-3.8});
    // internal_points.push_back({0,-3.72});
    // internal_points.push_back({0,-3.64});
    // internal_points.push_back({0,-3.56});
    // Scalar y = -h/3 * (3*N-1);
    internal_points.push_back({0,-h/3 * (3*N-1.9)});
    internal_points.push_back({0,-h/3 * (3*N-2.7)});
    
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
            if(std::abs(face.m_normal[2])>0.999 )          
                cmesh.m_boundaryTypes[faceId] = BoundaryType::Pseudo3DZ;
            else
                cmesh.m_boundaryTypes[faceId] = BoundaryType::WallTN;
        }
    }
    return cmesh;
}
