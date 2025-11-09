#include "base/type.h"
#include "mesh/mesh.h"
#include "mesh/cgal_mesh.h"
#include "matrix/matrix.h"
#include "dg/dg_basis/dg_basis.h"

inline ComputingMesh create_mesh(uInt N){
    Scalar h = 1.0/N;

    uInt Nx = 16*N;
    uInt Ny = 5*N;
    CGALMesh generator(0.125, h, h*0.1); // 设置长宽比为0.2，网格大小为h，厚度为0.5*h
    

    std::vector<std::array<double, 2>> points;// = {{0.0, 0.0}, {3.2, 0.0}, {3.2, 1.0}, {0.0, 1.0}};
    std::vector<std::array<double, 2>> internal_points;

    // 边界点
    for(uInt nx = 0; nx < Nx; nx++){
        Scalar x = 0.0 + (3.2 - 0.0) * nx / Nx;
        points.push_back({x, 0.0});
    }
    for(uInt ny = 0; ny < Ny; ny++){
        Scalar y = 0.0 + (1.0 - 0.0) * ny / Ny;
        points.push_back({3.2, y});
    }
    for(uInt nx = Nx; nx > 0; nx--){
        Scalar x = 0.0 + (3.2 - 0.0) * nx / Nx;
        points.push_back({x, 1.0});
    }
    for(uInt ny = Ny; ny > 0; ny--){
        Scalar y = 0.0 + (1.0 - 0.0) * ny / Ny;
        points.push_back({0.0, y});
    }
    
    // 内部点
    for(uInt nx = 1; nx < Nx; nx++){
        for(uInt ny = 1; ny < Ny; ny++){
            Scalar x = 0.0 + (3.2 - 0.0) * nx / Nx;
            Scalar y = 0.0 + (1.0 - 0.0) * ny / Ny;
            internal_points.push_back({x, y});
        }
    }
    // 偏移半格局部加密
    auto dist2 = [&](Scalar x, Scalar y, Scalar x0, Scalar y0) -> Scalar {
        // 这里可以根据需要定义哪些点需要局部加密
        return (x - x0) * (x - x0) + (y - y0) * (y - y0);
    };
    auto near_curve = [&](Scalar x, Scalar y) -> Scalar {
        Scalar val = -0.2034*x*x + 0.0597*x*y + -0.7398*y*y + 0.5771*x + -0.2638*y + -0.0718;
        return std::abs(val);
    };
    auto near_init = [&](Scalar x, Scalar y) -> Scalar {
        Scalar val = y - 1.732*(x-0.1667);
        return std::abs(val);
    };
    auto is_refined_1 = [&](Scalar x, Scalar y) -> bool {
        // 这里可以根据需要定义哪些点需要局部加密
        return dist2(x, y, 0.17, 0.0) < 0.1 * 0.1 || 
               (x>2.1 && x<3.1 && y>0.0 && y<0.58) || 
               near_curve(x, y) < 0.06 || near_init(x,y) < 0.06 || (y<0.19*(x-0.1667)+0.13);
    };
    auto is_refined_2 = [&](Scalar x, Scalar y) -> bool {
        // 这里可以根据需要定义哪些点需要局部加密
        // return false;
        return dist2(x, y, 0.17, 0.0) < 0.075 * 0.075 || 
               (x>2.25 && x<3.0 && y>0.0 && y<0.52) || 
               near_curve(x, y) < 0.035 || near_init(x,y) < 0.035 || (y<0.19*(x-0.1667)+0.08);
    };
    for(uInt nx = 0; nx < Nx; nx++){
        for(uInt ny = 0; ny < Ny; ny++){
            // 偏移半格
            Scalar x = 0.0 + (3.2 - 0.0) * (nx + 0.5) / Nx;
            Scalar y = 0.0 + (1.0 - 0.0) * (ny + 0.5) / Ny;
            Scalar hx = (3.2 - 0.0) / Nx;
            Scalar hy = (1.0 - 0.0) / Ny;
            if (is_refined_2(x, y)) {
                internal_points.push_back({x - hx/6.0, y - hy/6.0});
                internal_points.push_back({x + hx/6.0, y - hy/6.0});
                internal_points.push_back({x - hx/6.0, y + hy/6.0});
                internal_points.push_back({x + hx/6.0, y + hy/6.0});
                // 如果是二次加密
                // if(is_refined_2(x - h, y) && x!= 0.0){
                    internal_points.push_back({x - hx/2.0, y - hy/6.0});
                    internal_points.push_back({x - hx/2.0, y + hy/6.0});
                // }
                // if(is_refined_2(x, y - h) && y!= 0.0){
                    internal_points.push_back({x - hx/6.0, y - hy/2.0});
                    internal_points.push_back({x + hx/6.0, y - hy/2.0});
                // }
            }else if (is_refined_1(x, y)) {
                internal_points.push_back({x, y});
                // 只考虑左侧和下侧，不然重复了
                if(is_refined_2(x - hx, y) && x!= 0.0){
                    internal_points.push_back({x - hx/2.0, y - hy/6.0});
                    internal_points.push_back({x - hx/2.0, y + hy/6.0});
                }
                else{
                    internal_points.push_back({x - hx/2.0, y});
                }
                
                if(is_refined_2(x, y - hy) && y!= 0.0){
                    internal_points.push_back({x - hx/6.0, y - hy/2.0});
                    internal_points.push_back({x + hx/6.0, y - hy/2.0});
                }
                else{
                    internal_points.push_back({x, y - hy/2.0});
                }
                
            }
            else
            {
                if(is_refined_2(x - hx, y) && x!= 0.0){
                    internal_points.push_back({x - hx/2.0, y - hy/6.0});
                    internal_points.push_back({x - hx/2.0, y + hy/6.0});
                }else if(is_refined_1(x - hx, y) && x!= 0.0){
                    internal_points.push_back({x - hx/2.0, y});
                }
                
                if(is_refined_2(x, y - hy) && y!= 0.0){
                    internal_points.push_back({x - hx/6.0, y - hy/2.0});
                    internal_points.push_back({x + hx/6.0, y - hy/2.0});
                }else if(is_refined_1(x, y - hy) && y!= 0.0){
                    internal_points.push_back({x, y - hy/2.0});
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

