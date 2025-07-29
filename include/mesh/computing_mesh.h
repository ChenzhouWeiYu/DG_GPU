#pragma once

#include "base/type.h"
#include "mesh/general_mesh.h"
#include "matrix/dense_matrix.h"

#include "mesh/cgal_mesh.h"

//========================= 计算网格类 =========================//
class GeneralMesh; // 前向声明
class ComputingMesh;// 前向声明
class CompTetrahedron; // 前向声明



//========================= CompTriangleFace 声明 =========================//
class CompTriangleFace {
public:
    const ComputingMesh& m_parent_mesh;
    vector3u m_nodes{uInt(-1), uInt(-1), uInt(-1)};
    vector2u m_neighbor_cells{uInt(-1), uInt(-1)};
    vector3f m_normal{0.0, 0.0, 0.0};
    Scalar m_area = 0.0;
    std::array<std::array<vector3f,3>,2> m_natural_coords;

    CompTriangleFace(const ComputingMesh& parent, vector3u nodes,vector2u neighbor_cells);
    
    vector3f compute_physical_coord(const std::array<Scalar,2>& xi) const;
    Scalar compute_jacobian_det() const;

    template<typename QuadRule, typename Func>
    auto integrate(Func&& func) const;

    void compute_geometry();
};


//========================= CompTetrahedron 声明 =========================//
class CompTetrahedron {
public:
    const ComputingMesh& m_parent_mesh;
    vector4u m_nodes{uInt(-1), uInt(-1), uInt(-1), uInt(-1)};
    vector3f m_centroid{0.0, 0.0, 0.0};
    vector4u m_faces{uInt(-1), uInt(-1), uInt(-1), uInt(-1)};
    vector4u m_neighbors{uInt(-1), uInt(-1), uInt(-1), uInt(-1)};
    Scalar m_volume = 0.0;
    Scalar m_h = 0.0;
    DenseMatrix<3,3> m_JacMat;
    DenseMatrix<3,3> m_invJac;

    CompTetrahedron(const ComputingMesh& parent,const vector4u nodes,const vector4u faces);
    CompTetrahedron(const ComputingMesh& parent,const vector4u nodes,const vector4u faces,const vector4u cells);
    vector3f interpolate_phy(const vector4f& vol_coord) const;
    vector3f transform_to_physical(const vector4f& natural_coord) const;
    vector3f transform_to_physical(const vector3f& natural_coord) const;
    Scalar compute_jacobian_det() const;
    std::array<vector3f,3> compute_jacobian_mat() const;
    // const DenseMatrix<3,3>& CompTetrahedron::compute_JacMat() const ;
    // const DenseMatrix<3,3>& CompTetrahedron::compute_invJacMat() const;
    DenseMatrix<3,3> get_JacMat() const ;
    DenseMatrix<3,3> get_invJacMat() const;
    template<typename Func, typename QuadRule>
    auto integrate(Func&& func) const;

private:
    void compute_geometry();
    vector4f compute_face_centroid(uInt face_id) const;
};


enum class BoundaryType : uint8_t {
    Dirichlet,  // 狄利克雷
    Neumann,    // 诺伊曼
    Robin,      // 罗宾
    Pseudo3DX,  // X方向伪三维
    Pseudo3DY,  // Y方向伪三维
    Pseudo3DZ,  // Z方向伪三维  
    Symmetry,
    Inflow,     // 流入
    Outflow,    // 流出
    Wall, 
    WallTD,     // 温度壁面(Dirichlet)
    WallTN,     // 温度壁面(Neumann)
    WallTR,     // 温度壁面(Robin)
    // 添加最大值标记用于迭代
    COUNT
};

//========================= ComputingMesh 声明 =========================//
class ComputingMesh {
public:
    std::vector<vector3f> m_points;
    std::vector<CompTriangleFace> m_faces;
    std::vector<CompTetrahedron> m_cells;

    
    std::vector<BoundaryType> m_boundaryTypes;

    explicit ComputingMesh(const class GeneralMesh& geo_mesh);
    explicit ComputingMesh(const DGMesh& dg_mesh);
};






// 网格合法性检测
void check_mesh(const ComputingMesh& cmesh);


