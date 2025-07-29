#pragma once

#include "mesh/general_mesh.h"

//========================= 计算网格类 =========================//
class GeneralMesh; // 前向声明
class DualComputingMesh;// 前向声明
class DualCompTetrahedron; // 前向声明



//========================= DualFace 声明 =========================//
class DualFace {
public:
    const DualCompTetrahedron& m_parent_cell;
    vector2u m_edge_nodes;
    vector3f m_normal;
    Scalar m_area;
    std::array<vector4f,4> m_natural_coords;

    DualFace(const DualCompTetrahedron& parent);
    DualFace(const DualCompTetrahedron& parent, vector2u edge, vector3f normal,
            Scalar area, std::array<vector4f,4> coords);
    
    vector3f compute_physical_coord(const std::array<Scalar,2>& xi) const;
    Scalar compute_jacobian(const std::array<Scalar,2>& xi) const;

    template<typename Func, typename QuadRule>
    auto integrate(Func&& func) const;
};


//========================= DualCompTetrahedron 声明 =========================//
class DualCompTetrahedron {
public:
    const DualComputingMesh& m_parent_mesh;
    vector4u m_nodes;
    vector3f m_centroid;
    Scalar m_volume;
    std::array<class DualFace,6> m_dual_faces;

    DualCompTetrahedron(const DualComputingMesh& parent, vector4u nodes);
    vector3f interpolate_phy(const vector4f& vol_coord) const;
    vector3f transform_to_physical(const vector4f& natural_coord) const;
    Scalar compute_jacobian() const;
    template<typename Func, typename QuadRule>
    auto integrate(Func&& func) const;

private:
    void compute_geometry();
    void build_dual_faces();
    vector4f compute_face_centroid(uInt face_id) const;
};


//========================= DualComputingMesh 声明 =========================//
class DualComputingMesh {
public:
    std::vector<vector3f> m_points;
    std::vector<DualCompTetrahedron> m_cells;

    explicit DualComputingMesh(const class GeneralMesh& geo_mesh);
};






