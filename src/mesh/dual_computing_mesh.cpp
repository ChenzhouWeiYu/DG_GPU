#include "mesh/dual_computing_mesh.h"



DualFace::DualFace(const DualCompTetrahedron& parent) 
    : m_parent_cell(parent),
        m_edge_nodes{0,0},
        m_normal{0,0,0},
        m_area(0) {}

DualFace::DualFace(const class DualCompTetrahedron& parent,
        vector2u edge, vector3f normal, Scalar area, 
        std::array<vector4f,4> coords)
    : m_parent_cell(parent), 
        m_edge_nodes(edge), m_normal(normal), 
        m_area(area), m_natural_coords(coords) {}

// DualFace::DualFace(const class DualFace& dual)
//     : m_parent_cell(dual.m_parent_cell), 
//         m_edge_nodes(dual.m_edge_nodes), m_normal(dual.m_normal), 
//         m_area(dual.m_area), m_natural_coords(dual.m_natural_coords) {}



template<typename Func, typename QuadRule>
auto DualFace::integrate(Func&& func) const {
    using ResultType = decltype(func(vector3f{0}));
    ResultType result = 0;
    
    for (uInt i = 0; i < QuadRule::num_points; ++i) {
        const auto& xi = QuadRule::points[i];
        const vector3f phy_pos = compute_physical_coord(xi);
        const Scalar jacobian = compute_jacobian(xi);
        
        result += func(phy_pos) * jacobian * QuadRule::weights[i];
    }
    return result;
}

vector3f DualFace::compute_physical_coord(const std::array<Scalar,2>& xi) const {
    const Scalar u = xi[0], v = xi[1];
    return m_parent_cell.interpolate_phy(
        (m_natural_coords[0]*(1-u)*(1-v) + m_natural_coords[1]*(1+u)*(1-v) +
            m_natural_coords[2]*(1+u)*(1+v) + m_natural_coords[3]*(1-u)*(1+v)) / 4.0
    );
}

Scalar DualFace::compute_jacobian(const std::array<Scalar,2>& xi) const {
    const Scalar u = xi[0], v = xi[1];
    const auto& p0 = m_parent_cell.interpolate_phy(m_natural_coords[0]);
    const auto& p1 = m_parent_cell.interpolate_phy(m_natural_coords[1]);
    const auto& p2 = m_parent_cell.interpolate_phy(m_natural_coords[2]);
    const auto& p3 = m_parent_cell.interpolate_phy(m_natural_coords[3]);
    
    auto du = 
        (p0*(-1)*(1-v) + p1*(1)*(1-v) +
        p2*(1)*(1+v) + p3*(-1)*(1+v)) / 4.0;
    auto dv =
        (p0*(1-u)*(-1) + p1*(1+u)*(-1) +
        p2*(1+u)*(1) + p3*(1-u)*(1)) / 4.0;
    
    return vec_length(vec_cross(du,dv));


}












DualCompTetrahedron::DualCompTetrahedron(const DualComputingMesh& parent, vector4u nodes)
    : m_parent_mesh(parent), m_nodes(nodes),
    m_dual_faces({DualFace(*this),DualFace(*this),DualFace(*this),DualFace(*this),DualFace(*this),DualFace(*this)}) {
    compute_geometry();
    build_dual_faces();
}

//---------------- 单元级积分接口 ----------------
template<typename Func, typename QuadRule>
auto DualCompTetrahedron::integrate(Func&& func) const {
    using ResultType = decltype(func(vector3f{0}, vector4f{0}));
    ResultType result = 0;
    
    for (uInt i = 0; i < QuadRule::num_points; ++i) {
        const auto& xi = QuadRule::points[i];
        const vector3f phy_pos = transform_to_physical(xi);
        const Scalar jac = compute_jacobian();
        
        result += func(phy_pos, xi) * jac * QuadRule::weights[i];
    }
    return result;
}

//---------------- 坐标变换 ----------------
// vector3f DualCompTetrahedron::transform_to_physical(const vector4f& natural_coord) const {
//     const auto& pos = m_parent_mesh.m_points;
//     return pos[m_nodes[0]] * natural_coord[0] +
//             pos[m_nodes[1]] * natural_coord[1] +
//             pos[m_nodes[2]] * natural_coord[2] +
//             pos[m_nodes[3]] * natural_coord[3];
// }

// private:
void DualCompTetrahedron::compute_geometry() {
    // 计算体心和体积
    const auto& pos = m_parent_mesh.m_points;
    const auto& p0 = pos[m_nodes[0]];
    const auto& p1 = pos[m_nodes[1]];
    const auto& p2 = pos[m_nodes[2]];
    const auto& p3 = pos[m_nodes[3]];
    m_centroid = (p0 + p1 + p2 + p3) / 4.0;
    m_volume = std::abs(vec_dot(p1 - p0, 
        vec_cross(p2 - p0, p3 - p0))) / 6.0;
}

void DualCompTetrahedron::build_dual_faces() {
    // 构建对偶面
    // 四面体边-面映射表 [边索引][顶点局部索引]
    constexpr std::array<std::array<uInt,2>,6> edges = {{
        {0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}
    }};
    
    // 各边对应的两个面顶点 [边索引][面顶点]
    constexpr std::array<std::array<uInt,2>,6> adj_faces = {{
        {2,3}, {1,3}, {1,2}, {0,3}, {0,2}, {0,1}
    }};

    for (uInt e = 0; e < 6; ++e) {
        const uInt a = edges[e][0], b = edges[e][1];
        DualFace df(*this);
        vector2u m_edge_nodes = {m_nodes[a], m_nodes[b]};
        df.m_edge_nodes = m_edge_nodes;
        
        auto vector4f_unit = [](uInt id){vector4f ret={0,0,0,0};ret[id]=1;return ret;};
        // 计算对偶面顶点的体积坐标
        const vector4f edge_mid = 0.5*(vector4f_unit(a) + vector4f_unit(b));
        const vector4f face1_centroid = compute_face_centroid(adj_faces[e][0]);
        const vector4f face2_centroid = compute_face_centroid(adj_faces[e][1]);
        const vector4f cell_centroid = {0.25, 0.25, 0.25, 0.25};
        
        std::array<vector4f,4> m_natural_coords = {edge_mid, face1_centroid, cell_centroid, face2_centroid};
        
        // 计算法向量和面积
        const auto& phy_pos = m_parent_mesh.m_points;
        const auto& p0 = interpolate_phy(edge_mid);
        const auto& p1 = interpolate_phy(face1_centroid);
        const auto& p2 = interpolate_phy(cell_centroid);
        const auto& p3 = interpolate_phy(face2_centroid);
        
        vector3f m_normal = 0.5*(vec_cross(p1 - p0, p2 - p0) + vec_cross(p2 - p0, p3 - p0));
        Scalar m_area = vec_length(m_normal);
        m_normal = vec_unit(m_normal);
        
        df.m_normal = m_normal;
        df.m_area = m_area;
        df.m_natural_coords = m_natural_coords;
        
        // m_dual_faces[e] = df;
        // m_dual_faces[e].parent_cell = *this;
        m_dual_faces[e].m_edge_nodes = m_edge_nodes;
        m_dual_faces[e].m_normal = m_normal;
        m_dual_faces[e].m_area = m_area;
        m_dual_faces[e].m_natural_coords = m_natural_coords;
        
    }
}
// 坐标变换：标准单元 -> 物理单元
vector3f DualCompTetrahedron::transform_to_physical(const vector4f& natural_coord) const {
    const auto& phy_pos = m_parent_mesh.m_points;
    return phy_pos[m_nodes[0]] * natural_coord[0] +
            phy_pos[m_nodes[1]] * natural_coord[1] +
            phy_pos[m_nodes[2]] * natural_coord[2] +
            phy_pos[m_nodes[3]] * natural_coord[3];
}

// 雅可比行列式计算
Scalar DualCompTetrahedron::compute_jacobian() const {
    // 计算参考单元到物理单元的雅可比矩阵
    // 具体实现取决于单元类型，此处以四面体为例
    const auto& phy_pos = m_parent_mesh.m_points;
    const auto& p0 = phy_pos[m_nodes[0]];
    const auto& p1 = phy_pos[m_nodes[1]];
    const auto& p2 = phy_pos[m_nodes[2]];
    const auto& p3 = phy_pos[m_nodes[3]];
    
    const vector3f v1 = p1 - p0;
    const vector3f v2 = p2 - p0;
    const vector3f v3 = p3 - p0;
    
    return std::abs(vec_dot(v1, vec_cross(v2, v3))) / 6.0;
}
// private:
vector4f DualCompTetrahedron::compute_face_centroid(uInt face_id) const {
    // 面由三个顶点组成，计算平均体积坐标
    constexpr std::array<std::array<uInt,3>,4> faces = {{
        {1,2,3}, {0,2,3}, {0,1,3}, {0,1,2}
    }};
    auto vector4f_unit = [](uInt id){vector4f ret={0,0,0,0};ret[id]=1;return ret;};
    vector4f centroid = {0,0,0,0};
    for (uInt v : faces[face_id]) {
        centroid += vector4f_unit(v);
    }
    return centroid / 3.0;
}

vector3f DualCompTetrahedron::interpolate_phy(const vector4f& vol_coord) const {
    const auto& phy_pos = m_parent_mesh.m_points;
    const auto& p0 = phy_pos[m_nodes[0]];
    const auto& p1 = phy_pos[m_nodes[1]];
    const auto& p2 = phy_pos[m_nodes[2]];
    const auto& p3 = phy_pos[m_nodes[3]];
    return p0*vol_coord[0] + p1*vol_coord[1]
            + p2*vol_coord[2] + p3*vol_coord[3];
}









DualComputingMesh::DualComputingMesh(const GeneralMesh& geo_mesh) {
    // GeneralMesh 转移到 DualComputingMesh
    // 释放 GeneralMesh 所有内存
    // 构建对偶网格

    m_points = geo_mesh.m_points;
    m_cells.reserve(geo_mesh.m_cells.size());
    // 转换所有四面体单元
    for (const auto& cell : geo_mesh.m_cells) {
        if (std::get_if<Tetrahedron>(&cell)) {
            m_cells.push_back(DualCompTetrahedron(*this, std::get<Tetrahedron>(cell).nodes));
        }
    }
}