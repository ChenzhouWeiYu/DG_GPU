#include "mesh/computing_mesh.h"



CompTriangleFace::CompTriangleFace(const ComputingMesh& parent, vector3u nodes,vector2u neighbor)
    : m_parent_mesh(parent), m_nodes(nodes), m_neighbor_cells(neighbor){
    compute_geometry();
}



template<typename QuadRule, typename Func>
auto CompTriangleFace::integrate(Func&& func) const {
    using ResultType = decltype(func(vector3f{0}));
    ResultType result = 0;
    
    for (uInt i = 0; i < QuadRule::num_points; ++i) {
        const auto& xi = QuadRule::points[i];
        const auto& phy_pos = compute_physical_coord(xi);
        const Scalar jacobian = compute_jacobian(xi);
        
        result += func(phy_pos) * jacobian * QuadRule::weights[i];
    }
    return result;
}

vector3f CompTriangleFace::compute_physical_coord(const std::array<Scalar,2>& xi) const {
    const Scalar u = xi[0], v = xi[1];
    const auto& pos = m_parent_mesh.m_points;
    const auto& p0 = pos[m_nodes[0]];
    const auto& p1 = pos[m_nodes[1]];
    const auto& p2 = pos[m_nodes[2]];
    return p0*(1-u-v) + p1*u + p2*v;
}

Scalar CompTriangleFace::compute_jacobian_det() const {
    return m_area*2.0;
}

void CompTriangleFace::compute_geometry() {
    // 计算体心和体积
    const auto& pos = m_parent_mesh.m_points;
    const auto& p0 = pos[m_nodes[0]];
    const auto& p1 = pos[m_nodes[1]];
    const auto& p2 = pos[m_nodes[2]];
    m_normal = vec_cross(p1 - p0, p2 - p0);
    if(m_neighbor_cells[1]==uInt(-1)){
        const auto& cell = m_parent_mesh.m_cells[m_neighbor_cells[0]];
        vector3f p3 = {0,0,0};
        for(const auto& n:cell.m_nodes){
            if(n==m_nodes[0]) continue;
            if(n==m_nodes[1]) continue;
            if(n==m_nodes[2]) continue;
            p3 = pos[n];
        }
        const auto& v1 = p1-p0, v2 = p2-p0, v3 = p3-p0;
        if(vec_dot(vec_cross(v1,v2),v3) * vec_dot(vec_cross(v1,v2),m_normal) > 0){
            m_normal *= -1;
        }
    }
    else
    if(vec_dot(m_normal,m_parent_mesh.m_cells[m_neighbor_cells[1]].m_centroid-m_parent_mesh.m_cells[m_neighbor_cells[0]].m_centroid)<0){
        m_normal *= -1;
    }
    m_area = vec_length(m_normal)*0.5;
    m_normal = vec_unit(m_normal);
    for(uInt k=0;k<3;k++){
        m_natural_coords[0][k]={0,0,0};
        m_natural_coords[1][k]={0,0,0};
        for(uInt ki=0;ki<3;ki++){
            if(m_nodes[k]==m_parent_mesh.m_cells[m_neighbor_cells[0]].m_nodes[ki+1]) m_natural_coords[0][k][ki]=1;
            if(m_neighbor_cells[1]!=uInt(-1))
            if(m_nodes[k]==m_parent_mesh.m_cells[m_neighbor_cells[1]].m_nodes[ki+1]) m_natural_coords[1][k][ki]=1;
        }
    }
}








CompTetrahedron::CompTetrahedron(const ComputingMesh& parent,const vector4u nodes,const vector4u faces)
    : m_parent_mesh(parent), m_nodes(nodes), m_faces(faces) {
    compute_geometry();
}

CompTetrahedron::CompTetrahedron(const ComputingMesh& parent,const vector4u nodes,const vector4u faces,const vector4u cells)
    : m_parent_mesh(parent), m_nodes(nodes), m_faces(faces), m_neighbors(cells) {
    compute_geometry();
}

//---------------- 单元级积分接口 ----------------
template<typename Func, typename QuadRule>
auto CompTetrahedron::integrate(Func&& func) const {
    using ResultType = decltype(func(vector3f{0}, vector4f{0}));
    ResultType result = 0;
    
    for (uInt i = 0; i < QuadRule::num_points; ++i) {
        const auto& xi = QuadRule::points[i];
        const vector3f phy_pos = transform_to_physical(xi);
        const Scalar jac = compute_jacobian_det();
        
        result += func(phy_pos, xi) * jac * QuadRule::weights[i];
    }
    return result;
}



void CompTetrahedron::compute_geometry() {
    // 计算体心和体积
    const auto& pos = m_parent_mesh.m_points;
    const auto& p0 = pos[m_nodes[0]];
    const auto& p1 = pos[m_nodes[1]];
    const auto& p2 = pos[m_nodes[2]];
    const auto& p3 = pos[m_nodes[3]];
    m_centroid = (p0 + p1 + p2 + p3) / 4.0;
    m_volume = std::abs(vec_dot(p1 - p0, 
        vec_cross(p2 - p0, p3 - p0))) / 6.0;
    m_JacMat = DenseMatrix<3,3>(compute_jacobian_mat()).transpose();
    m_invJac = DenseMatrix<3,3>(compute_jacobian_mat()).inverse();
}

// 坐标变换：标准单元 -> 物理单元
vector3f CompTetrahedron::transform_to_physical(const vector4f& natural_coord) const {
    const auto& phy_pos = m_parent_mesh.m_points;
    return phy_pos[m_nodes[0]] * natural_coord[0] +
            phy_pos[m_nodes[1]] * natural_coord[1] +
            phy_pos[m_nodes[2]] * natural_coord[2] +
            phy_pos[m_nodes[3]] * natural_coord[3];
}

vector3f CompTetrahedron::transform_to_physical(const vector3f& natural_coord) const {
    const auto& phy_pos = m_parent_mesh.m_points;
    return phy_pos[m_nodes[0]] * (1-natural_coord[0]-natural_coord[1]-natural_coord[2]) +
            phy_pos[m_nodes[1]] * natural_coord[0] +
            phy_pos[m_nodes[2]] * natural_coord[1] +
            phy_pos[m_nodes[3]] * natural_coord[2];
}

// 雅可比行列式计算
Scalar CompTetrahedron::compute_jacobian_det() const {
    return m_volume*6.0;
}

std::array<vector3f,3> CompTetrahedron::compute_jacobian_mat() const {
    // return m_volume*6.0;
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
    return {v1,v2,v3}; // 其实写成 JacT 了
    // return std::abs(vec_dot(v1, vec_cross(v2, v3))) / 6.0;
}
DenseMatrix<3,3> CompTetrahedron::get_JacMat() const {
    return m_JacMat;
}
// 实际上是返回了 JacMat 的转置的逆
DenseMatrix<3,3> CompTetrahedron::get_invJacMat() const{
    return m_invJac;
}

// private:
vector4f CompTetrahedron::compute_face_centroid(uInt face_id) const {
    // 面由三个顶点组成，计算平均体积坐标
    constexpr std::array<std::array<uInt,3>,4> faces = {{
        {1,2,3}, {0,2,3}, {0,1,3}, {0,1,2}
    }};
    auto vector4f_unit = [](uInt id){vector4f ret={0,0,0,0};ret[id]=1;return ret;};
    vector4f centroid = {0,0,0,0};
    for (uInt v : faces[face_id]) {
        centroid += vector4f_unit(v);
    }
    return centroid / 3.0f;
}

vector3f CompTetrahedron::interpolate_phy(const vector4f& vol_coord) const {
    const auto& phy_pos = m_parent_mesh.m_points;
    const auto& p0 = phy_pos[m_nodes[0]];
    const auto& p1 = phy_pos[m_nodes[1]];
    const auto& p2 = phy_pos[m_nodes[2]];
    const auto& p3 = phy_pos[m_nodes[3]];
    return p0*vol_coord[0] + p1*vol_coord[1]
            + p2*vol_coord[2] + p3*vol_coord[3];
}









ComputingMesh::ComputingMesh(const GeneralMesh& geo_mesh) {
    // GeneralMesh 转移到 ComputingMesh
    // 释放 GeneralMesh 所有内存
    // 构建对偶网格

    m_points = geo_mesh.m_points;
    m_faces.reserve(geo_mesh.m_faces.size());
    m_cells.reserve(geo_mesh.m_cells.size());
    // 转换所有四面体单元
    for (const auto& cell : geo_mesh.m_cells) {
        if (std::get_if<Tetrahedron>(&cell)) {
            uInt cellId = m_cells.size();
            const auto& nodes = std::get<Tetrahedron>(cell).nodes;
            const auto& faces = std::get<Tetrahedron>(cell).faces;
            m_cells.push_back(CompTetrahedron(*this, nodes, faces));
        }
    }
    for (const auto& face : geo_mesh.m_faces) {
        if (std::get_if<TriangleFace>(&face)) {
            uInt faceId = m_faces.size();
            const auto& nodes = std::get<TriangleFace>(face).nodes;
            const auto& cells = std::get<TriangleFace>(face).neighbor_cells;
            m_faces.push_back(CompTriangleFace(*this, nodes, cells));
        }
    }
    for (uInt cellId = 0; cellId < m_cells.size(); cellId++){
        const auto& cell = m_cells[cellId];
        const auto& nodes = cell.m_nodes;
        const auto& faces = cell.m_faces;
        for(uInt localFaceId=0; localFaceId<4;localFaceId++){
            const auto& faceId = faces[localFaceId];
            const auto& face = m_faces[faceId];
            const auto& neig = face.m_neighbor_cells;
            if(neig[1] == uInt(-1)){
                m_cells[cellId].m_neighbors[localFaceId] = uInt(-1);
            }
            else{
                m_cells[cellId].m_neighbors[localFaceId] = (neig[0]!=cellId) ? neig[0] : neig[1];
            }
        }
        Scalar m_h = 0.0;
        for(const auto& faceId : faces){
            m_h += m_faces[faceId].m_area;
        }
        m_cells[cellId].m_h = cell.m_volume/m_h;
    }
                

    // for (auto& cell : m_cells) {
    //     Scalar m_h = 0.0;
    //     for(const auto& faceId:cell.m_faces){
    //         m_h += m_faces[faceId].m_area;
    //     }
    //     cell.m_h = cell.m_volume/m_h;
    // }


    
}




ComputingMesh::ComputingMesh(const DGMesh& dg_mesh) {
    // GeneralMesh 转移到 ComputingMesh
    // 释放 GeneralMesh 所有内存
    // 构建对偶网格

    m_points = dg_mesh.points;
    m_faces.reserve(dg_mesh.faces.size());
    m_cells.reserve(dg_mesh.cells.size());
    // 转换所有四面体单元
    for (uInt cellId = 0; cellId < dg_mesh.cells.size(); cellId++) {
        const auto& nodes = dg_mesh.cells[cellId];
        const auto& faces = dg_mesh.cell_faces[cellId];
        const auto& cells = dg_mesh.cell_cells[cellId];
        m_cells.push_back(CompTetrahedron(*this, nodes, faces, cells));
    }
    for (uInt faceId = 0; faceId < dg_mesh.faces.size(); faceId++) {
        const auto& nodes = dg_mesh.faces[faceId];
        const auto& cells = dg_mesh.face_cells[faceId];
        m_faces.push_back(CompTriangleFace(*this, nodes, cells));
    }
    for (uInt cellId = 0; cellId < m_cells.size(); cellId++){
        const auto& cell = m_cells[cellId];
        const auto& nodes = cell.m_nodes;
        const auto& faces = cell.m_faces;
        Scalar m_h = 0.0;
        for(const auto& faceId : faces){
            m_h += m_faces[faceId].m_area;
        }
        m_cells[cellId].m_h = cell.m_volume/m_h;
    }
}





// 网格合法性检测
void check_mesh(const ComputingMesh& cmesh){
    for(uInt faceId = 0; faceId < cmesh.m_faces.size(); faceId++){
        const auto& face = cmesh.m_faces[faceId];
        const auto& nb_L = face.m_neighbor_cells[0];
        const auto& nb_R = face.m_neighbor_cells[1];
        if(nb_L!=uInt(-1)){
            const auto& cell = cmesh.m_cells[nb_L];
            bool is_exist = false;
            for(const uInt cell2face :cell.m_faces){
                if (cell2face == faceId) is_exist = true;
            }
            if (!is_exist){
                std::cout << "面 " << faceId << " 的左邻居 " << nb_L << " 不匹配" << std::endl;
                std::cout << cell.m_faces[0] << "\t" << cell.m_faces[1] << "\t" << cell.m_faces[2] << "\t" << cell.m_faces[3] << std::endl;
            }
        }
        if(nb_R!=uInt(-1)){
            const auto& cell = cmesh.m_cells[nb_R];
            bool is_exist = false;
            for(const uInt cell2face :cell.m_faces){
                if (cell2face == faceId) is_exist = true;
            }
            if (!is_exist){
                std::cout << "面 " << faceId << " 的右邻居 " << nb_R << "不匹配" << std::endl;
                std::cout << cell.m_faces[0] << "\t" << cell.m_faces[1] << "\t" << cell.m_faces[2] << "\t" << cell.m_faces[3] << std::endl;
            }
        }
    }

    for(uInt cellId = 0; cellId < cmesh.m_cells.size(); cellId++){
        const auto& cell = cmesh.m_cells[cellId];
        const auto& volume = cell.m_volume;
        if (volume<1e-8){
            std::cout << "cell " << cellId << "  \t volume  " << volume << std::endl;
            for(const auto& pid : cell.m_nodes){
                const auto [x,y,z] = cmesh.m_points[pid];
                std::cout << pid << "\t\t" << x << "\t" << y << "\t" << z << std::endl;
            }
        }
    }
    for(uInt faceId = 0; faceId < cmesh.m_faces.size(); faceId++){
        const auto& face = cmesh.m_faces[faceId];
        const auto& area = face.m_area;
        if (area<1e-8){
            std::cout << "face " << faceId << "  \t area    " << area << std::endl;
            for(const auto& pid : face.m_nodes){
                const auto [x,y,z] = cmesh.m_points[pid];
                std::cout << pid << "\t\t" << x << "\t" << y << "\t" << z << std::endl;
            }
        }
    }
}
