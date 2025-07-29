#include "mesh/general_mesh.h"


// 辅助函数：标记所有Quad面的剖分对角线
void GeneralMesh::mark_all_quad_diagonals() {
    // 构建三角形面哈希库
    std::unordered_set<uInt> tri_face_hashes;
    for (const auto& face : m_faces) {
        if (const auto* tri = std::get_if<TriangleFace>(&face)) {
            tri_face_hashes.insert(face_hash(*tri));
        }
    }

    // 处理所有Quad面
    for (auto& face : m_faces) {
        if (auto* qf = std::get_if<QuadFace>(&face)) {
            // 已标记的跳过
            // if (qf->diagonal_nodes[0] != uInt(-1)) continue;

            // 尝试两种剖分方式
            const vector4u& n = qf->nodes;
            bool split02 = check_tri_exist({n[0],n[1],n[2]}, tri_face_hashes) &&
                            check_tri_exist({n[0],n[2],n[3]}, tri_face_hashes);

            bool split13 = check_tri_exist({n[0],n[1],n[3]}, tri_face_hashes) &&
                            check_tri_exist({n[1],n[2],n[3]}, tri_face_hashes);

            if (split02) {
                if (qf->diagonal_nodes[0] != uInt(-1)){
                    if(qf->diagonal_nodes[0]!=n[0] && qf->diagonal_nodes[0]!=n[2]){
                        throw std::runtime_error("11111111");
                    }
                }
                qf->diagonal_nodes = {n[0], n[2]};
            } else if (split13) {
                if (qf->diagonal_nodes[0] != uInt(-1)){
                    if(qf->diagonal_nodes[0]!=n[1] && qf->diagonal_nodes[0]!=n[3]){
                        throw std::runtime_error("22222222");
                    }
                }
                qf->diagonal_nodes = {n[1], n[3]};
            } else {
                // 自由剖分选择最短对角线
                // qf->diagonal_nodes = choose_shortest_diagonal(n);
            }
        }
    }
    check_prism_faces_diags(0);
    check_prism_faces_diags(1);
    check_prism_faces_diags(0);
    check_prism_faces_diags(1);
    check_prism_faces_diags(); // 第一次扫描，填写只有唯一解的情况，有多解（第三边自由、只有一条边）的跳过
    check_prism_faces_diags(); // 第二次扫描，再次检查只有一条边的是否变成两条，以及是否唯一解
    check_prism_faces_diags(); // 暂时没想到例外，再加一个也不会出错
    // check_prism_faces_diags(); // 暂时没想到例外，再加一个也不会出错

    for(auto& cell : m_cells){
        if(auto* prism = std::get_if<Prism>(&cell)){
            auto& faces = prism->faces;
            std::array<vector2u,3> diags;
            uInt  count = 0;
            for (uInt i = 0; i < 3; ++i) {
                const QuadFace& qf = std::get<QuadFace>(m_faces[faces[i+2]]);
                diags[i] = qf.diagonal_nodes;
                if(diags[i][0]!=uInt(-1)&&diags[i][1]!=uInt(-1)){
                    count++;
                }
            }
            std::unordered_set<uInt> unique_nodes;
            for (const auto& d : diags) {
                unique_nodes.insert(d[0]);
                unique_nodes.insert(d[1]);
            }
            if(unique_nodes.size()==6)
            debug(vector2u{count,unique_nodes.size()});
        }
    }
    // 自由剖分选择最短对角线
    for (auto& face : m_faces) {
        if (auto* qf = std::get_if<QuadFace>(&face)) {
            // 已标记的跳过
            if (qf->diagonal_nodes[0] != uInt(-1)) continue;

            // 尝试两种剖分方式
            const vector4u& n = qf->nodes;
            bool split02 = check_tri_exist({n[0],n[1],n[2]}, tri_face_hashes) &&
                            check_tri_exist({n[0],n[2],n[3]}, tri_face_hashes);

            bool split13 = check_tri_exist({n[0],n[1],n[3]}, tri_face_hashes) &&
                            check_tri_exist({n[1],n[2],n[3]}, tri_face_hashes);

            if (split02) {
                qf->diagonal_nodes = {n[0], n[2]};
            } else if (split13) {
                qf->diagonal_nodes = {n[1], n[3]};
            } else {
                // 自由剖分选择最短对角线
                qf->diagonal_nodes = choose_shortest_diagonal(n);
            }
        }
    }
    check_prism_faces_diags(); // 暂时没想到例外，再加一个也不会出错
    for (auto& face : m_faces) {
        if (auto* qf = std::get_if<QuadFace>(&face)) {
            // 已标记的跳过
            if (qf->diagonal_nodes[0] != uInt(-1)) continue;

            // 尝试两种剖分方式
            const vector4u& n = qf->nodes;
            bool split02 = check_tri_exist({n[0],n[1],n[2]}, tri_face_hashes) &&
                            check_tri_exist({n[0],n[2],n[3]}, tri_face_hashes);

            bool split13 = check_tri_exist({n[0],n[1],n[3]}, tri_face_hashes) &&
                            check_tri_exist({n[1],n[2],n[3]}, tri_face_hashes);

            if (split02) {
                qf->diagonal_nodes = {n[0], n[2]};
            } else if (split13) {
                qf->diagonal_nodes = {n[1], n[3]};
            } else {
                // 自由剖分选择最短对角线
                qf->diagonal_nodes = choose_shortest_diagonal_inverse(n);
            }
        }
    }
    

    // uInt count = 0;
    // for (auto& face : m_faces) {
    //     if (auto* qf = std::get_if<QuadFace>(&face)) {
    //         const auto& faces = std::get<QuadFace>(face);
    //         const vector4u& n = faces.nodes;
    //         bool split02 = check_tri_exist({n[0],n[1],n[2]}, tri_face_hashes) &&
    //                       check_tri_exist({n[0],n[2],n[3]}, tri_face_hashes);

    //         bool split13 = check_tri_exist({n[0],n[1],n[3]}, tri_face_hashes) &&
    //                       check_tri_exist({n[1],n[2],n[3]}, tri_face_hashes);
    //         if(split02 && split13){
    //             std::runtime_error("22222222222");
    //         }
    //         if(split02 || split13){

    //         }
    //         else{
    //             count += 2;
    //         }
    //     }
    //     if (auto* qf = std::get_if<TriangleFace>(&face)) {
    //         const auto& faces = std::get<TriangleFace>(face);
    //         count += 1;
    //     }
    // }
    // debug(count); // cells*2 + faces*2 = 125*2 + 450*2 = 1150

}

// 辅助函数：检查四边形是否合法构成三棱柱剖分，并标记必须剖分的面
void GeneralMesh::check_prism_faces_diags(uInt flag){
    for(auto& cell : m_cells){
        if(auto* prism = std::get_if<Prism>(&cell)){
            auto& faces = prism->faces;
            std::array<vector2u,3> diags;
            uInt  count = 0;
            for (uInt i = 0; i < 3; ++i) {
                const QuadFace& qf = std::get<QuadFace>(m_faces[faces[i+2]]);
                diags[i] = qf.diagonal_nodes;
                if(diags[i][0]!=uInt(-1)&&diags[i][1]!=uInt(-1)){
                    count++;
                }
            }
            if(count==3){
                std::unordered_set<uInt> unique_nodes;
                for (const auto& d : diags) {
                    unique_nodes.insert(d[0]);
                    unique_nodes.insert(d[1]);
                }

                if (unique_nodes.size() == 6) {
                    std::get<QuadFace>(m_faces[faces[0+2]]).diagonal_nodes = {uInt(-1),uInt(-1)};
                    
                    // throw std::runtime_error("diags split for prism error 111.");
                }
                else{
                    // 至少有解，可以做
                }
            }
            if(count==2){
                for(uInt idx=0;idx<3;idx++){
                    uInt ii=(idx+0)%3;
                    uInt jj=(idx+1)%3;
                    uInt kk=(idx+2)%3;
                    if(diags[0][0]!=uInt(-1)&&diags[1][0]!=uInt(-1)){
                    //     std::unordered_set<uInt> unique_nodes{diags[0][0],diags[0][1],diags[1][0],diags[1][1]};
                    //     if (unique_nodes.size() == 3){
                    //         // 有公共，第三条随意
                    //     }
                    //     else{
                    //         // 选第三条，保证有公共
                    //     }
                        const auto& qf_nodes = std::get<QuadFace>(m_faces[faces[kk+2]]).nodes;
                        std::unordered_set<uInt> unique_nodes02{diags[ii][0],diags[ii][1],
                                                                diags[jj][0],diags[jj][1],
                                                                qf_nodes[0],qf_nodes[2]};
                        std::unordered_set<uInt> unique_nodes13{diags[ii][0],diags[ii][1],
                                                                diags[jj][0],diags[jj][1],
                                                                qf_nodes[1],qf_nodes[3]};
                        if (unique_nodes02.size() == 6 && unique_nodes13.size()==6) {
                            throw std::runtime_error("diags split for prism error 222.");
                        }
                        else if(unique_nodes02.size() == 4 && unique_nodes13.size()==4){
                            // 自由解，可以随便做
                        }
                        if (unique_nodes02.size() == 4 && unique_nodes13.size()==6) {
                            std::get<QuadFace>(m_faces[faces[kk+2]]).diagonal_nodes = {qf_nodes[0],qf_nodes[2]};
                        }
                        else{
                            std::get<QuadFace>(m_faces[faces[kk+2]]).diagonal_nodes = {qf_nodes[1],qf_nodes[3]};
                        }
                    }
                }
            }
            if(count<2){
                for (uInt i = 0; i < 3; ++i) {
                    QuadFace& qf = std::get<QuadFace>(m_faces[faces[i+2]]);
                    if(qf.diagonal_nodes[0]==uInt(-1)){
                        if(flag)
                        qf.diagonal_nodes = choose_shortest_diagonal(qf.nodes);
                        else
                        qf.diagonal_nodes = choose_shortest_diagonal_inverse(qf.nodes);
                        break;
                    }
                }
            }
        }
    }
}









// 辅助函数：检查三角形是否存在
bool GeneralMesh::check_tri_exist(const vector3u& nodes, 
                    const std::unordered_set<uInt>& hashes) const {
    std::array<uInt,3> sorted_nodes = nodes;
    std::sort(sorted_nodes.begin(), sorted_nodes.end());
    return hashes.count(hash_range(sorted_nodes.begin(), sorted_nodes.end()));
}

// 辅助函数：生成排序后节点的哈希
template<uInt N>
uInt GeneralMesh::hash_sorted_nodes(const std::array<uInt,N>& nodes) const {
    std::vector<uInt> sorted(nodes.begin(), nodes.end());
    std::sort(sorted.begin(), sorted.end());
    return hash_range(sorted.begin(), sorted.end());
}

// 辅助函数：选择几何最短对角线
vector2u GeneralMesh::choose_shortest_diagonal(const vector4u& nodes) const {
    const auto& p0 = m_points[nodes[0]];
    const auto& p1 = m_points[nodes[1]];
    const auto& p2 = m_points[nodes[2]];
    const auto& p3 = m_points[nodes[3]];
    
    const Scalar d02 = vec_length(p0-p2);
    const Scalar d13 = vec_length(p1-p3);
    
    return (d02 < d13) ? vector2u{nodes[0], nodes[2]} 
                        : vector2u{nodes[1], nodes[3]};
}
vector2u GeneralMesh::choose_shortest_diagonal_inverse(const vector4u& nodes) const {
    const auto& p0 = m_points[nodes[0]];
    const auto& p1 = m_points[nodes[1]];
    const auto& p2 = m_points[nodes[2]];
    const auto& p3 = m_points[nodes[3]];
    
    const Scalar d02 = vec_length(p0-p2);
    const Scalar d13 = vec_length(p1-p3);
    
    return (d02 < d13) ? vector2u{nodes[1], nodes[3]} 
                        : vector2u{nodes[0], nodes[2]};
}


// 辅助函数：面哈希
uInt GeneralMesh::face_hash(const GeneralFace& face) const {
    return std::visit([&](const auto& f) {
        std::vector<uInt> sorted_nodes(f.nodes.begin(), f.nodes.end());
        std::sort(sorted_nodes.begin(), sorted_nodes.end());
        uInt hash = 17;
        for (uInt node : sorted_nodes) {
            hash = hash * 997 + node;
        }
        return hash;
    }, face);
}

// 更新面邻接信息
void GeneralMesh::update_face_neighbor(GeneralFace& face, uInt cell_id) const {
    std::visit([&](auto& f) {
        if (f.neighbor_cells[1] == uInt(-1)) {
            f.neighbor_cells[1] = cell_id;
        }
    }, face);
}

// 创建新面
GeneralFace GeneralMesh::create_new_face(const GeneralFace& proto, uInt cell_id) const {
    return std::visit([&](const auto& f) {
        auto new_face = f;
        new_face.neighbor_cells = {cell_id, uInt(-1)};
        return GeneralFace(new_face);
    }, proto);
}

template<typename Iter>
size_t GeneralMesh::hash_range(Iter begin, Iter end) const {
    size_t seed = 997;
    for (Iter it = begin; it != end; ++it) {
        seed ^= std::hash<uInt>{}(*it) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}