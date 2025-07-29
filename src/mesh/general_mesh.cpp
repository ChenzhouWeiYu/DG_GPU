#include "mesh/general_mesh.h"


void GeneralMesh::rebuild_cell_topology() {
    // 步骤1：为每个单元收集关联的面
    std::vector<std::vector<uInt>> cell_face_map(m_cells.size());
    
    // 遍历所有面，记录单元-面关系
    for (uInt face_id = 0; face_id < m_faces.size(); ++face_id) {
        std::visit([&](auto&& face) {
            for (uInt cell_id : face.neighbor_cells) {
                if (cell_id != uInt(-1)) {
                    cell_face_map[cell_id].push_back(face_id);
                }
            }
        }, m_faces[face_id]);
    }
    // 步骤2：为每个单元提取节点
    for (uInt cell_id = 0; cell_id < m_cells.size(); ++cell_id) {
        auto& CF_map = cell_face_map[cell_id];
        std::visit([&](auto&& cell) {
            using T = std::decay_t<decltype(cell)>;
            
            // 收集所有关联面的节点
            std::unordered_set<uInt> node_set;
            for (uInt face_id : CF_map) {
                std::visit([&](auto&& face) {
                    for (uInt n : face.nodes) node_set.insert(n);
                }, m_faces[face_id]);
            }

            // 验证节点数量
            constexpr uInt expected_nodes = []{
                if constexpr (std::is_same_v<T, Hexahedron>) return 8;
                if constexpr (std::is_same_v<T, Prism>) return 6;
                if constexpr (std::is_same_v<T, Pyramid>) return 5;
                if constexpr (std::is_same_v<T, Tetrahedron>) return 4;
            }();

            if (node_set.size() != expected_nodes) {
                std::cout << CF_map.size()<<"\t"<<node_set.size() <<"\t"<< expected_nodes<<std::endl;
                throw std::runtime_error("Cell node count mismatch");
            }

            // 填充节点（临时顺序，然后再考虑重排）
            std::vector<uInt> temp_nodes(node_set.begin(), node_set.end());
            std::copy(temp_nodes.begin(), temp_nodes.end(), cell.nodes.begin());
            std::copy(CF_map.begin(), CF_map.end(), cell.faces.begin());
            cell.reorder(m_points,m_faces);
        }, m_cells[cell_id]);
    }
}

void GeneralMesh::validate_mesh() const {
    // 验证所有面至少连接一个有效单元
    for (const auto& face : m_faces) {
        std::visit([&](auto&& f) {
            if (f.neighbor_cells[0] == uInt(-1) && f.neighbor_cells[1] == uInt(-1)) {
                throw std::runtime_error("Orphan face detected");
            }
        }, face);
    }

    // 验证单元面索引有效性
    for (uInt cell_id = 0; cell_id < m_cells.size(); ++cell_id) {
        std::visit([&](auto&& cell) {
            for (uInt face_id : cell.faces) {
                if (face_id >= m_faces.size()) {
                    throw std::runtime_error("Invalid face index in cell");
                }
            }
        }, m_cells[cell_id]);
    }
}


// 核心剖分实现
void GeneralMesh::split_hex_impl(bool prefer_split5) {
    std::vector<GeneralCell> new_cells;
    std::vector<GeneralFace> new_faces;
    std::unordered_map<uInt, uInt> face_map;

    auto add_face = [&](const GeneralFace& face) -> uInt {
        const uInt hash = face_hash(face);
        if (auto it = face_map.find(hash); it != face_map.end()) {
            update_face_neighbor(new_faces[it->second], new_cells.size());
            return it->second;
        }
        new_faces.push_back(create_new_face(face, new_cells.size()));
        face_map[hash] = new_faces.size() - 1;
        return new_faces.size() - 1;
    };

    for (size_t cellId = 0; cellId < m_cells.size(); ++cellId) {
        if (!std::holds_alternative<Hexahedron>(m_cells[cellId])) {
            // new_cells.push_back(m_cells[cellId]);
            // continue;
            // 处理非六面体单元的面转换
            GeneralCell new_cell = convert_non_hex_cell(m_cells[cellId], add_face);
            new_cells.push_back(new_cell);
            continue;
        }

        Hexahedron& hex = std::get<Hexahedron>(m_cells[cellId]);
        bool split_success = false;

        // 根据优先级选择剖分方式
        if (prefer_split5) {
            split_success = try_split5(hex, new_cells, new_faces, add_face);
            if (!split_success) split_success = try_split6(hex, new_cells, new_faces, add_face);
        } else {
            split_success = try_split6(hex, new_cells, new_faces, add_face);
            if (!split_success) split_success = try_split5(hex, new_cells, new_faces, add_face);
        }

        if (!split_success) {
            throw std::runtime_error("Hexahedron splitting failed");
        }
    }

    // 第二阶段：后处理所有Quad面

    // 更新网格数据
    m_cells = std::move(new_cells);
    m_faces = std::move(new_faces);

    mark_all_quad_diagonals();
    process_non_hex_cells();
}


// 5剖分尝试
template <typename Func>
bool GeneralMesh::try_split5(const Hexahedron& hex, 
                std::vector<GeneralCell>& new_cells,
                std::vector<GeneralFace>& new_faces,
                Func&& add_face) {
    const auto& nodes = hex.nodes;
    Split5Candidate candidate = find_split5_candidate(hex);

    if (!candidate.valid) return false;

    // 生成5个四面体
    const auto tets = generate_split5_tets(nodes, candidate);
    for (const auto& tet : tets) {
        std::array<uInt,4> face_ids = create_tet_faces(tet, add_face);
        new_cells.push_back(Tetrahedron{tet, face_ids});
    }

    update_split5_faces(hex, candidate);
    return true;
}

// 6剖分尝试
template <typename Func>
bool GeneralMesh::try_split6(const Hexahedron& hex,
                std::vector<GeneralCell>& new_cells,
                std::vector<GeneralFace>& new_faces,
                Func&& add_face) {
    std::optional<uInt> axis = find_split6_axis(hex);
    // debug(axis.has_value());
    if (!axis.has_value()) return false;

    const auto prisms = split_hex_prisms(hex, axis.value());
    for (const auto& prism : prisms) {
        std::array<uInt,5> face_ids = create_prism_faces(prism, add_face);
        new_cells.push_back(Prism{prism, face_ids});
    }

    update_split6_faces(hex, axis.value());
    return true;
}



// 六面体剖分结束后，处理所有非六面体
void GeneralMesh::process_non_hex_cells() {
    std::vector<GeneralCell> new_cells;
    std::vector<GeneralFace> new_faces;
    std::unordered_map<uInt, uInt> face_map;
    
    auto add_face = [&](const GeneralFace& face) -> uInt {
        const uInt hash = face_hash(face);
        if (auto it = face_map.find(hash); it != face_map.end()) {
            update_face_neighbor(new_faces[it->second], new_cells.size());
            return it->second;
        }
        new_faces.push_back(create_new_face(face, new_cells.size()));
        face_map[hash] = new_faces.size() - 1;
        return new_faces.size() - 1;
    };

    for (const auto& cell : m_cells) {
        std::visit([&](const auto& c) {
            using T = std::decay_t<decltype(c)>;
            if constexpr (std::is_same_v<T, Hexahedron>) {
                // 已处理过
            } else if constexpr (std::is_same_v<T, Prism>) {
                // debug(vector2u{new_cells.size(),new_faces.size()});
                process_prism(c, new_cells, add_face);
                // debug(vector2u{new_cells.size(),new_faces.size()});
            } else if constexpr (std::is_same_v<T, Pyramid>) {
                process_pyramid(c, new_cells, add_face);
            } else if constexpr (std::is_same_v<T, Tetrahedron>) {
                process_tetrahedron(c, new_cells, add_face);
            }
        }, cell);
    }

    m_cells = std::move(new_cells);
    m_faces = std::move(new_faces);
}


// 三棱柱，核心步骤：找共享剖分
template <typename Func>
void GeneralMesh::process_prism(const Prism& prism, 
                std::vector<GeneralCell>& new_cells,
                Func&& add_face) {
    const auto& nodes = prism.nodes;
    std::array<vector2u,3> diags;

    // 获取三个侧面剖分信息
    for (uInt i = 0; i < 3; ++i) {
        const QuadFace& qf = std::get<QuadFace>(m_faces[prism.faces[i+2]]);
        diags[i] = qf.diagonal_nodes;
    }

    // 判断剖分模式：统一模式需6个唯一节点
    std::unordered_set<uInt> unique_nodes;
    for (const auto& d : diags) {
        unique_nodes.insert(d[0]);
        unique_nodes.insert(d[1]);
    }

    if (unique_nodes.size() == 6) { // 统一剖分模式
        throw std::runtime_error("diags split for prism error 333.");
        // 生成三个标准四面体
        const std::array<vector4u,3> tets = {{
            {diags[0][0], diags[0][1], diags[1][0], diags[1][1]},
            {diags[1][0], diags[1][1], diags[2][0], diags[2][1]},
            {diags[2][0], diags[2][1], diags[0][0], diags[0][1]}
        }};

        for (const auto& tet : tets) {
            new_cells.push_back(create_tet(tet, add_face));
        }
    } else { // 混合剖分模式
        vector4u set_diag{uInt(-1),uInt(-1),uInt(-1),uInt(-1)};
        vector6u flat_diags{diags[0][0],diags[0][1],diags[1][0],diags[1][1],diags[2][0],diags[2][1]};
        for(const auto& n:unique_nodes){
            if(std::count(flat_diags.begin(),flat_diags.end(),n)==2){
                if(std::count(nodes.begin(),nodes.end()-3,n)){
                    set_diag[0] = n;
                }
                else{
                    set_diag[1] = n;
                }
            }
            else{
                if(set_diag[2]==uInt(-1)){
                    set_diag[2] = n;
                }
                else{
                    set_diag[3] = n;
                }
            }
        }

        // 生成三个标准四面体
        std::array<vector4u,3> tets = {{
            {set_diag[0], set_diag[1], set_diag[2], set_diag[3]},
            {nodes[0], nodes[1], nodes[2], set_diag[1]},
            {nodes[3], nodes[4], nodes[5], set_diag[0]}
        }};


        for (const auto& tet : tets) {
            new_cells.push_back(create_tet(tet, add_face));
        }
    }
}


// 四棱锥，核心步骤：找到四边形
template <typename Func>
void GeneralMesh::process_pyramid(const Pyramid& pyramid,
                    std::vector<GeneralCell>& new_cells,
                    Func&& add_face) {
    const auto& nodes = pyramid.nodes;
    const QuadFace& base_face = std::get<QuadFace>(m_faces[pyramid.faces[0]]);

    // 根据底面剖分生成2个四面体
    if (base_face.diagonal_nodes == vector2u{nodes[0], nodes[2]}) {
        new_cells.push_back(create_tet({nodes[0], nodes[1], nodes[2], nodes[4]}, add_face));
        new_cells.push_back(create_tet({nodes[0], nodes[2], nodes[3], nodes[4]}, add_face));
    } else {
        new_cells.push_back(create_tet({nodes[1], nodes[2], nodes[3], nodes[4]}, add_face));
        new_cells.push_back(create_tet({nodes[1], nodes[3], nodes[0], nodes[4]}, add_face));
    }
}

// 辅助函数：转换非六面体单元的面索引
template <typename Func>
GeneralCell GeneralMesh::convert_non_hex_cell(const GeneralCell& cell, 
                                            Func&& add_face) {
    return std::visit([&](const auto& c) -> GeneralCell {
        using T = std::decay_t<decltype(c)>;
        if constexpr (std::is_same_v<T, Prism>) {
            std::array<uInt,5> new_faces;
            for (size_t i = 0; i < 5; ++i) {
                new_faces[i] = std::forward<Func>(add_face)(m_faces[c.faces[i]]);
            }
            return Prism{c.nodes, new_faces};
        } else if constexpr (std::is_same_v<T, Pyramid>) {
            std::array<uInt,5> new_faces;
            for (size_t i = 0; i < 5; ++i) {
                new_faces[i] = std::forward<Func>(add_face)(m_faces[c.faces[i]]);
            }
            return Pyramid{c.nodes, new_faces};
        } else if constexpr (std::is_same_v<T, Tetrahedron>) {
            std::array<uInt,4> new_faces;
            for (size_t i = 0; i < 4; ++i) {
                new_faces[i] = std::forward<Func>(add_face)(m_faces[c.faces[i]]);
            }
            return Tetrahedron{c.nodes, new_faces};
        } else {
            return cell; // 其他类型暂不处理
        }
    }, cell);
}

// 四面体：直接添加
template <typename Func>
void GeneralMesh::process_tetrahedron(const Tetrahedron& tet,
                    std::vector<GeneralCell>& new_cells,
                    Func&& add_face) {
    const auto& nodes = tet.nodes;
    new_cells.push_back(create_tet(nodes, add_face));
}

// 创建四面体单元
template <typename Func>
Tetrahedron GeneralMesh::create_tet(const vector4u& nodes, Func&& add_face) {
    return Tetrahedron{
        nodes,
        {{
            std::forward<Func>(add_face)(TriangleFace{{nodes[0], nodes[1], nodes[2]}}),
            std::forward<Func>(add_face)(TriangleFace{{nodes[0], nodes[1], nodes[3]}}),
            std::forward<Func>(add_face)(TriangleFace{{nodes[1], nodes[2], nodes[3]}}),
            std::forward<Func>(add_face)(TriangleFace{{nodes[2], nodes[0], nodes[3]}})
        }}
    };
}


GeneralMesh::Split5Candidate GeneralMesh::find_split5_candidate(const Hexahedron& hex) const {
    Split5Candidate result;
    bool can_0257 = true, can_1346 = true;

    for (uInt faceId : hex.faces) {
        const QuadFace& qf = std::get<QuadFace>(m_faces[faceId]);
        if (qf.diagonal_nodes[0] == uInt(-1)) continue;

        can_0257 &= is_in_set(qf.diagonal_nodes, {hex.nodes[0], hex.nodes[2], hex.nodes[5], hex.nodes[7]});
        can_1346 &= is_in_set(qf.diagonal_nodes, {hex.nodes[1], hex.nodes[3], hex.nodes[4], hex.nodes[6]});
    }

    result.valid = can_0257 || can_1346;
    result.use_0257 = can_0257;
    return result;
}

// 6剖分轴向查找
std::optional<uInt> GeneralMesh::find_split6_axis(const Hexahedron& hex) const {
    for (uInt axis = 0; axis < 3; ++axis) {
        uInt f1 = hex.faces[axis*2], f2 = hex.faces[axis*2+1];
        if (check_split6_axis(f1, f2)) {
            return axis;
        }
    }
    return std::nullopt;
}

// 通用辅助函数
template<typename T>
bool GeneralMesh::is_in_set(const T& values, const std::initializer_list<uInt>& set) const {
    for (const auto& v : values) {
        if (std::find(set.begin(), set.end(), v) == set.end()) {
            return false;
        }
    }
    return true;
}

bool GeneralMesh::check_split6_axis(uInt f1, uInt f2) const {
    const QuadFace& qf1 = std::get<QuadFace>(m_faces[f1]);
    const QuadFace& qf2 = std::get<QuadFace>(m_faces[f2]);
    // debug(qf1.diagonal_nodes);
    // debug(qf2.diagonal_nodes);
    
    const bool both_unset = (qf1.diagonal_nodes[0] == uInt(-1)) || 
                            (qf2.diagonal_nodes[0] == uInt(-1));
    const bool same_dir = (qf1.diagonal_nodes[0] == qf2.diagonal_nodes[0])||(qf1.diagonal_nodes[0] == qf2.diagonal_nodes[1]);
    
    return both_unset || same_dir;
}

// 生成四面体面
template <typename Func>
std::array<uInt,4> GeneralMesh::create_tet_faces(const vector4u& nodes, Func&& add_face) {
    std::array<uInt,4> face_ids;
    // for (uInt i = 0; i < 4; ++i) {
    //     vector3u face_nodes;
    //     std::copy(nodes.begin(), nodes.end()-1, face_nodes.begin());
    //     face_nodes[i] = nodes[3]; // 替换第i个顶点
    //     face_ids[i] = add_face(TriangleFace{face_nodes});
    // }
    face_ids[0] = std::forward<Func>(add_face)(TriangleFace{{nodes[3],nodes[1],nodes[2]}});
    face_ids[1] = std::forward<Func>(add_face)(TriangleFace{{nodes[0],nodes[3],nodes[2]}});
    face_ids[2] = std::forward<Func>(add_face)(TriangleFace{{nodes[0],nodes[1],nodes[3]}});
    face_ids[3] = std::forward<Func>(add_face)(TriangleFace{{nodes[0],nodes[1],nodes[2]}});
    return face_ids;
}

// 生成三棱柱面
template <typename Func>
std::array<uInt,5> GeneralMesh::create_prism_faces(const vector6u& nodes, Func&& add_face) {
    return {{
        std::forward<Func>(add_face)(TriangleFace{{nodes[0], nodes[1], nodes[2]}}),
        std::forward<Func>(add_face)(TriangleFace{{nodes[3], nodes[4], nodes[5]}}),
        std::forward<Func>(add_face)(QuadFace{{nodes[0], nodes[1], nodes[4], nodes[3]}}),
        std::forward<Func>(add_face)(QuadFace{{nodes[1], nodes[2], nodes[5], nodes[4]}}),
        std::forward<Func>(add_face)(QuadFace{{nodes[2], nodes[0], nodes[3], nodes[5]}})
    }};
}

// 更新原始面信息（5剖分）
void GeneralMesh::update_split5_faces(const Hexahedron& hex, const Split5Candidate& candidate) {
    const auto& node_set = candidate.use_0257 ? 
        std::array{hex.nodes[0], hex.nodes[2], hex.nodes[5], hex.nodes[7]} :
        std::array{hex.nodes[1], hex.nodes[3], hex.nodes[4], hex.nodes[6]};

    for (uInt faceId : hex.faces) {
        QuadFace& qf = std::get<QuadFace>(m_faces[faceId]);
        if (qf.diagonal_nodes[0] != uInt(-1)) continue;

        uInt count = 0;
        for (uInt n : node_set) {
            if (std::count(qf.nodes.begin(), qf.nodes.end(), n) > 0) {
                qf.diagonal_nodes[count++] = n;
                if (count == 2) break;
            }
        }
    }
}

// 更新原始面信息（6剖分）
void GeneralMesh::update_split6_faces(const Hexahedron& hex, uInt axis) {
    const auto& face_pair = get_opposite_faces(hex, axis);
    auto set_diagonal = [&](uInt faceId) {
        QuadFace& qf = std::get<QuadFace>(m_faces[faceId]);
        if (qf.diagonal_nodes[0] != uInt(-1)) return;

        const vector3f p0 = m_points[qf.nodes[0]];
        const vector3f p1 = m_points[qf.nodes[1]];
        const vector3f p2 = m_points[qf.nodes[2]];
        const vector3f p3 = m_points[qf.nodes[3]];
        
        qf.diagonal_nodes = (vec_length(p0-p2) < vec_length(p1-p3)) ?
            vector2u{qf.nodes[0], qf.nodes[2]} :
            vector2u{qf.nodes[1], qf.nodes[3]};
    };

    set_diagonal(face_pair.first);
    set_diagonal(face_pair.second);
}

std::pair<uInt,uInt> GeneralMesh::get_opposite_faces(const Hexahedron& hex, uInt axis) const {
    switch(axis) {
        case 0: return {hex.faces[0], hex.faces[1]};
        case 1: return {hex.faces[2], hex.faces[3]};
        case 2: return {hex.faces[4], hex.faces[5]};
        default: throw std::invalid_argument("Invalid axis");
    }
}

std::array<vector4u,5> GeneralMesh::generate_split5_tets(const vector8u& nodes, const Split5Candidate& c) {
    if (c.use_0257) {
        return {{ 
            {nodes[0], nodes[1], nodes[2], nodes[5]},
            {nodes[0], nodes[2], nodes[3], nodes[7]},
            {nodes[7], nodes[6], nodes[5], nodes[2]},
            {nodes[7], nodes[5], nodes[4], nodes[0]},
            {nodes[0], nodes[2], nodes[5], nodes[7]}
        }};
    }
    return {{
        {nodes[1], nodes[2], nodes[3], nodes[6]},
        {nodes[1], nodes[3], nodes[0], nodes[4]},
        {nodes[6], nodes[5], nodes[4], nodes[1]},
        {nodes[6], nodes[4], nodes[7], nodes[3]},
        {nodes[1], nodes[3], nodes[4], nodes[6]}
    }};
}

std::array<vector6u,2> GeneralMesh::split_hex_prisms(const Hexahedron& hex, uInt axis) {
    switch(axis) {
        case 0: return {{
            {hex.nodes[0], hex.nodes[1], hex.nodes[2], hex.nodes[4], hex.nodes[5], hex.nodes[6]},
            {hex.nodes[2], hex.nodes[3], hex.nodes[0], hex.nodes[6], hex.nodes[7], hex.nodes[4]}
        }};
        case 1: return {{
            {hex.nodes[0], hex.nodes[1], hex.nodes[5], hex.nodes[3], hex.nodes[2], hex.nodes[6]},
            {hex.nodes[3], hex.nodes[0], hex.nodes[5], hex.nodes[7], hex.nodes[4], hex.nodes[6]}
        }};
        case 2: return {{
            {hex.nodes[0], hex.nodes[2], hex.nodes[3], hex.nodes[4], hex.nodes[6], hex.nodes[7]},
            {hex.nodes[1], hex.nodes[0], hex.nodes[3], hex.nodes[5], hex.nodes[4], hex.nodes[7]}
        }};
        default: throw std::invalid_argument("Invalid split axis");
    }
}



