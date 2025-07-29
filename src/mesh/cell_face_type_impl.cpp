#include "mesh/cell_type.h"

void Tetrahedron::reorder(const std::vector<vector3f>& points, 
                const std::vector<GeneralFace>& all_faces)  {
    // 无需重排面
    vector3f v1 = points[nodes[1]] - points[nodes[0]];
    vector3f v2 = points[nodes[2]] - points[nodes[0]];
    vector3f v3 = points[nodes[3]] - points[nodes[0]];
    if(vec_dot(vec_cross(v1,v2),v3)<0){
        nodes =  {nodes[0],nodes[2],nodes[1],nodes[3]};
    }
}





void Hexahedron::reorder(const std::vector<vector3f>& points, 
            const std::vector<GeneralFace>& all_faces) {
    // Step 1: 提取所有四边形面的节点
    std::array<vector4u, 6> face_nodes;
    for (uInt i = 0; i < 6; ++i) {
        face_nodes[i] = std::get<QuadFace>(all_faces[faces[i]]).nodes;
    }

    // Step 2: 识别三对相对面
    std::array<std::pair<uInt, uInt>, 3> opposite_pairs = find_opposite_pairs(face_nodes);

    // Step 3: 确定底面（平均Z最低的面）
    uInt bottom_pair_idx = find_bottom_pair(opposite_pairs, face_nodes, points);
    auto& bottom_pair = opposite_pairs[bottom_pair_idx];
    
    // Step 4: 排列底面节点（极角排序）
    vector4u base_nodes = sort_quad_by_polar(face_nodes[bottom_pair.first], points);
    vector4u top_nodes = align_top_nodes(base_nodes, face_nodes[bottom_pair.second], points);

    // Step 5: 构建六面体节点顺序
    nodes = {
        base_nodes[0], base_nodes[1], base_nodes[2], base_nodes[3],
        top_nodes[0],  top_nodes[1],  top_nodes[2],  top_nodes[3]
    };

    // Step 6: 重建面顺序 [底面, 顶面, 前, 右, 后, 左]
    rebuild_face_order(base_nodes, top_nodes, face_nodes);
}

std::array<std::pair<uInt, uInt>, 3> Hexahedron::find_opposite_pairs(const std::array<vector4u,6>& faces) {
    std::array<std::pair<uInt, uInt>, 3> pairs;
    std::unordered_set<uInt> matched;
    
    uInt pair_idx = 0;
    for (uInt i = 0; i < 6; ++i) {
        if (matched.count(i)) continue;
        for (uInt j = i+1; j < 6; ++j) {
            if (is_opposite(faces[i], faces[j])) {
                pairs[pair_idx++] = {i, j};
                matched.insert(i);
                matched.insert(j);
                break;
            }
        }
    }
    return pairs;
}

// 判断是否为相对面
bool Hexahedron::is_opposite(const vector4u& a, const vector4u& b) const {
    std::unordered_set<uInt> s(a.begin(), a.end());
    for (uInt n : b) if (s.count(n)) return false;
    return true;
}

// 通过平均Z坐标找到最下面的面对
uInt Hexahedron::find_bottom_pair(const std::array<std::pair<uInt, uInt>,3>& pairs,
                        const std::array<vector4u,6>& faces,
                        const std::vector<vector3f>& points) const {
    uInt min_idx = 0;
    Scalar min_z = std::numeric_limits<Scalar>::max();
    
    for (uInt i = 0; i < 3; ++i) {
        Scalar z_sum = 0;
        for (uInt n : faces[pairs[i].first]) z_sum += points[n][2];
        if (z_sum < min_z) {
            min_z = z_sum;
            min_idx = i;
        }
    }
    return min_idx;
}

// 极角排序四边形节点（绕几何中心逆时针）
vector4u Hexahedron::sort_quad_by_polar(vector4u quad, const std::vector<vector3f>& points) {
    // 计算几何中心
    vector3f center = {0,0,0};
    for (uInt n : quad) center += points[n];
    center /= 4;

    // 按极角排序
    std::sort(quad.begin(), quad.end(), [&](uInt a, uInt b) {
        vector3f va = points[a] - center;
        vector3f vb = points[b] - center;
        return std::atan2(va[1], va[0]) < std::atan2(vb[1], vb[0]);
    });
    // 确保逆时针（通过顶点0到1到2的向量叉积）
    vector3f v01 = points[quad[1]] - points[quad[0]];
    vector3f v02 = points[quad[2]] - points[quad[0]];
    if (v01[0]*v02[1] - v01[1]*v02[0] < 0) { // 仅需二维判断
        std::swap(quad[1], quad[3]);
    }
    return quad;
}

// 对齐顶面节点到底面
vector4u Hexahedron::align_top_nodes(const vector4u& base, const vector4u& top_raw,
                        const std::vector<vector3f>& points) {
    vector4u top = top_raw;
    // 建立顶点映射：每个顶面节点对应最近的底面节点
    std::array<uInt,4> mapping;
    for (uInt i = 0; i < 4; ++i) {
        Scalar min_dist = std::numeric_limits<Scalar>::max();
        for (uInt j = 0; j < 4; ++j) {
            Scalar dx = points[base[i]][0] - points[top[j]][0];
            Scalar dy = points[base[i]][1] - points[top[j]][1];
            Scalar dist = dx*dx + dy*dy; // 仅考虑XY平面距离
            if (dist < min_dist) {
                min_dist = dist;
                mapping[i] = j;
            }
        }
    }
    // 重新排列顶面节点
    return {top[mapping[0]], top[mapping[1]], top[mapping[2]], top[mapping[3]]};
}

// 根据节点顺序重建面索引
void Hexahedron::rebuild_face_order(const vector4u& base, const vector4u& top,
                        std::array<vector4u,6>& face_nodes) {
    // 预期的标准面连接模式
    const std::array<vector4u,6> expected_faces = {{
        {base[0], base[1], base[2], base[3]},  // 底面
        {top[0],  top[1],  top[2],  top[3]},   // 顶面
        {base[0], base[1], top[1],  top[0]},   // 前
        {base[1], base[2], top[2],  top[1]},   // 右
        {base[2], base[3], top[3],  top[2]},   // 后
        {base[3], base[0], top[0],  top[3]}    // 左
    }};

    // 匹配并更新面节点顺序
    for (auto& face : face_nodes) {
        for (const auto& pattern : expected_faces) {
            if (is_cyclic_match(face, pattern)) {
                face = pattern;
                break;
            }
        }
    }
}

// 循环匹配检查
bool Hexahedron::is_cyclic_match(const vector4u& a, const vector4u& b) const {
    for (uInt offset = 0; offset < 4; ++offset) {
        bool match = true;
        for (uInt i = 0; i < 4; ++i) {
            if (a[i] != b[(i+offset)%4]) {
                match = false;
                break;
            }
        }
        if (match) return true;
    }
    return false;
}





































void Prism::reorder(const std::vector<vector3f>& points, 
            const std::vector<GeneralFace>& all_faces) {
    // Step 1: 提取两个三角形面
    vector2u tri_faces;
    uInt tri_count = 0;
    for (uInt i = 0; i < 5; ++i) {
        if (std::holds_alternative<TriangleFace>(all_faces[faces[i]])) {
            tri_faces[tri_count++] = faces[i];
            if (tri_count == 2) break;
        }
    }

    // Step 2: 确定底面和顶面（通过面索引位置）
    auto& base_face = std::get<TriangleFace>(all_faces[tri_faces[0]]);
    auto& top_face = std::get<TriangleFace>(all_faces[tri_faces[1]]);
    vector3u base_nodes = base_face.nodes;
    vector3u top_nodes = top_face.nodes;

    // Step 3: 极角排序底面节点（仅需二维投影）
    {
        vector3f center = {0,0,0};
        for (uInt n : base_nodes) center += points[n];
        center /= 3;

        std::sort(base_nodes.begin(), base_nodes.end(), [&](uInt a, uInt b) {
            vector3f va = points[a] - center;
            vector3f vb = points[b] - center;
            return std::atan2(va[1], va[0]) < std::atan2(vb[1], vb[0]);
        });

        // 验证二维法线方向
        vector3f v1 = points[base_nodes[1]] - points[base_nodes[0]];
        vector3f v2 = points[base_nodes[2]] - points[base_nodes[0]];
        if (v1[0]*v2[1] - v1[1]*v2[0] < 0) {
            std::swap(base_nodes[1], base_nodes[2]);
        }
    }

    // Step 4: 安全处理侧面四边形
    std::array<vector4u,3> quad_nodes;
    uInt quad_count = 0;
    for (uInt i = 0; i < 5; ++i) { // 仅处理侧面面
        if (std::holds_alternative<QuadFace>(all_faces[faces[i]])) { // 关键修复点
            quad_nodes[quad_count++] = std::get<QuadFace>(all_faces[faces[i]]).nodes;
        } else {
            // throw std::runtime_error("Prism has non-quad lateral face");
        }
    }

    // Step 5: 建立顶面节点映射
    std::array<uInt,3> top_mapping;
    for (uInt i = 0; i < 3; ++i) {
        const auto& quad = quad_nodes[i];
        uInt a = base_nodes[i];
        uInt b = base_nodes[(i+1)%3];
        
        // 查找对应的顶面边
        for (uInt j = 0; j < 3; ++j) {
            uInt c = top_nodes[j];
            uInt d = top_nodes[(j+1)%3];
            if (contains_edge(quad, c, d)) {
                top_mapping[i] = j;
                break;
            }
        }
    }

    // Step 6: 对齐顶面节点
    vector3u ordered_top;
    for (uInt i = 0; i < 3; ++i) {
        ordered_top[i] = top_nodes[top_mapping[i]];
    }
    top_nodes = ordered_top;

    // Step 7: 更新节点和面顺序
    nodes = {base_nodes[0], base_nodes[1], base_nodes[2],
                top_nodes[0],  top_nodes[1],  top_nodes[2]};
    faces = {tri_faces[0], tri_faces[1], faces[2], faces[3], faces[4]};
}

// 检查四边形是否包含指定边
bool Prism::contains_edge(const vector4u& quad, uInt a, uInt b) const {
    for (uInt i = 0; i < 4; ++i) {
        if ((quad[i] == a && quad[(i+1)%4] == b) ||
            (quad[i] == b && quad[(i+1)%4] == a)) {
            return true;
        }
    }
    return false;
}





















void Pyramid::reorder(const std::vector<vector3f>& points,
            const std::vector<GeneralFace>& all_faces) {
    // Step 1: 识别底面四边形
    uInt base_face = 0;
    for (uInt i = 0; i < 5; ++i) {
        if (std::holds_alternative<QuadFace>(all_faces[faces[i]])) {
            base_face = i;
            break;
        }
    }

    // Step 2: 调整底面节点顺序
    vector4u base_nodes = std::get<QuadFace>(all_faces[faces[base_face]]).nodes;
    sort_quad_ccw(base_nodes, points);

    // Step 3: 确定顶点
    uInt apex = find_apex(base_nodes, all_faces);

    // Step 4: 重建节点顺序
    nodes = {base_nodes[0], base_nodes[1], base_nodes[2], base_nodes[3], apex};
}

// 四边形逆时针排序
void Pyramid::sort_quad_ccw(vector4u& quad, const std::vector<vector3f>& points) {
    vector3f center = {0,0,0};
    for (uInt n : quad) center += points[n];
    center /= 4;

    std::sort(quad.begin(), quad.end(), [&](uInt a, uInt b) {
        vector3f va = points[a] - center;
        vector3f vb = points[b] - center;
        return std::atan2(va[1], va[0]) < std::atan2(vb[1], vb[0]);
    });

    vector3f v1 = points[quad[1]] - points[quad[0]];
    vector3f v2 = points[quad[2]] - points[quad[0]];
    if (vec_dot(vec_cross(v1, v2), {0,0,1}) < 0) {
        std::reverse(quad.begin(), quad.end());
    }
}

// 查找顶点（不属于底面的唯一节点）
uInt Pyramid::find_apex(const vector4u& base, const std::vector<GeneralFace>& all_faces) const {
    std::unordered_set<uInt> base_set(base.begin(), base.end());
    for (uInt face_id : faces) {
        if (auto* tri = std::get_if<TriangleFace>(&all_faces[face_id])) {
            for (uInt n : tri->nodes) {
                if (!base_set.count(n)) return n;
            }
        }
    }
    return -1; // Should not reach here
}













// 判断四边形顶点顺序是否满足右手法则
bool QuadFace::is_rhs(const std::vector<vector3f>& points) const {
    const vector3f& p0 = points[nodes[0]];
    const vector3f& p1 = points[nodes[1]];
    const vector3f& p2 = points[nodes[2]];
    const vector3f& p3 = points[nodes[3]];
    
    // 计算三个边的向量
    const vector3f& v1 = p1-p0;
    const vector3f& v2 = p2-p1;
    const vector3f& v3 = p3-p2;
    
    // 计算两个三角形的法向量
    const vector3f& n1 = vec_cross(v1, v2);
    const vector3f& n2 = vec_cross(v2, v3);
    
    // 法向量应大致同向
    return vec_dot(n1, n2) > 0;
}

// 重新排序四边形顶点保证右手法则和对角线正确
void QuadFace::reorder(const std::vector<vector3f>& points) {
    if (is_rhs(points)) return;
    
    // 不满足右手法则时调整顶点顺序
    // 方案1：交换最后两个顶点
    std::swap(nodes[2], nodes[3]);
    
    // 二次验证
    if (!is_rhs(points)) {
        // 方案2：逆序整个四边形
        std::reverse(nodes.begin()+1, nodes.end());
    }
}

// 选择最短的对角线进行剖分
void QuadFace::split_diagonal(const std::vector<vector3f>& points) {
    const auto& a = points[nodes[0]];
    const auto& b = points[nodes[1]];
    const auto& c = points[nodes[2]];
    const auto& d = points[nodes[3]];
    
    // 比较对角线长度
    const Scalar len_ac = distance(a, c);
    const Scalar len_bd = distance(b, d);
    
    if (len_ac < len_bd) {
        diagonal_nodes = {nodes[0], nodes[2]}; // 对角线 a-c
    } else {
        diagonal_nodes = {nodes[1], nodes[3]}; // 对角线 b-d
    }
}
