#pragma once

#include "mesh/face_type.h"
#include "mesh/cell_type.h"



//========================= 一般网格类 =========================//
class GeneralMesh {
public:
    // 核心数据存储（内存连续）
    std::vector<vector3f> m_points;
    std::vector<GeneralFace> m_faces;
    std::vector<GeneralCell> m_cells;

public:
    //---------------- 核心构建流程 ----------------
    void rebuild_cell_topology();

    //---------------- 网格验证 ----------------
    void validate_mesh() const ;

    // 剖分入口函数
    void split_hex5_scan() { split_hex_impl(true); }
    void split_hex6_scan() { split_hex_impl(false); }

private:
    // 核心剖分实现
    void split_hex_impl(bool prefer_split5);

    // 5剖分尝试
    template <typename Func>
    bool try_split5(const Hexahedron& hex, 
                   std::vector<GeneralCell>& new_cells,
                   std::vector<GeneralFace>& new_faces,
                   Func&& add_face);

    // 6剖分尝试
    template <typename Func>
    bool try_split6(const Hexahedron& hex,
                   std::vector<GeneralCell>& new_cells,
                   std::vector<GeneralFace>& new_faces,
                   Func&& add_face);

    
    // 六面体剖分结束后，处理所有非六面体
    void process_non_hex_cells();

    // 三棱柱，核心步骤：找共享剖分
    template <typename Func>
    void process_prism(const Prism& prism, 
                 std::vector<GeneralCell>& new_cells,
                 Func&& add_face);

    // 四棱锥，核心步骤：找到四边形
    template <typename Func>
    void process_pyramid(const Pyramid& pyramid,
                        std::vector<GeneralCell>& new_cells,
                        Func&& add_face);
    // 四面体：直接添加
    template <typename Func>
    void process_tetrahedron(const Tetrahedron& tet,
                        std::vector<GeneralCell>& new_cells,
                        Func&& add_face);

    // 创建四面体单元
    template <typename Func>
    Tetrahedron create_tet(const vector4u& nodes, Func&& add_face);


    // 辅助函数：转换非六面体单元的面索引
    template <typename Func>
    GeneralCell convert_non_hex_cell(const GeneralCell& cell, Func&& add_face);

    // 辅助函数：标记所有Quad面的剖分对角线
    void mark_all_quad_diagonals();

    // 辅助函数：检查四边形是否合法构成三棱柱剖分，并标记必须剖分的面
    void check_prism_faces_diags(uInt flag = 0);

    // 辅助函数：检查三角形是否存在
    bool check_tri_exist(const vector3u& nodes, 
                        const std::unordered_set<uInt>& hashes) const;

    // 辅助函数：生成排序后节点的哈希（似乎没用到）
    template<uInt N>
    uInt hash_sorted_nodes(const std::array<uInt,N>& nodes) const;
    // 辅助函数：选择几何最短对角线
    vector2u choose_shortest_diagonal(const vector4u& nodes) const;
    vector2u choose_shortest_diagonal_inverse(const vector4u& nodes) const;

    // 辅助函数：面哈希
    uInt face_hash(const GeneralFace& face) const;

    template<typename Iter>
    size_t hash_range(Iter begin, Iter end) const;

    // 更新面邻接信息
    void update_face_neighbor(GeneralFace& face, uInt cell_id) const;

    // 创建新面
    GeneralFace create_new_face(const GeneralFace& proto, uInt cell_id) const;

    // 5剖分候选查找
    struct Split5Candidate {
        bool valid = false;
        bool use_0257 = true;
    };

    Split5Candidate find_split5_candidate(const Hexahedron& hex) const;

    // 6剖分轴向查找
    std::optional<uInt> find_split6_axis(const Hexahedron& hex) const;

    // 通用辅助函数
    template<typename T>
    bool is_in_set(const T& values, const std::initializer_list<uInt>& set) const;

    bool check_split6_axis(uInt f1, uInt f2) const;

    // 生成四面体面
    template <typename Func>
    std::array<uInt,4> create_tet_faces(const vector4u& nodes, Func&& add_face);

    // 生成三棱柱面
    template <typename Func>
    std::array<uInt,5> create_prism_faces(const vector6u& nodes, Func&& add_face);

    // 更新原始面信息（5剖分）
    void update_split5_faces(const Hexahedron& hex, const Split5Candidate& candidate);

    // 更新原始面信息（6剖分）
    void update_split6_faces(const Hexahedron& hex, uInt axis);

    std::pair<uInt,uInt> get_opposite_faces(const Hexahedron& hex, uInt axis) const;

    std::array<vector4u,5> generate_split5_tets(const vector8u& nodes, const Split5Candidate& c);

    std::array<vector6u,2> split_hex_prisms(const Hexahedron& hex, uInt axis);


};