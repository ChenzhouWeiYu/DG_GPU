#pragma once

#include "mesh/face_type.h"

//---------------- 六面体单元 ----------------
struct Hexahedron {
    vector8u nodes;
    vector6u faces;

    void reorder(const std::vector<vector3f>& points, 
                const std::vector<GeneralFace>& all_faces) ;

private:
    // 寻找三对相对面（无共享节点）
    std::array<std::pair<uInt, uInt>, 3> find_opposite_pairs(const std::array<vector4u,6>& faces) ;

    // 判断是否为相对面
    bool is_opposite(const vector4u& a, const vector4u& b) const ;

    // 通过平均Z坐标找到最下面的面对
    uInt find_bottom_pair(const std::array<std::pair<uInt, uInt>,3>& pairs,
                         const std::array<vector4u,6>& faces,
                         const std::vector<vector3f>& points) const;

    // 极角排序四边形节点（绕几何中心逆时针）
    vector4u sort_quad_by_polar(vector4u quad, const std::vector<vector3f>& points) ;

    // 对齐顶面节点到底面
    vector4u align_top_nodes(const vector4u& base, const vector4u& top_raw,
                            const std::vector<vector3f>& points) ;

    // 根据节点顺序重建面索引
    void rebuild_face_order(const vector4u& base, const vector4u& top,
                           std::array<vector4u,6>& face_nodes);

    // 循环匹配检查
    bool is_cyclic_match(const vector4u& a, const vector4u& b) const ;
};