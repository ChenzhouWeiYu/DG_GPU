#pragma once

#include "base/type.h"

//---------------- 面类型 ----------------
struct TriangleFace {
    vector3u nodes;
    vector2u neighbor_cells = {uInt(-1), uInt(-1)};
    
    void reorder() {
        // 三角形的节点序不需要处理
    }
};

struct QuadFace {
    vector4u nodes;
    vector2u neighbor_cells = {uInt(-1), uInt(-1)};
    vector2u diagonal_nodes = {uInt(-1), uInt(-1)};; // 剖分对角线信息，记录对角节点的全局索引
    
    // 判断四边形顶点顺序是否满足右手法则
    bool is_rhs(const std::vector<vector3f>& points) const ;

    // 重新排序四边形顶点保证右手法则和对角线正确
    void reorder(const std::vector<vector3f>& points) ;

    // 选择最短的对角线进行剖分
    void split_diagonal(const std::vector<vector3f>& points) ;
};

using GeneralFace = std::variant<TriangleFace, QuadFace>;
