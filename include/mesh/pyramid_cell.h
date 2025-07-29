#pragma once

#include "mesh/face_type.h"

//==================== 四棱锥单元 ====================//
struct Pyramid {
    vector5u nodes; // [底面0-3, 顶点4]
    vector5u faces; // [底面quad, 侧面tri1-4]

    void reorder(const std::vector<vector3f>& points,
                const std::vector<GeneralFace>& all_faces) ;

private:
    // 四边形逆时针排序
    void sort_quad_ccw(vector4u& quad, const std::vector<vector3f>& points) ;

    // 查找顶点（不属于底面的唯一节点）
    uInt find_apex(const vector4u& base, const std::vector<GeneralFace>& all_faces) const ;
};