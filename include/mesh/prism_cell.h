#pragma once

#include "mesh/face_type.h"

//==================== 三棱柱单元 ====================//
struct Prism {
    vector6u nodes; // [底面0,1,2 | 顶面3,4,5]
    vector5u faces; // [底面tri, 顶面tri, 侧面quad1, quad2, quad3]

    void reorder(const std::vector<vector3f>& points, 
                const std::vector<GeneralFace>& all_faces);

private:
    // 检查四边形是否包含指定边
    bool contains_edge(const vector4u& quad, uInt a, uInt b) const ;
};