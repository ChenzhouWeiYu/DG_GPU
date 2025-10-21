#pragma once

#include "mesh/general_mesh/face_type.h"
#include "mesh/general_mesh/hexahedron_cell.h"
#include "mesh/general_mesh/prism_cell.h"
#include "mesh/general_mesh/pyramid_cell.h"



struct Tetrahedron {
    vector4u nodes;
    vector4u faces;
    
    void reorder(const std::vector<vector3f>& points, 
                const std::vector<GeneralFace>& all_faces) ;
};

using GeneralCell = std::variant<Hexahedron, Prism, Pyramid, Tetrahedron>;