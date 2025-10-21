#pragma once
#include "base/type.h"

/// 面类型：Internal 必须为 0
enum class FaceType : uint8_t {
    Internal = 0,  // 内部面
    Dirichlet,
    Neumann,
    Robin,
    Pseudo3DX,
    Pseudo3DY,
    Pseudo3DZ,
    Symmetry,
    Inflow,
    Outflow,
    Wall,
    WallTD,
    WallTN,
    WallTR,
    COUNT  // 总类型数
};

static constexpr uInt NumFaceTypes = static_cast<uInt>(FaceType::COUNT);