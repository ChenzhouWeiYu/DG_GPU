#pragma once
#include "base/type.h"
#include "mesh/face_type.h"
#include "mesh/computing_mesh.h"

// 注意：这些结构未来将在 __global__ kernel 内直接使用
struct GPUTetrahedron {
    vector4u nodes;
    // vector4u faces;
    vector4u neighbor_cells;
    // vector3f centroid;
    Scalar volume;
    Scalar m_h;
    DenseMatrix<3,3> JacMat;
    DenseMatrix<3,3> invJac;
};

struct GPUTriangleFace {
    vector3u nodes;
    vector3f normal;
    Scalar area;
    vector2u neighbor_cells;
    FaceType face_type;
    std::array<std::array<vector3f,3>,2> natural_coords;
};


/// POD 视图：传入 kernel
struct MeshView {
    const GPUTetrahedron* cells;
    const GPUTriangleFace* faces;
    const vector3f* points;

    const uInt* face_indices;   // 全局面索引重排序
    const uInt* face_offsets;   // 偏移数组，长度 = NumFaceTypes + 1

    uInt num_cells;
    uInt num_faces;
    uInt num_points;

    // 辅助函数
    HostDevice const GPUTetrahedron& getCell(uInt i) const { return cells[i]; }
    HostDevice const GPUTriangleFace& getFace(uInt i) const { return faces[i]; }
    HostDevice const vector3f& getPoint(uInt i) const { return points[i]; }

    /// 获取第 ft 类型的面数量
    HostDevice uInt numFacesOfType(uInt ft) const {
        return face_offsets[ft + 1] - face_offsets[ft];
    }

    /// 获取第 ft 类型的第 local_idx 个面的全局索引
    HostDevice uInt getFaceGlobalIndex(uInt ft, uInt local_idx) const {
        return face_indices[face_offsets[ft] + local_idx];
    }
};

/// GPU 网格管理类（非 POD，仅用于 host 管理）
class DeviceMesh {
public:
    DeviceMesh() = default;
    ~DeviceMesh() { release(); }

    void initialize_from(const ComputingMesh& cpu_mesh);
    void upload_to_gpu();
    void release();

    // 返回 POD 视图（用于 kernel）
    HostDevice MeshView view() const {
        MeshView v;
        v.cells = d_cells_;
        v.faces = d_faces_;
        v.points = d_points_;
        v.face_indices = d_face_indices_;
        v.face_offsets = d_face_offsets_;
        v.num_cells = num_cells_;
        v.num_faces = num_faces_;
        v.num_points = num_points_;
        return v;
    }

    HostDevice Scalar get_memory_usage() const;
    HostDevice uInt num_cells() const { return num_cells_; }
    HostDevice uInt num_faces() const { return num_faces_; }
    HostDevice uInt num_points() const { return num_points_; }
    
    /// 获取第 ft 类型的面数量
    HostDevice uInt numFacesOfType(uInt ft) const {
        return h_face_offsets_[ft + 1] - h_face_offsets_[ft];
    }

    /// 获取第 ft 类型的第 local_idx 个面的全局索引
    HostDevice uInt getFaceGlobalIndex(uInt ft, uInt local_idx) const {
        return h_face_indices_[h_face_offsets_[ft] + local_idx];
    }

private:
    // Host 端数据（用 malloc/free 管理）
    GPUTetrahedron* h_cells_ = nullptr;
    GPUTriangleFace* h_faces_ = nullptr;
    vector3f* h_points_ = nullptr;
    uInt* h_face_indices_ = nullptr;
    uInt* h_face_offsets_ = nullptr;

    // Device 端数据
    GPUTetrahedron* d_cells_ = nullptr;
    GPUTriangleFace* d_faces_ = nullptr;
    vector3f* d_points_ = nullptr;
    uInt* d_face_indices_ = nullptr;
    uInt* d_face_offsets_ = nullptr;

    uInt num_cells_ = 0;
    uInt num_faces_ = 0;
    uInt num_points_ = 0;
};

// class DeviceMesh {
// public:
//     DeviceMesh() = default;
//     ~DeviceMesh() { release_gpu(); }

//     // 从 CPU 完整网格初始化（不考虑 AMR 先）
//     void initialize_from(const ComputingMesh& cpu_mesh);

//     // 上传到 GPU
//     void upload_to_gpu();

//     // 释放 GPU 内存
//     void release_gpu();

//     // 内核调用用到的接口
//     HostDevice GPUTetrahedron* device_cells() { return d_cells_; }
//     HostDevice const GPUTetrahedron* device_cells() const { return d_cells_; }
//     std::vector<GPUTetrahedron> host_cells() const ;
//     HostDevice GPUTriangleFace* device_faces() { return d_faces_; }
//     HostDevice const GPUTriangleFace* device_faces() const { return d_faces_; }
//     std::vector<GPUTriangleFace> host_faces() const ;
//     HostDevice vector3f* device_points() { return d_points_; }
//     HostDevice const vector3f* device_points() const { return d_points_; }
//     std::vector<vector3f> host_points() const ;
//     HostDevice uInt num_cells() const { return num_cells_; }
//     HostDevice uInt num_faces() const { return num_faces_; }
//     HostDevice uInt num_points() const { return num_points_; }
//     Scalar get_memory_usage() const ;

// private:
//     std::vector<GPUTetrahedron> h_cells_;
//     std::vector<GPUTriangleFace> h_faces_;
//     std::vector<vector3f> h_points_;

//     GPUTetrahedron* d_cells_ = nullptr;
//     GPUTriangleFace* d_faces_ = nullptr;
//     vector3f* d_points_;

//     uInt num_cells_ = 0;
//     uInt num_faces_ = 0;
//     uInt num_points_ = 0;
// };
