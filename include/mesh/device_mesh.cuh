#pragma once
#include "base/type.h"
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
    BoundaryType boundaryType;
    std::array<std::array<vector3f,3>,2> natural_coords;
};

class DeviceMesh {
public:
    DeviceMesh() = default;
    ~DeviceMesh() { release_gpu(); }

    // 从 CPU 完整网格初始化（不考虑 AMR 先）
    void initialize_from(const ComputingMesh& cpu_mesh);

    // 上传到 GPU
    void upload_to_gpu();

    // 释放 GPU 内存
    void release_gpu();

    // 内核调用用到的接口
    HostDevice GPUTetrahedron* device_cells() { return d_cells_; }
    HostDevice const GPUTetrahedron* device_cells() const { return d_cells_; }
    std::vector<GPUTetrahedron> host_cells() const ;
    HostDevice GPUTriangleFace* device_faces() { return d_faces_; }
    HostDevice const GPUTriangleFace* device_faces() const { return d_faces_; }
    std::vector<GPUTriangleFace> host_faces() const ;
    HostDevice vector3f* device_points() { return d_points_; }
    HostDevice const vector3f* device_points() const { return d_points_; }
    std::vector<vector3f> host_points() const ;
    HostDevice uInt num_cells() const { return num_cells_; }
    HostDevice uInt num_faces() const { return num_faces_; }
    HostDevice uInt num_points() const { return num_points_; }
    Scalar get_memory_usage() const ;

private:
    std::vector<GPUTetrahedron> h_cells_;
    std::vector<GPUTriangleFace> h_faces_;
    std::vector<vector3f> h_points_;

    GPUTetrahedron* d_cells_ = nullptr;
    GPUTriangleFace* d_faces_ = nullptr;
    vector3f* d_points_;

    uInt num_cells_ = 0;
    uInt num_faces_ = 0;
    uInt num_points_ = 0;
};
