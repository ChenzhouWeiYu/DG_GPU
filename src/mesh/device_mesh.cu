#include "mesh/device_mesh.cuh"
#include <cuda_runtime.h>
#include <cstdlib> // for malloc/free

void DeviceMesh::initialize_from(const ComputingMesh& cpu_mesh) {
    num_cells_ = cpu_mesh.m_cells.size();
    num_faces_ = cpu_mesh.m_faces.size();
    num_points_ = cpu_mesh.m_points.size();
    
    // std::cout << "Number of Points: " << num_points_ << std::endl;
    // std::cout << "Number of Faces: " << num_faces_ << std::endl;
    // std::cout << "Number of Cells: " << num_cells_ << std::endl;

    // 分配 host 内存（用 malloc）
    h_cells_ = static_cast<GPUTetrahedron*>(malloc(num_cells_ * sizeof(GPUTetrahedron)));
    h_faces_ = static_cast<GPUTriangleFace*>(malloc(num_faces_ * sizeof(GPUTriangleFace)));
    h_points_ = static_cast<vector3f*>(malloc(num_points_ * sizeof(vector3f)));

    // 填充单元
    for (uInt i = 0; i < num_cells_; ++i) {
        const auto& src = cpu_mesh.m_cells[i];
        auto& dst = h_cells_[i];
        dst.nodes = src.m_nodes;
        dst.neighbor_cells = src.m_neighbors;
        dst.volume = src.m_volume;
        dst.m_h = src.m_h;
        dst.JacMat = src.m_JacMat;
        dst.invJac = src.m_invJac;
    }

    // 填充面
    for (uInt i = 0; i < num_faces_; ++i) {
        const auto& src = cpu_mesh.m_faces[i];
        auto& dst = h_faces_[i];
        dst.nodes = src.m_nodes;
        dst.normal = src.m_normal;
        dst.area = src.m_area;
        dst.neighbor_cells = src.m_neighbor_cells;
        dst.face_type = cpu_mesh.m_face_type[i];
        dst.natural_coords = src.m_natural_coords;
    }

    // 填充点
    for (uInt i = 0; i < num_points_; ++i) {
        h_points_[i] = cpu_mesh.m_points[i];
    }

    // 构建 face_indices 和 face_offsets
    std::vector<uInt> face_type_count(NumFaceTypes, 0);
    for (uInt i = 0; i < num_faces_; ++i) {
        FaceType ft = cpu_mesh.m_face_type[i];
        face_type_count[static_cast<uInt>(ft)]++;
    }

    // face_offsets[0] = 0
    // face_offsets[i+1] = face_offsets[i] + count[i]
    h_face_offsets_ = static_cast<uInt*>(malloc((NumFaceTypes + 1) * sizeof(uInt)));
    h_face_offsets_[0] = 0;
    for (uInt ft = 0; ft < NumFaceTypes; ++ft) {
        h_face_offsets_[ft + 1] = h_face_offsets_[ft] + face_type_count[ft];
    }

    // 临时计数器
    std::vector<uInt> counters(NumFaceTypes, 0);
    h_face_indices_ = static_cast<uInt*>(malloc(num_faces_ * sizeof(uInt)));
    for (uInt i = 0; i < num_faces_; ++i) {
        FaceType ft = cpu_mesh.m_face_type[i];
        uInt pos = h_face_offsets_[static_cast<uInt>(ft)] + counters[static_cast<uInt>(ft)];
        h_face_indices_[pos] = i;
        counters[static_cast<uInt>(ft)]++;
    }
}

void DeviceMesh::upload_to_gpu() {
    cudaMalloc(&d_cells_, num_cells_ * sizeof(GPUTetrahedron));
    cudaMemcpy(d_cells_, h_cells_, num_cells_ * sizeof(GPUTetrahedron), cudaMemcpyHostToDevice);

    cudaMalloc(&d_faces_, num_faces_ * sizeof(GPUTriangleFace));
    cudaMemcpy(d_faces_, h_faces_, num_faces_ * sizeof(GPUTriangleFace), cudaMemcpyHostToDevice);

    cudaMalloc(&d_points_, num_points_ * sizeof(vector3f));
    cudaMemcpy(d_points_, h_points_, num_points_ * sizeof(vector3f), cudaMemcpyHostToDevice);

    cudaMalloc(&d_face_indices_, num_faces_ * sizeof(uInt));
    cudaMemcpy(d_face_indices_, h_face_indices_, num_faces_ * sizeof(uInt), cudaMemcpyHostToDevice);

    cudaMalloc(&d_face_offsets_, (NumFaceTypes + 1) * sizeof(uInt));
    cudaMemcpy(d_face_offsets_, h_face_offsets_, (NumFaceTypes + 1) * sizeof(uInt), cudaMemcpyHostToDevice);
}

void DeviceMesh::release() {
    // 释放 host
    free(h_cells_); h_cells_ = nullptr;
    free(h_faces_); h_faces_ = nullptr;
    free(h_points_); h_points_ = nullptr;
    free(h_face_indices_); h_face_indices_ = nullptr;
    free(h_face_offsets_); h_face_offsets_ = nullptr;

    // 释放 device
    if (d_cells_) cudaFree(d_cells_);
    if (d_faces_) cudaFree(d_faces_);
    if (d_points_) cudaFree(d_points_);
    if (d_face_indices_) cudaFree(d_face_indices_);
    if (d_face_offsets_) cudaFree(d_face_offsets_);
    d_cells_ = nullptr;
    d_faces_ = nullptr;
    d_points_ = nullptr;
    d_face_indices_ = nullptr;
    d_face_offsets_ = nullptr;
}

Scalar DeviceMesh::get_memory_usage() const {
    size_t total = 0;
    total += num_cells_ * sizeof(GPUTetrahedron);
    total += num_faces_ * sizeof(GPUTriangleFace);
    total += num_points_ * sizeof(vector3f);
    total += num_faces_ * sizeof(uInt);
    total += (NumFaceTypes + 1) * sizeof(uInt);
    return static_cast<Scalar>(total) / (1024.0 * 1024.0);
}

// #include "mesh/device_mesh.cuh"
// #include <cuda_runtime.h>

// void DeviceMesh::initialize_from(const ComputingMesh& cpu_mesh) {
//     num_cells_ = cpu_mesh.m_cells.size();
//     num_faces_ = cpu_mesh.m_faces.size();
//     num_points_ = cpu_mesh.m_points.size();

//     h_cells_.resize(num_cells_);
//     for (uInt i = 0; i < num_cells_; ++i) {
//         const auto& src = cpu_mesh.m_cells[i];
//         auto& dst = h_cells_[i];
//         dst.nodes = src.m_nodes;
//         // dst.faces = src.m_faces;
//         dst.neighbor_cells = src.m_neighbors;
//         // dst.centroid = src.m_centroid;
//         dst.volume = src.m_volume;
//         dst.m_h = src.m_h;
//         dst.JacMat = src.m_JacMat;
//         dst.invJac = src.m_invJac;
//     }

//     h_faces_.resize(num_faces_);
//     for (uInt i = 0; i < num_faces_; ++i) {
//         const auto& src = cpu_mesh.m_faces[i];
//         auto& dst = h_faces_[i];
//         dst.nodes = src.m_nodes;
//         dst.normal = src.m_normal;
//         dst.area = src.m_area;
//         dst.neighbor_cells = src.m_neighbor_cells;
//         dst.boundaryType = cpu_mesh.m_boundaryTypes[i];
//         dst.natural_coords = src.m_natural_coords;
//     }

//     h_points_.resize(num_points_);
//     for (uInt i = 0; i < num_points_; ++i) {
//         h_points_[i] = cpu_mesh.m_points[i];
//     }
// }

// void DeviceMesh::upload_to_gpu() {
//     cudaMalloc(&d_cells_, num_cells_ * sizeof(GPUTetrahedron));
//     cudaMemcpy(d_cells_, h_cells_.data(), num_cells_ * sizeof(GPUTetrahedron), cudaMemcpyHostToDevice);

//     cudaMalloc(&d_faces_, num_faces_ * sizeof(GPUTriangleFace));
//     cudaMemcpy(d_faces_, h_faces_.data(), num_faces_ * sizeof(GPUTriangleFace), cudaMemcpyHostToDevice);

//     cudaMalloc(&d_points_, num_points_ * sizeof(vector3f));
//     cudaMemcpy(d_points_, h_points_.data(), num_points_ * sizeof(vector3f), cudaMemcpyHostToDevice);
// }

// void DeviceMesh::release_gpu() {
//     if (d_cells_) cudaFree(d_cells_);
//     if (d_faces_) cudaFree(d_faces_);
//     if (d_points_) cudaFree(d_points_);
//     d_cells_ = nullptr;
//     d_faces_ = nullptr;
//     d_points_ = nullptr;
// }

// // debug function
// std::vector<GPUTetrahedron> DeviceMesh::host_cells() const {
//     std::vector<GPUTetrahedron> tmp(num_cells_);
//     cudaMemcpy(tmp.data(), d_cells_, num_cells_ * sizeof(GPUTetrahedron), cudaMemcpyDeviceToHost);
//     return tmp;
// }

// // debug function
// std::vector<GPUTriangleFace> DeviceMesh::host_faces() const {
//     std::vector<GPUTriangleFace> tmp(num_faces_);
//     cudaMemcpy(tmp.data(), d_faces_, num_faces_ * sizeof(GPUTriangleFace), cudaMemcpyDeviceToHost);
//     return tmp;
// }


// Scalar DeviceMesh::get_memory_usage() const {
//     size_t total_bytes = 0;

//     // 单元（cells）
//     total_bytes += num_cells_ * sizeof(GPUTetrahedron);

//     // 面（faces）
//     total_bytes += num_faces_ * sizeof(GPUTriangleFace);

//     // 点（points）
//     total_bytes += num_points_ * sizeof(vector3f);

//     // 转换为 MB
//     return static_cast<Scalar>(total_bytes) / (1024.0 * 1024.0);  // MB
// }