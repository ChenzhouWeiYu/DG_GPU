#include "mesh/device_mesh.h"
#include <cuda_runtime.h>

void DeviceMesh::initialize_from(const ComputingMesh& cpu_mesh) {
    num_cells_ = cpu_mesh.m_cells.size();
    num_faces_ = cpu_mesh.m_faces.size();
    num_points_ = cpu_mesh.m_points.size();

    h_cells_.resize(num_cells_);
    for (uInt i = 0; i < num_cells_; ++i) {
        const auto& src = cpu_mesh.m_cells[i];
        auto& dst = h_cells_[i];
        dst.nodes = src.m_nodes;
        // dst.faces = src.m_faces;
        dst.neighbor_cells = src.m_neighbors;
        // dst.centroid = src.m_centroid;
        dst.volume = src.m_volume;
        dst.m_h = src.m_h;
        dst.JacMat = src.m_JacMat;
        dst.invJac = src.m_invJac;
    }

    h_faces_.resize(num_faces_);
    for (uInt i = 0; i < num_faces_; ++i) {
        const auto& src = cpu_mesh.m_faces[i];
        auto& dst = h_faces_[i];
        dst.nodes = src.m_nodes;
        dst.normal = src.m_normal;
        dst.area = src.m_area;
        dst.neighbor_cells = src.m_neighbor_cells;
        dst.boundaryType = cpu_mesh.m_boundaryTypes[i];
        dst.natural_coords = src.m_natural_coords;
    }

    h_points_.resize(num_points_);
    for (uInt i = 0; i < num_points_; ++i) {
        h_points_[i] = cpu_mesh.m_points[i];
    }
}

void DeviceMesh::upload_to_gpu() {
    cudaMalloc(&d_cells_, num_cells_ * sizeof(GPUTetrahedron));
    cudaMemcpy(d_cells_, h_cells_.data(), num_cells_ * sizeof(GPUTetrahedron), cudaMemcpyHostToDevice);

    cudaMalloc(&d_faces_, num_faces_ * sizeof(GPUTriangleFace));
    cudaMemcpy(d_faces_, h_faces_.data(), num_faces_ * sizeof(GPUTriangleFace), cudaMemcpyHostToDevice);

    cudaMalloc(&d_points_, num_points_ * sizeof(vector3f));
    cudaMemcpy(d_points_, h_points_.data(), num_points_ * sizeof(vector3f), cudaMemcpyHostToDevice);
}

void DeviceMesh::release_gpu() {
    if (d_cells_) cudaFree(d_cells_);
    if (d_faces_) cudaFree(d_faces_);
    if (d_points_) cudaFree(d_points_);
    d_cells_ = nullptr;
    d_faces_ = nullptr;
    d_points_ = nullptr;
}

std::vector<GPUTetrahedron> DeviceMesh::host_cells() const {
    std::vector<GPUTetrahedron> tmp(num_cells_);
    cudaMemcpy(tmp.data(), d_cells_, num_cells_ * sizeof(GPUTetrahedron), cudaMemcpyDeviceToHost);
    return tmp;
}

std::vector<GPUTriangleFace> DeviceMesh::host_faces() const {
    std::vector<GPUTriangleFace> tmp(num_faces_);
    cudaMemcpy(tmp.data(), d_faces_, num_faces_ * sizeof(GPUTriangleFace), cudaMemcpyDeviceToHost);
    return tmp;
}


Scalar DeviceMesh::get_memory_usage() const {
    size_t total_bytes = 0;

    // 单元（cells）
    total_bytes += num_cells_ * sizeof(GPUTetrahedron);

    // 面（faces）
    total_bytes += num_faces_ * sizeof(GPUTriangleFace);

    // 点（points）
    total_bytes += num_points_ * sizeof(vector3f);

    // 转换为 MB
    return static_cast<Scalar>(total_bytes) / (1024.0 * 1024.0);  // MB
}