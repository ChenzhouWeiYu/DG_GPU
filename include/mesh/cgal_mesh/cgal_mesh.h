#pragma once
#include "base/type.h"
#include "mesh/cgal_mesh/hole.h"
#include <functional>


struct DGMesh {
    // 几何数据
    std::vector<std::array<double, 3>> points;   // 节点的坐标
    std::vector<std::array<size_t, 3>> faces;    // 组成三角形的节点索引
    std::vector<std::array<size_t, 4>> cells;    // 组成四面体的节点索引

    // 邻接关系
    std::vector<std::array<size_t, 4>> cell_faces;    // 组成四面体单元的三角形面索引（单元的邻接面）
    std::vector<std::array<size_t, 4>> cell_cells;    // 三角形面对应的四面体单元索引（单元的邻接单元）
    std::vector<std::array<size_t, 2>> face_cells;    // 三角形面两侧的四面体单元索引（面的邻接单元）
    std::vector<bool> is_boundary_face;
};

struct MeshData {
    // 基本元素
    std::vector<std::array<double, 3>> vertices;
    std::vector<std::array<size_t, 3>> faces;       // 所有三角面（边界+内部）
    std::vector<std::array<size_t, 4>> tetrahedra;  // 所有四面体
    
    // 邻接关系
    std::vector<std::array<size_t, 4>> cell_adjacency; // 每个四面体的4个邻接单元
    std::vector<std::array<size_t, 2>> face_adjacency; // 每个面的2个邻接单元（边界面为-1）
    
    // 辅助信息
    std::vector<bool> is_boundary_face;          // 标记是否为边界面
};


DGMesh build_dg_mesh(const MeshData& input);
void export_dgmesh_to_vtk(const DGMesh& mesh, const std::string& filename);
void export_dgmesh_to_vtk_impl(const DGMesh& mesh, const std::string& filename);


class CGALMesh {
public:
    CGALMesh(double height = 0.05);
    CGALMesh(double size_bound = 0.05, double height = 0.05);
    CGALMesh(double aspect_ratio = 0.2, double size_bound = 0.05, double height = 0.05);

    // 生成二维网格
    void generate_2d_mesh(const std::vector<std::array<double, 2>>& polygon_points);
    // 生成二维网格，包含内部点
    void generate_2d_mesh(const std::vector<std::array<double, 2>>& polygon_points, const std::vector<std::array<double, 2>>& internal_points);

    void generate_2d_mesh(const std::vector<std::array<double, 2>>& polygon_points, const std::vector<std::array<double, 2>>& internal_points, const std::array<Scalar,2>);
    void generate_2d_mesh(const std::vector<std::array<double, 2>>& polygon_points, const std::vector<std::array<double, 2>>& internal_points, const Hole& hole);
    void generate_2d_mesh(const std::vector<std::array<double, 2>>& polygon_points, const std::vector<std::array<double, 2>>& internal_points, const std::function<bool(double, double, double)>& is_hole);

    // 获取网格数据
    MeshData get_mesh_data() const;
    DGMesh get_dg_mesh();
    // 设置网格参数
    void set_aspect_size_height(double aspect_ratio = 0.2, double size_bound = 0.05, double height = 0.05);
    void set_aspect(double aspect_ratio = 0.2);
    void set_size(double size_bound = 0.05);
    void set_height(double height = 0.05);
    void build_dg_mesh();
    void export_dgmesh_to_vtk(const std::string& filename);

private:
    double m_aspect_ratio = 0.2; // 默认长宽比
    double m_size_bound = 0.05;   // 默认网格大小
    double m_height = 0.05;       // 默认高度

    MeshData m_mesh_data;         // 存储生成的网格数据
    DGMesh m_dg_mesh;         // 存储生成的网格数据
    bool is_build_dg_mesh = false; // 是否已经构建了DG网格
};


#include <array>
#include <vector>
#include <map>
#include <cstddef>
#include <algorithm>
#include <iostream>


class StructuredTetMeshGenerator {
public:
    enum class SubdivisionScheme {
        FiveTet,  // 5-tet，奇偶交替模板
        SixTet    // 6-tet，统一模板
    };

    using Point = std::array<double, 3>;
    using TetNode = std::array<size_t, 4>;
    using TriNode = std::array<size_t, 3>;
    using HexNode = std::array<size_t, 8>;

    template<size_t Nx, size_t Ny, size_t Nz>
    static DGMesh generate(
        const std::array<double, 3>& lb,
        const std::array<double, 3>& ub,
        const std::array<std::array<std::array<bool, Nx>, Ny>, Nz>& mask,
        SubdivisionScheme scheme = SubdivisionScheme::FiveTet
    ) {
        const double dx = (ub[0] - lb[0]) / static_cast<double>(Nx);
        const double dy = (ub[1] - lb[1]) / static_cast<double>(Ny);
        const double dz = (ub[2] - lb[2]) / static_cast<double>(Nz);

        // Step 1: 构建所有有效 Hex 的 (i,j,k) 列表
        std::vector<std::array<size_t, 3>> hex_list;
        for (size_t k = 0; k < Nz; ++k) {
            for (size_t j = 0; j < Ny; ++j) {
                for (size_t i = 0; i < Nx; ++i) {
                    if (mask[k][j][i]) {
                        hex_list.push_back({i, j, k});
                    }
                }
            }
        }

        // Step 2: 构建 points
        std::vector<Point> points;
        points.reserve((Nx + 1) * (Ny + 1) * (Nz + 1));
        for (size_t k = 0; k <= Nz; ++k) {
            double z = lb[2] + k * dz;
            for (size_t j = 0; j <= Ny; ++j) {
                double y = lb[1] + j * dy;
                for (size_t i = 0; i <= Nx; ++i) {
                    double x = lb[0] + i * dx;
                    points.push_back({x, y, z});
                }
            }
        }

        // Step 3: 构建 cells（四面体）
        std::vector<TetNode> cells;
        const size_t n_hex = hex_list.size();
        if (scheme == SubdivisionScheme::FiveTet) {
            cells.reserve(5 * n_hex);
        } else {
            cells.reserve(6 * n_hex);
        }

        // 辅助 lambda：获取六面体8个顶点索引
        auto get_hex_vertices = [&](size_t i, size_t j, size_t k) -> HexNode {
            size_t base = k * (Nx + 1) * (Ny + 1) + j * (Nx + 1) + i;
            size_t v000 = base;
            size_t v100 = base + 1;
            size_t v110 = base + 1 + (Nx + 1);
            size_t v010 = base + (Nx + 1);
            size_t v001 = base + (Nx + 1) * (Ny + 1);
            size_t v101 = base + (Nx + 1) * (Ny + 1) + 1;
            size_t v111 = base + (Nx + 1) * (Ny + 1) + 1 + (Nx + 1);
            size_t v011 = base + (Nx + 1) * (Ny + 1) + (Nx + 1);
            return {v000, v100, v110, v010, v001, v101, v111, v011};
        };

        // 辅助：添加5-tet或6-tet
        for (size_t idx = 0; idx < n_hex; ++idx) {
            auto [i, j, k] = hex_list[idx];
            HexNode hex = get_hex_vertices(i, j, k);
            size_t v000 = hex[0], v100 = hex[1], v110 = hex[2], v010 = hex[3];
            size_t v001 = hex[4], v101 = hex[5], v111 = hex[6], v011 = hex[7];

            if (scheme == SubdivisionScheme::FiveTet) {
                // 判断奇偶模板：(i + j + k) % 2
                bool use_template_A = ((i + j + k) % 2 == 0);
                if (use_template_A) {
                    // 中心四面体: (v000, v110, v101, v011)
                    cells.push_back({v111, v110, v101, v011});
                    cells.push_back({v000, v001, v101, v011});
                    cells.push_back({v000, v110, v010, v011});
                    cells.push_back({v000, v110, v101, v100});
                    cells.push_back({v000, v110, v101, v011});
                } else {
                    // 模板B：中心 (v111, v001, v010, v100)
                    cells.push_back({v000, v001, v010, v100});
                    cells.push_back({v111, v110, v010, v100});
                    cells.push_back({v111, v001, v101, v100});
                    cells.push_back({v111, v001, v010, v011});
                    cells.push_back({v111, v001, v010, v100});
                }
            } else { // SixTet
                // 暂时先不实现
            }
        }

        // Step 4: 构建 faces 和邻接关系
        std::vector<TriNode> faces;
        std::vector<std::array<size_t, 4>> cell_faces;
        std::vector<std::array<size_t, 2>> face_cells;
        std::vector<bool> is_boundary_face;

        // 四面体的四个面的局部索引（右手定则，外法向）
        const std::array<TriNode, 4> tet_faces_local = {{
            {1, 2, 3}, // 面0，对顶点0
            {0, 3, 2}, // 面1，对顶点1
            {0, 1, 3}, // 面2，对顶点2
            {0, 2, 1}  // 面3，对顶点3
        }};

        // 用于去重和查找 face -> cell 映射
        std::map<TriNode, size_t> tri_to_face_index;

        // 先收集所有 cell 的 face，并去重
        cell_faces.resize(cells.size());
        for (size_t c = 0; c < cells.size(); ++c) {
            const auto& tet = cells[c];
            for (size_t f = 0; f < 4; ++f) {
                TriNode tri_raw = {
                    tet[tet_faces_local[f][0]],
                    tet[tet_faces_local[f][1]],
                    tet[tet_faces_local[f][2]]
                };
                // 归一化三角形：排序使最小顶点在前，保证唯一表示
                TriNode tri_sorted = tri_raw;
                std::sort(tri_sorted.begin(), tri_sorted.end());
                auto it = tri_to_face_index.find(tri_sorted);
                if (it == tri_to_face_index.end()) {
                    size_t new_face_idx = faces.size();
                    faces.push_back(tri_sorted);
                    tri_to_face_index[tri_sorted] = new_face_idx;
                    face_cells.push_back({c, static_cast<size_t>(-1)}); // -1 表示尚未知另一侧
                    is_boundary_face.push_back(true);
                    cell_faces[c][f] = new_face_idx;
                } else {
                    size_t face_idx = it->second;
                    cell_faces[c][f] = face_idx;
                    // 更新邻接
                    if (face_cells[face_idx][0] == c) {
                        // 不应发生
                    } else if (face_cells[face_idx][1] == static_cast<size_t>(-1)) {
                        face_cells[face_idx][1] = c;
                        is_boundary_face[face_idx] = false;
                    } else {
                        // 三个单元共享一面？理论上不应发生
                    }
                }
            }
        }

        // 最终构建 cell_cells
        std::vector<std::array<size_t, 4>> cell_cells(cells.size());
        for (size_t c = 0; c < cells.size(); ++c) {
            for (size_t f = 0; f < 4; ++f) {
                size_t face_idx = cell_faces[c][f];
                size_t nb = (face_cells[face_idx][0] == c) ? face_cells[face_idx][1] : face_cells[face_idx][0];
                cell_cells[c][f] = nb;
            }
        }

        return DGMesh{
            .points = std::move(points),
            .faces = std::move(faces),
            .cells = std::move(cells),
            .cell_faces = std::move(cell_faces),
            .cell_cells = std::move(cell_cells),
            .face_cells = std::move(face_cells),
            .is_boundary_face = std::move(is_boundary_face)
        };
    }
};