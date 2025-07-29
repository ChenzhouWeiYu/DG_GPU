#pragma once
#include "base/type.h"


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
