#pragma once

#include "mesh/cgal_mesh.h"
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_mesher_2.h>
#include <CGAL/Delaunay_mesh_face_base_2.h>
#include <CGAL/Delaunay_mesh_size_criteria_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Triangulation_data_structure_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Triangulation_cell_base_3.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_vertex_base_with_info_2<size_t, K> Vb;
typedef CGAL::Delaunay_mesh_face_base_2<K> Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb> Tds;
typedef CGAL::Constrained_Delaunay_triangulation_2<K, Tds> CDT;
typedef CGAL::Delaunay_mesh_size_criteria_2<CDT> Criteria;


// 三维 Delaunay 网格类型定义
typedef CGAL::Delaunay_triangulation_3<K> DT3;
typedef DT3::Vertex_handle Vertex_handle;
typedef DT3::Cell_handle Cell_handle;



class IDomainGenerator {
public:
    virtual ~IDomainGenerator() = default;

    // 生成二维网格
    virtual void generate(CDT& cdt) = 0;

    // 判断点是否在洞内（如圆柱绕流中的圆柱）
    virtual bool is_inside_hole(double x, double y) const {
        return false; // 默认无洞
    }

    // 可选：获取边界点（用于调试）
    virtual std::vector<std::array<double, 2>> get_boundary_points() const {
        return {};
    }
};

class MeshGenerator {
public:
    MeshGenerator();
    
    void set_refinement_criteria(double aspect_ratio = 0.125, double size_bound = 0.05);
    void set_size_field(const std::function<double(double, double)>& size_func);
    void generate_convex_polygon(const std::vector<std::array<double, 2>>& polygon_points);
    void generate_convex_polygon(const std::vector<std::array<double, 2>>& polygon_points, const std::vector<std::array<double, 2>>& internal_points);
    void generate_l_shape();
    void generate_cylinder_flow(double radius, double center_x, double center_y);
    void refine_mesh();
    void extrude_to_3d(double height);
    void tetrahedralize();
    void tetrahedralize(std::array<double,2>);
    MeshData get_mesh_data() const;
    void export_to_file(const std::string& filename) const;

    // 添加区域生成方法
    void generate_domain(std::unique_ptr<IDomainGenerator> Igenerator) {
        generator = std::move(Igenerator);
        generator->generate(cdt);
        refine_mesh();
    }
    void check_adjacency() const;
    void export_to_vtk(const std::string& filename) const;

private:
    CDT cdt;
    std::unique_ptr<IDomainGenerator> generator;
    MeshData mesh_data;
    Criteria refinement_criteria;
    std::function<double(double, double)> size_field_func;
    void build_adjacency(
        const DT3& dt3,
        const std::map<Cell_handle, size_t>& ch_to_index,
        const std::map<DT3::Facet, size_t>& facet_to_index,
        const std::vector<DT3::Facet>& facet_list) ;
    void apply_size_field();
};


DGMesh build_dg_mesh_impl(const MeshData& input);
void export_dgmesh_to_vtk_impl(const DGMesh& mesh, const std::string& filename);