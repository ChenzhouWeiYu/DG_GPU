#include "mesh/cgal_mesh.h"
#include "mesh/mesh_generator.h"


CGALMesh::CGALMesh(double height) 
    : m_aspect_ratio(0.2), m_size_bound(0.05), m_height(height) {
}
CGALMesh::CGALMesh(double size_bound, double height) 
    : m_aspect_ratio(0.2), m_size_bound(size_bound), m_height(height) {
}
CGALMesh::CGALMesh(double aspect_ratio, double size_bound, double height) 
    : m_aspect_ratio(aspect_ratio), m_size_bound(size_bound), m_height(height) {
}
void CGALMesh::generate_2d_mesh(const std::vector<std::array<double, 2>>& polygon_points) {
    MeshGenerator generator;
    generator.set_refinement_criteria(m_aspect_ratio, m_size_bound);
    generator.generate_convex_polygon(polygon_points);
    generator.extrude_to_3d(m_height);
    generator.tetrahedralize();
    m_mesh_data = generator.get_mesh_data();
}
void CGALMesh::generate_2d_mesh(const std::vector<std::array<double, 2>>& polygon_points, const std::vector<std::array<double, 2>>& internal_points) {
    MeshGenerator generator;
    generator.set_refinement_criteria(m_aspect_ratio, m_size_bound);
    generator.generate_convex_polygon(polygon_points, internal_points);
    generator.extrude_to_3d(m_height);
    generator.tetrahedralize();
    m_mesh_data = generator.get_mesh_data();
}

void CGALMesh::generate_2d_mesh(
                    const std::vector<std::array<double, 2>>& polygon_points, 
                    const std::vector<std::array<double, 2>>& internal_points,
                    const std::array<Scalar,2> exclude_corner
    ) {
    return generate_2d_mesh(polygon_points, internal_points, [&](double x, double y, double z) {
        return exclude_corner[0] < x && y < exclude_corner[1];
    });
}

// 支持 Hole 对象
void CGALMesh::generate_2d_mesh(
    const std::vector<std::array<double, 2>>& polygon_points,
    const std::vector<std::array<double, 2>>& internal_points,
    const Hole& hole) 
{
    MeshGenerator generator;
    generator.set_refinement_criteria(m_aspect_ratio, m_size_bound);
    generator.generate_convex_polygon(polygon_points, internal_points);
    generator.extrude_to_3d(m_height);
    auto is_hole = [&hole](double x, double y, double z) {
        return hole.contains(x, y, z);
    };
    generator.tetrahedralize(is_hole);
    m_mesh_data = generator.get_mesh_data();
}

void CGALMesh::generate_2d_mesh(
    const std::vector<std::array<double, 2>>& polygon_points,
    const std::vector<std::array<double, 2>>& internal_points,
    const std::function<bool(double, double, double)>& is_hole)
{
    MeshGenerator generator;
    generator.set_refinement_criteria(m_aspect_ratio, m_size_bound);
    generator.generate_convex_polygon(polygon_points, internal_points);
    generator.extrude_to_3d(m_height);
    generator.tetrahedralize(is_hole);
    m_mesh_data = generator.get_mesh_data();
}


MeshData CGALMesh::get_mesh_data() const {
    return m_mesh_data;
}
DGMesh CGALMesh::get_dg_mesh() {
    if (!is_build_dg_mesh) {
        build_dg_mesh();
    }
    return m_dg_mesh;
}

void CGALMesh::build_dg_mesh() {
    m_dg_mesh = build_dg_mesh_impl(m_mesh_data);
    is_build_dg_mesh = true;
}

void CGALMesh::export_dgmesh_to_vtk(const std::string& filename) {
    export_dgmesh_to_vtk_impl(m_dg_mesh, filename);
}

void CGALMesh::set_aspect_size_height(double aspect_ratio, double size_bound, double height) {
    m_aspect_ratio = aspect_ratio;
    m_size_bound = size_bound;
    m_height = height;
}
void CGALMesh::set_aspect(double aspect_ratio) {
    m_aspect_ratio = aspect_ratio;
}
void CGALMesh::set_size(double size_bound) {
    m_size_bound = size_bound;
}
void CGALMesh::set_height(double height) {
    m_height = height;
}

