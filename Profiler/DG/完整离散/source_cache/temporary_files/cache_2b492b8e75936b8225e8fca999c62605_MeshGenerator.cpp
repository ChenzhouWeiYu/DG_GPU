#include "mesh/mesh_generator.h"
#include <CGAL/lloyd_optimize_mesh_2.h>
#include <CGAL/Delaunay_mesh_size_criteria_2.h>
#include <cmath>
#include <map>
// #include <CGAL/make_mesh_2.h>
// #include <CGAL/refine_mesh_2.h>



MeshGenerator::MeshGenerator() 
    : refinement_criteria(0.125, 0.05) {}

void MeshGenerator::set_refinement_criteria(double aspect_ratio, double size_bound) {
    refinement_criteria = Criteria(aspect_ratio, size_bound);
}

void MeshGenerator::set_size_field(const std::function<double(double, double)>& size_func) {
    size_field_func = size_func;
}

void MeshGenerator::generate_convex_polygon(const std::vector<std::array<double, 2>>& polygon_points) {
    cdt.clear();
    std::vector<CDT::Vertex_handle> vertices;
    
    for (size_t i = 0; i < polygon_points.size(); ++i) {
        auto vh = cdt.insert(CDT::Point(polygon_points[i][0], polygon_points[i][1]));
        vh->info() = i;
        vertices.push_back(vh);
    }
    
    for (size_t i = 0; i < vertices.size(); ++i) {
        size_t j = (i + 1) % vertices.size();
        cdt.insert_constraint(vertices[i], vertices[j]);
    }
    
    refine_mesh();
}

void MeshGenerator::generate_convex_polygon(
    const std::vector<std::array<double, 2>>& polygon_points,
    const std::vector<std::array<double, 2>>& internal_points) 
{
    cdt.clear();
    std::vector<CDT::Vertex_handle> vertices;
    
    for (size_t i = 0; i < polygon_points.size(); ++i) {
        auto vh = cdt.insert(CDT::Point(polygon_points[i][0], polygon_points[i][1]));
        vh->info() = i;
        vertices.push_back(vh);
    }

    for (size_t i = 0; i < internal_points.size(); ++i) {
        auto vh = cdt.insert(CDT::Point(internal_points[i][0], internal_points[i][1]));
    }
    
    for (size_t i = 0; i < vertices.size(); ++i) {
        size_t j = (i + 1) % vertices.size();
        cdt.insert_constraint(vertices[i], vertices[j]);
    }
    
    refine_mesh();
}

void MeshGenerator::generate_l_shape() {
    std::vector<std::array<double, 2>> points = {
        {0, 0}, {2, 0}, {2, 1}, {1, 1}, {1, 2}, {0, 2}
    };
    generate_convex_polygon(points);
}

void MeshGenerator::generate_cylinder_flow(double radius, double center_x, double center_y) {
    cdt.clear();
    
    // 外边界
    std::vector<CDT::Vertex_handle> outer_vertices;
    std::vector<std::array<double, 2>> outer_points = {
        {-5, -5}, {5, -5}, {5, 5}, {-5, 5}
    };
    
    for (size_t i = 0; i < outer_points.size(); ++i) {
        auto vh = cdt.insert(CDT::Point(outer_points[i][0], outer_points[i][1]));
        vh->info() = i;
        outer_vertices.push_back(vh);
    }
    
    for (size_t i = 0; i < outer_vertices.size(); ++i) {
        size_t j = (i + 1) % outer_vertices.size();
        cdt.insert_constraint(outer_vertices[i], outer_vertices[j]);
    }
    
    // 内边界
    const size_t circle_segments = 36;
    std::vector<CDT::Vertex_handle> inner_vertices;
    
    for (size_t i = 0; i < circle_segments; ++i) {
        double angle = 2.0 * M_PI * i / circle_segments;
        double x = center_x + radius * cos(angle);
        double y = center_y + radius * sin(angle);
        auto vh = cdt.insert(CDT::Point(x, y));
        vh->info() = i + 1000;
        inner_vertices.push_back(vh);
    }
    
    for (size_t i = 0; i < inner_vertices.size(); ++i) {
        size_t j = (i + 1) % inner_vertices.size();
        cdt.insert_constraint(inner_vertices[i], inner_vertices[j]);
    }
    
    refine_mesh();
}

void MeshGenerator::refine_mesh() {
    if (size_field_func) {
        // 使用CGAL的网格生成器处理尺寸场
        // 使用自定义尺寸标准
        // Custom_size_criteria custom_criteria(size_field_func);
        CGAL::refine_Delaunay_mesh_2(cdt, refinement_criteria);
        // CGAL::refine_mesh_2(cdt, refinement_criteria);
        
        // 如果需要更精确的尺寸控制，可以在这里添加额外逻辑
    } else {
        CGAL::refine_Delaunay_mesh_2(cdt, refinement_criteria);
        // CGAL::refine_mesh_2(cdt, refinement_criteria);
    }
}


void MeshGenerator::extrude_to_3d(double height) {
    mesh_data.vertices.clear();


    // 收集所有顶点
    for (auto vit = cdt.finite_vertices_begin(); vit != cdt.finite_vertices_end(); ++vit) {
        double x = vit->point().x();
        double y = vit->point().y();
        if (generator && generator->is_inside_hole(x, y)) {
            continue;  // 跳过洞内点
        }
        mesh_data.vertices.push_back({x, y, 0.0});
        mesh_data.vertices.push_back({x, y, height});
    }

}

void MeshGenerator::tetrahedralize() {
    mesh_data.faces.clear();
    mesh_data.tetrahedra.clear();

    DT3 dt3;

    // Step 1: 插入所有三维点到 Delaunay_triangulation_3 中
    for (const auto& pt : mesh_data.vertices) {
        dt3.insert(DT3::Point(pt[0], pt[1], pt[2]));
    }

    // Step 2: 提取四面体单元（排除圆柱内部）
    std::map<Vertex_handle, size_t> vh_to_index;
    size_t v_index = 0;

    for (auto vit = dt3.finite_vertices_begin(); vit != dt3.finite_vertices_end(); ++vit) {
        vh_to_index[vit] = v_index++;
    }

    std::map<Cell_handle, size_t> ch_to_index;
    size_t c_index = 0;

    for (auto cit = dt3.finite_cells_begin(); cit != dt3.finite_cells_end(); ++cit) {
        std::array<size_t, 4> tet;
        for (int i = 0; i < 4; ++i) {
            tet[i] = vh_to_index[cit->vertex(i)];
        }

        // 计算重心
        double x = (mesh_data.vertices[tet[0]][0] + mesh_data.vertices[tet[1]][0] +
                    mesh_data.vertices[tet[2]][0] + mesh_data.vertices[tet[3]][0]) / 4.0;
        double y = (mesh_data.vertices[tet[0]][1] + mesh_data.vertices[tet[1]][1] +
                    mesh_data.vertices[tet[2]][1] + mesh_data.vertices[tet[3]][1]) / 4.0;

        if (generator && generator->is_inside_hole(x, y)) {
            continue;  // 排除洞内单元
        }

        mesh_data.tetrahedra.push_back(tet);
        ch_to_index[cit] = c_index++;
    }

    // Step 3: 提取与有效单元相关的面
    std::vector<DT3::Facet> effective_facets;
    std::map<DT3::Facet, size_t> facet_to_index;
    size_t f_index = 0;

    for (auto cit = dt3.finite_cells_begin(); cit != dt3.finite_cells_end(); ++cit) {
        if (ch_to_index.find(cit) == ch_to_index.end()) {
            continue;  // 该单元已被删除
        }

        for (int i = 0; i < 4; ++i) {
            DT3::Facet current_facet(cit, i);
            Cell_handle neighbor = cit->neighbor(i);

            // 只处理单元对
            if (dt3.is_infinite(neighbor)) {
                // 边界面
                effective_facets.push_back(current_facet);
                facet_to_index[current_facet] = f_index++;
            } else {
                // 内部面：只保留一次（避免重复添加）
                if (facet_to_index.find(current_facet) == facet_to_index.end()) {
                    effective_facets.push_back(current_facet);
                    facet_to_index[current_facet] = f_index++;
                }
            }
        }
    }

    // Step 4: 填充 faces 数据
    mesh_data.faces.resize(f_index);  // 实际有效的面数量

    for (size_t f_idx = 0; f_idx < effective_facets.size(); ++f_idx) {
        const auto& facet = effective_facets[f_idx];
        Cell_handle cell = facet.first;
        int local_facet_id = facet.second;

        std::array<size_t, 3> face_nodes;
        int node_count = 0;
        for (int i = 0; i < 4; ++i) {
            if (i == local_facet_id) continue;
            Vertex_handle v = cell->vertex(i);
            face_nodes[node_count++] = vh_to_index[v];
        }

        mesh_data.faces[f_idx] = face_nodes;
    }

    // Step 5: 构建邻接关系
    build_adjacency(dt3, ch_to_index, facet_to_index, effective_facets);
}

void MeshGenerator::build_adjacency(
    const DT3& dt3,
    const std::map<Cell_handle, size_t>& ch_to_index,
    const std::map<DT3::Facet, size_t>& facet_to_index,
    const std::vector<DT3::Facet>& effective_facets) 
{
    const size_t num_faces = effective_facets.size();
    const size_t num_cells = ch_to_index.size();

    mesh_data.face_adjacency.resize(num_faces);
    mesh_data.is_boundary_face.resize(num_faces, false);

    // Step 1: 填充 face_adjacency 和 is_boundary_face
    for (size_t f_idx = 0; f_idx < num_faces; ++f_idx) {
        const auto& facet = effective_facets[f_idx];
        Cell_handle cell = facet.first;
        int local_facet_id = facet.second;

        auto neighbor = cell->neighbor(local_facet_id);
        bool is_infinite = dt3.is_infinite(neighbor);

        size_t c0 = ch_to_index.at(cell);
        size_t c1 = size_t(-1);

        if (!is_infinite) {
            auto it = ch_to_index.find(neighbor);
            if (it != ch_to_index.end()) {
                c1 = it->second;
            } else {
                is_infinite = true;
            }
        }

        mesh_data.face_adjacency[f_idx][0] = c0;
        mesh_data.face_adjacency[f_idx][1] = c1;
        mesh_data.is_boundary_face[f_idx] = is_infinite;
    }

    // Step 2: 构建 cell_adjacency（每个单元的四个邻接面索引）
    mesh_data.cell_adjacency.resize(num_cells);
    for (size_t cid = 0; cid < num_cells; ++cid) {
        mesh_data.cell_adjacency[cid].fill(size_t(-1));
    }

    // Step 3: 遍历所有有效单元的每个面
    for (auto cit = dt3.finite_cells_begin(); cit != dt3.finite_cells_end(); ++cit) {
        if (ch_to_index.find(cit) == ch_to_index.end()) {
            continue;  // 该单元已被删除
        }

        size_t c_idx = ch_to_index.at(cit);

        for (int i = 0; i < 4; ++i) {
            Cell_handle neighbor = cit->neighbor(i);
            if (dt3.is_infinite(neighbor)) {
                mesh_data.cell_adjacency[c_idx][i] = size_t(-1);
                continue;
            }

            DT3::Facet current_facet(cit, i);
            auto f_it = facet_to_index.find(current_facet);
            if (f_it == facet_to_index.end()) {
                mesh_data.cell_adjacency[c_idx][i] = size_t(-1);
                continue;
            }

            size_t f_idx = f_it->second;
            mesh_data.cell_adjacency[c_idx][i] = f_idx;
        }
    }
}

MeshData MeshGenerator::get_mesh_data() const {
    return mesh_data;
}



void MeshGenerator::export_to_file(const std::string& filename) const {
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return;
    }
    
    // 输出顶点
    out << "Vertices:" << std::endl;
    for (const auto& v : mesh_data.vertices) {
        out << v[0] << " " << v[1] << " " << v[2] << std::endl;
    }
    
    // 输出四面体
    out << "Tetrahedra:" << std::endl;
    for (const auto& t : mesh_data.tetrahedra) {
        out << t[0] << " " << t[1] << " " << t[2] << " " << t[3] << std::endl;
    }
    
    // 输出面
    out << "Faces:" << std::endl;
    for (const auto& f : mesh_data.faces) {
        out << f[0] << " " << f[1] << " " << f[2] << std::endl;
    }
    
    out.close();
}

void MeshGenerator::export_to_vtk(const std::string& filename) const {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return;
    }

    // VTK header
    out << "# vtk DataFile Version 3.0\n";
    out << "DG Mesh Data Export\n";
    out << "ASCII\n";
    out << "DATASET UNSTRUCTURED_GRID\n";

    // 写入顶点
    out << "POINTS " << mesh_data.vertices.size() << " double\n";
    for (const auto& pt : mesh_data.vertices) {
        out << pt[0] << " " << pt[1] << " " << pt[2] << "\n";
    }
    out << "\n";

    // 写入单元（四面体）
    size_t total_cell_nodes = 0;
    for (const auto& tet : mesh_data.tetrahedra) {
        total_cell_nodes += 5;  // 每个单元前缀是节点数 + 4 个索引
    }

    out << "CELLS " << mesh_data.tetrahedra.size() << " " << total_cell_nodes << "\n";
    for (const auto& tet : mesh_data.tetrahedra) {
        out << "4 "
            << tet[0] << " "
            << tet[1] << " "
            << tet[2] << " "
            << tet[3] << "\n";
    }
    out << "\n";

    // 写入单元类型（四面体为 10）
    out << "CELL_TYPES " << mesh_data.tetrahedra.size() << "\n";
    for (size_t i = 0; i < mesh_data.tetrahedra.size(); ++i) {
        out << "10\n";  // 四面体类型码为 10
    }
    out << "\n";

    // 输出 CELL_DATA（必须与单元数量一致）
    out << "CELL_DATA " << mesh_data.tetrahedra.size() << "\n";

    // 示例：输出单元 ID
    out << "SCALARS cell_ids int 1\nLOOKUP_TABLE default\n";
    for (size_t c_idx = 0; c_idx < mesh_data.tetrahedra.size(); ++c_idx) {
        out << c_idx << "\n";
    }
    out << "\n";

    // 示例：输出边界面计数（每个单元连接的边界面数量）
    std::vector<int> face_count(mesh_data.tetrahedra.size(), 0);

    for (const auto& adj : mesh_data.cell_adjacency) {
        for (int i = 0; i < 4; ++i) {
            if (adj[i] == size_t(-1)) {
                face_count[i] += 1;
            }
        }
    }

    out << "SCALARS boundary_face_count int 1\nLOOKUP_TABLE default\n";
    for (int count : face_count) {
        out << count << "\n";
    }
    out << "\n";

    // 输出 POINT_DATA（可选）
    out << "POINT_DATA " << mesh_data.vertices.size() << "\n";
    out << "SCALARS point_ids int 1\nLOOKUP_TABLE default\n";
    for (size_t v_idx = 0; v_idx < mesh_data.vertices.size(); ++v_idx) {
        out << v_idx << "\n";
    }
    out << "\n";

    out.close();
}

void MeshGenerator::check_adjacency() const {
    size_t missing_neighbor = 0;
    for (size_t f_idx = 0; f_idx < mesh_data.face_adjacency.size(); ++f_idx) {
        size_t c0 = mesh_data.face_adjacency[f_idx][0];
        size_t c1 = mesh_data.face_adjacency[f_idx][1];

        if (c0 == size_t(-1)) {
            std::cerr << "警告：face[" << f_idx << "] 的主单元不存在" << std::endl;
            continue;
        }

        if (c1 == size_t(-1)) {
            // 正常边界面
            continue;
        }

        // 检查 c0 和 c1 是否互为邻居
        bool found = false;
        for (int i = 0; i < 4; ++i) {
            if (mesh_data.cell_adjacency[c0][i] == f_idx) {
                found = true;
                break;
            }
        }

        if (!found) {
            std::cerr << "错误：单元[" << c0 << "] 没有记录 face[" << f_idx << "]" << std::endl;
            missing_neighbor++;
        }
    }

    if (missing_neighbor == 0) {
        std::cout << "邻接关系完整无误" << std::endl;
    } else {
        std::cout << "发现 " << missing_neighbor << " 个面未正确注册到单元邻接表中" << std::endl;
    }
}





struct FaceKey {
    std::array<size_t, 3> v;

    bool operator==(const FaceKey& other) const {
        return v[0] == other.v[0] && v[1] == other.v[1] && v[2] == other.v[2];
    }

    struct Hash {
        size_t operator()(const FaceKey& key) const {
            size_t h = 0;
            // A simple hash that maintains uniqueness for sorted triples
            h ^= std::hash<size_t>()(key.v[0]) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<size_t>()(key.v[1]) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<size_t>()(key.v[2]) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };
};






DGMesh build_dg_mesh_impl(const MeshData& input) {
    DGMesh output;
    output.points = input.vertices;
    output.cells = input.tetrahedra;

    // Step 1: Build all faces and count cell neighbors
    std::unordered_map<FaceKey, std::vector<size_t>, FaceKey::Hash> face_to_cells;

    // For each cell
    for (size_t cell_id = 0; cell_id < input.tetrahedra.size(); ++cell_id) {
        const auto& tet = input.tetrahedra[cell_id];
        
        // Generate all 4 faces of the tetrahedron
        const std::array<std::array<size_t, 3>, 4> faces = {{
            {tet[0], tet[1], tet[2]},  // face 0 (opposite vertex 3)
            {tet[0], tet[1], tet[3]},  // face 1 (opposite vertex 2)
            {tet[0], tet[2], tet[3]},  // face 2 (opposite vertex 1)
            {tet[1], tet[2], tet[3]}   // face 3 (opposite vertex 0)
        }};

        // For each face of this cell
        for (int local_face = 0; local_face < 4; ++local_face) {
            // Create sorted face key
            FaceKey key;
            key.v = faces[local_face];
            std::sort(key.v.begin(), key.v.end());

            // Record that this cell touches this face
            face_to_cells[key].push_back(cell_id);
        }
    }

    // Step 2: Create unique faces and face-cell adjacency
    std::unordered_map<FaceKey, size_t, FaceKey::Hash> face_to_index;
    output.faces.reserve(face_to_cells.size());
    output.face_cells.reserve(face_to_cells.size());
    output.is_boundary_face.reserve(face_to_cells.size());

    for (const auto& [key, cells] : face_to_cells) {
        // Add to faces list (in sorted order)
        size_t face_id = output.faces.size();
        output.faces.push_back(key.v);
        face_to_index[key] = face_id;

        // Set face adjacency
        if (cells.size() == 1) {
            // Boundary face
            output.face_cells.push_back({cells[0], static_cast<size_t>(-1)});
            output.is_boundary_face.push_back(true);
        } else if (cells.size() == 2) {
            // Internal face
            output.face_cells.push_back({cells[0], cells[1]});
            output.is_boundary_face.push_back(false);
        } else {
            throw std::runtime_error("Invalid mesh: face shared by more than 2 cells");
        }
    }

    // Step 3: Build cell-face and cell-cell adjacency
    output.cell_faces.resize(input.tetrahedra.size());
    output.cell_cells.resize(input.tetrahedra.size());

    for (size_t cell_id = 0; cell_id < input.tetrahedra.size(); ++cell_id) {
        const auto& tet = input.tetrahedra[cell_id];
        auto& cell_faces = output.cell_faces[cell_id];
        auto& cell_cells = output.cell_cells[cell_id];

        // Initialize with invalid indices
        cell_faces.fill(static_cast<size_t>(-1));
        cell_cells.fill(static_cast<size_t>(-1));

        // Generate all 4 faces again
        const std::array<std::array<size_t, 3>, 4> faces = {{
            {tet[0], tet[1], tet[2]},  // face 0
            {tet[0], tet[1], tet[3]},  // face 1
            {tet[0], tet[2], tet[3]},  // face 2
            {tet[1], tet[2], tet[3]}   // face 3
        }};

        // For each face of this cell
        for (int local_face = 0; local_face < 4; ++local_face) {
            // Create sorted face key
            FaceKey key;
            key.v = faces[local_face];
            std::sort(key.v.begin(), key.v.end());

            // Find the global face index
            auto it = face_to_index.find(key);
            if (it == face_to_index.end()) {
                throw std::runtime_error("Face not found - should never happen");
            }
            size_t face_id = it->second;

            // Record in cell_faces
            cell_faces[local_face] = face_id;

            // Find neighboring cell through this face
            const auto& adj = output.face_cells[face_id];
            if (adj[0] == cell_id) {
                cell_cells[local_face] = adj[1];
            } else if (adj[1] == cell_id) {
                cell_cells[local_face] = adj[0];
            }
            // else remains -1 (for boundary faces)
        }
    }

    return output;
}


void export_dgmesh_to_vtk_impl(const DGMesh& mesh, const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return;
    }

    // VTK header
    out << "# vtk DataFile Version 3.0\n";
    out << "DG Mesh Data Export\n";
    out << "ASCII\n";
    out << "DATASET UNSTRUCTURED_GRID\n";

    // 写入顶点
    out << "POINTS " << mesh.points.size() << " double\n";
    for (const auto& pt : mesh.points) {
        out << pt[0] << " " << pt[1] << " " << pt[2] << "\n";
    }
    out << "\n";

    // 写入单元（四面体）
    size_t total_cell_nodes = 5 * mesh.cells.size();  // 每个单元: 4节点 + 长度前缀
    out << "CELLS " << mesh.cells.size() << " " << total_cell_nodes << "\n";
    for (const auto& tet : mesh.cells) {
        out << "4 " << tet[0] << " " << tet[1] << " " << tet[2] << " " << tet[3] << "\n";
    }
    out << "\n";

    // 写入单元类型（四面体为10）
    out << "CELL_TYPES " << mesh.cells.size() << "\n";
    for (size_t i = 0; i < mesh.cells.size(); ++i) {
        out << "10\n";
    }
    out << "\n";

    // CELL_DATA - 必须与单元数量一致
    out << "CELL_DATA " << mesh.cells.size() << "\n";

    // 1. 输出单元ID
    out << "SCALARS cell_id int 1\nLOOKUP_TABLE default\n";
    for (size_t i = 0; i < mesh.cells.size(); ++i) {
        out << i << "\n";
    }

    // 2. 输出每个单元的边界面对计数
    out << "SCALARS boundary_face_count int 1\nLOOKUP_TABLE default\n";
    for (const auto& cell_faces : mesh.cell_faces) {
        int count = 0;
        for (size_t f : cell_faces) {
            if (f != static_cast<size_t>(-1) && mesh.is_boundary_face[f]) {
                count++;
            }
        }
        out << count << "\n";
    }

    // 3. 输出邻接单元数（可视化孤立单元）
    out << "SCALARS neighbor_count int 1\nLOOKUP_TABLE default\n";
    for (const auto& cell_cells : mesh.cell_cells) {
        int count = 0;
        for (size_t c : cell_cells) {
            if (c != static_cast<size_t>(-1)) count++;
        }
        out << count << "\n";
    }
    out << "\n";

    // 输出面数据（作为POINT_DATA附加属性）
    out << "POINT_DATA " << mesh.points.size() << "\n";

    // 1. 输出顶点ID
    out << "SCALARS point_id int 1\nLOOKUP_TABLE default\n";
    for (size_t i = 0; i < mesh.points.size(); ++i) {
        out << i << "\n";
    }

    // 2. 输出顶点是否在边界上（通过关联的面判断）
    out << "SCALARS is_boundary_point int 1\nLOOKUP_TABLE default\n";
    std::vector<bool> is_boundary(mesh.points.size(), false);
    for (size_t f = 0; f < mesh.faces.size(); ++f) {
        if (mesh.is_boundary_face[f]) {
            for (size_t v : mesh.faces[f]) {
                is_boundary[v] = true;
            }
        }
    }
    for (bool b : is_boundary) {
        out << (b ? 1 : 0) << "\n";
    }
    out << "\n";

    // 可选：输出面数据（需要转换为CELL_DATA）
    if (!mesh.faces.empty()) {
        out << "CELL_DATA " << mesh.cells.size() << "\n";
        
        // 输出每个单元的面ID（4个分量）
        out << "SCALARS face_ids int 4\nLOOKUP_TABLE default\n";
        for (const auto& faces : mesh.cell_faces) {
            for (size_t f : faces) {
                out << (f != static_cast<size_t>(-1) ? f : -1) << " ";
            }
            out << "\n";
        }
    }

    out.close();
}