#include "base/type.h"
#include "mesh/mesh_generator.h"
class CylinderGenerator : public IDomainGenerator {
public:
    CylinderGenerator(vector2f lb = {-4,-4}, vector2f ub={6,4}) 
        : lb(lb), ub(ub) {}

    CylinderGenerator(vector3f cylinder) 
        : radius(cylinder[0]), 
        center_x(cylinder[1]), center_y(cylinder[2]) { }

    CylinderGenerator(vector2f lb, vector2f ub, vector3f cylinder) 
        : lb(lb), ub(ub), radius(cylinder[0]), 
        center_x(cylinder[1]), center_y(cylinder[2]) { }

    void generate(CDT& cdt) override {
        // 实现圆柱生成逻辑
        cdt.clear();
        set_params();

        // 1. 生成外边界点(矩形)
        std::vector<CDT::Vertex_handle> outer_vertices;
        std::vector<std::array<Scalar, 2>> outer_points = {
            {lb[0], lb[1]}, {ub[0], lb[1]}, {ub[0], ub[1]}, {lb[0], ub[1]}
        };
        
        for (size_t i = 0; i < outer_points.size(); ++i) {
            auto vh = cdt.insert(CDT::Point(outer_points[i][0], outer_points[i][1]));
            outer_vertices.push_back(vh);
        }
        
        // 2. 生成圆柱周围的层状点
        for (uInt i = 0; i < 4*points_per_layer; ++i) {
            Scalar angle = 2.0 * M_PI * i / (4*points_per_layer);
            Scalar x = center_x + radius * cos(angle);
            Scalar y = center_y + radius * sin(angle);
            auto vh = cdt.insert(CDT::Point(x, y));
        }
        for (uInt layer = 0; layer < num_layers; ++layer) {
            Scalar current_radius = radius + initial_offset * (pow(growth_rate, layer+1)-1)/(growth_rate-1);
            if(layer == 1)
            for (uInt i = 0; i < 3*points_per_layer; ++i) {
                Scalar angle = 2.0 * M_PI * (i + 3*0.5*(layer%2 + 1)) / (3*points_per_layer);
                Scalar x = center_x + current_radius * cos(angle);
                Scalar y = center_y + current_radius * sin(angle);
                cdt.insert(CDT::Point(x, y));
            }
            else
            for (uInt i = 0; i < points_per_layer; ++i) {
                Scalar angle = 2.0 * M_PI * (i + 0.5*(layer%2 + 1)) / (1*points_per_layer);
                Scalar x = center_x + current_radius * cos(angle);
                Scalar y = center_y + current_radius * sin(angle);
                cdt.insert(CDT::Point(x, y));
            }
        }
        
        // 3. 添加边界约束
        // 外边界约束
        for (size_t i = 0; i < outer_vertices.size(); ++i) {
            size_t j = (i + 1) % outer_vertices.size();
            cdt.insert_constraint(outer_vertices[i], outer_vertices[j]);
        }

    }
    bool is_inside_hole(Scalar x, Scalar y) const override {
        Scalar dx = x - center_x;
        Scalar dy = y - center_y;
        return dx*dx + dy*dy < radius*radius - 1e-16;
    }
private:
    void set_params(){
        Scalar k1 = 1.0;
        Scalar k2 = 1.0;
        Scalar rate_rhs = std::exp(k1)*k2/(2*M_PI) - 1;
        Scalar rate_lhs = 1.414*(k2*(std::exp(k1)-1)/k1);
        // num_layers = uInt(points_per_layer * rate_rhs / rate_lhs);
        num_layers = uInt((6*radius -  radius) / (radius * k2/points_per_layer * (std::exp(k1)-1)/k1));
        print(num_layers);
        growth_rate = 1.0 + k1 / num_layers;
        initial_offset = k2/points_per_layer*radius;
        print(radius + initial_offset * (pow(growth_rate, num_layers)-1)/(growth_rate-1));
        // print(vector2f{k2 / points_per_layer * radius, initial_offset});
        // print(vector2f{1 + k1/num_layers, growth_rate});
        print(radius * (1 + k2/points_per_layer * (std::exp(k1)-1) * num_layers/k1));
    }
    vector2f lb{-15,-15}, ub{35,15};
    Scalar radius = 0.5, center_x = 0.0, center_y = 0.0;
    uInt num_layers = 20, points_per_layer = 36;
    Scalar initial_offset, growth_rate;

    // 如果 growth 满足和 layers 的关系：growth = 1 + k1/layers，growth^layers = exp(k1)，
    // 如果 init = k2 / n_points * radius，那么 radius + init * other = radius*(1 + k2/n_points + other)
    // 最外层的半径：radius + init*(growth^layers-1)/(growth-1) = radius * (1 + k2/n_points * (exp(k1)-1) * layers/k1)
    // 最外层的弧长：radius * (1 + k2/n_points * (exp(k1)-1) * layers/k1) * (2*pi / n_points)
    // 最外层的宽度：init * growth^layers = exp(k1) * k2 * radius/n_points
    // 假如相等的话：消去公共的 radius/n_points，得 (1 + k2/n_points * (exp(k1)-1) * layers/k1) * 2 * pi = exp(k1) * k2
    // (k2*(exp(k1)-1)/k1) * layers/n_points = exp(k1)*k2/(2*pi) - 1
    // k1, k2 至少要有一个足够大，才能 exp(k1)*k2/(2*pi) - 1 > 0，至少 exp(k1)*k2 > 6.28
    // 增大 k1 导致 growth = 1 + k1/layers 增大
    // 增大 k2 导致 init = k2 / n_points * radius 增大
};