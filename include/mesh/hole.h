// hole.h
#pragma once
#include "base/type.h"

// 抽象基类：Hole
class Hole {
public:
    virtual ~Hole() = default;
    virtual bool contains(double x, double y, double z = 0.0) const = 0;
    bool contains(const std::array<double,3>& p) const {
        return contains(p[0], p[1], p[2]);
    }
};

// 矩形孔
class RectangularHole : public Hole {
    double min_x, max_x, min_y, max_y;  // x > min_x && y < max_y → 挖掉
public:
    RectangularHole(std::array<double,2> min_xy, std::array<double,2> max_xy) : min_x(min_xy[0]), max_x(max_xy[0]), min_y(min_xy[1]), max_y(max_xy[1]) {}

    bool contains(double x, double y, double z) const override {
        return x > min_x && x < max_x && y > min_y && y < max_y;
    }
};

// 圆形孔
class CircularHole : public Hole {
    double center_x, center_y, radius_sq;
public:
    CircularHole(double cx, double cy, double r)
        : center_x(cx), center_y(cy), radius_sq(r * r) {}

    bool contains(double x, double y, double z) const override {
        double dx = x - center_x;
        double dy = y - center_y;
        return (dx*dx + dy*dy) < radius_sq;
    }
};

// 多边形孔
class PolygonalHole : public Hole {
    std::vector<std::array<double, 2>> points;
public:
    PolygonalHole(const std::vector<std::array<double, 2>>& pts) : points(pts) {}

    bool contains(double x, double y, double z) const override {
        // 射线法判断点是否在多边形内
        bool inside = false;
        size_t n = points.size();
        for (size_t i = 0, j = n - 1; i < n; j = i++) {
            if (((points[i][1] > y) != (points[j][1] > y)) &&
                (x < (points[j][0] - points[i][0]) * (y - points[i][1]) / (points[j][1] - points[i][1]) + points[i][0]))
                inside = !inside;
        }
        return inside;
    }
};