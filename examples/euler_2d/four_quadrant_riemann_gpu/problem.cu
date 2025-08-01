#include "base/type.h"
#include "mesh/computing_mesh.h"
#include "base/exact.h"

HostDevice Scalar get_gamma() {return 1.4;}

// 给定四个区块的状态
struct State {
    Scalar rho, u, v, p;
};

struct WaveSpeeds {
    Scalar mass, momentum, energy;
};

// Compute 1D RH wave speed in normal direction between two states
__device__  constexpr  inline WaveSpeeds compute_wave_speeds_x(const State& L, const State& R) {
    WaveSpeeds s{0.0,0.0,0.0};
    Scalar gamma = 1.4;
    s.mass = (R.rho * R.u - L.rho * L.u) / (R.rho - L.rho);
    s.momentum = ((R.rho * R.u * R.u + R.p) - (L.rho * L.u * L.u + L.p)) / (R.rho * R.u - L.rho * L.u);
    Scalar EL = (L.p / (gamma - 1.0) + 0.5 * L.rho * (L.u * L.u + L.v * L.v));// / L.rho;
    Scalar ER = (R.p / (gamma - 1.0) + 0.5 * R.rho * (R.u * R.u + R.v * R.v));// / R.rho;
    s.energy = (R.u * (ER + R.p) - L.u * (EL + L.p)) / (ER - EL);
    return s;
}

__device__  constexpr  inline WaveSpeeds compute_wave_speeds_y(const State& L, const State& R) {
    WaveSpeeds s{0.0,0.0,0.0};
    Scalar gamma = 1.4;
    s.mass = (R.rho * R.v - L.rho * L.v) / (R.rho - L.rho);
    s.momentum = ((R.rho * R.v * R.v + R.p) - (L.rho * L.v * L.v + L.p)) / (R.rho * R.v - L.rho * L.v);
    Scalar EL = (L.p / (gamma - 1.0) + 0.5 * L.rho * (L.u * L.u + L.v * L.v));// / L.rho;
    Scalar ER = (R.p / (gamma - 1.0) + 0.5 * R.rho * (R.u * R.u + R.v * R.v));// / R.rho;
    s.energy = (R.v * (ER + R.p) - L.v * (EL + L.p)) / (ER - EL);
    return s;
}

// 四个象限初始状态
__device__ __constant__  constexpr State s1 = {1.5, 0.0,   0.0,   1.5};
__device__ __constant__  constexpr State s2 = {0.5323, 1.206, 0.0,   0.3};
__device__ __constant__  constexpr State s3 = {0.138,  1.206, 1.206, 0.029};
__device__ __constant__  constexpr State s4 = {0.5323, 0.0,   1.206, 0.3};
constexpr vector2f init_x0y0 = {0.80, 0.80};

// __device__ __constant__  constexpr State s1 = {  1.00,  0.75, -0.50,  1.00};
// __device__ __constant__  constexpr State s2 = {  2.00,  0.75,  0.50,  1.00};
// __device__ __constant__  constexpr State s3 = {  1.00, -0.75,  0.50,  1.00};
// __device__ __constant__  constexpr State s4 = {  3.00, -0.75, -0.50,  1.00};
// constexpr vector2f init_x0y0 = {0.50, 0.50};


// constexpr vector2f init_x0y0 = {0.80, 0.80};
// constexpr vector4f init_rho  = {1.5, 0.5323, 0.138, 0.5323};
// constexpr vector4f init_u    = {0.0, 1.206, 1.206, 0.0};
// constexpr vector4f init_v    = {0.0, 0.0, 1.206, 1.206};
// constexpr vector4f init_p    = {1.5, 0.3, 0.029, 0.3};


// constexpr vector2f init_x0y0 = {0.50, 0.50};
// constexpr vector4f init_rho  = {  1.00,  2.00,  1.00,  3.00};
// constexpr vector4f init_u    = {  0.75,  0.75, -0.75, -0.75};
// constexpr vector4f init_v    = { -0.50,  0.50,  0.50, -0.50};
// constexpr vector4f init_p    = {  1.00,  1.00,  1.00,  1.00};


// rho 使用质量波速
template <typename Type>
HostDevice Type rho_xyz(Type x, Type y, Type z, Type t) {
    constexpr auto s21 = compute_wave_speeds_x(s2, s1).mass;
    constexpr auto s34 = compute_wave_speeds_x(s3, s4).mass;
    constexpr auto s32 = compute_wave_speeds_y(s3, s2).mass;
    constexpr auto s41 = compute_wave_speeds_y(s4, s1).mass;
    Type x_top = init_x0y0[0] + s21 * (1.0 - 0.04 * (t/0.8)) * t;
    Type x_bot = init_x0y0[0] + s34 * t;
    Type y_left = init_x0y0[1] + s32 * t;
    Type y_right = init_x0y0[1] + s41 * (1.0 - 0.04 * (t/0.8)) * t;
    if (x >= x_top && y >= y_right) return static_cast<Type>(s1.rho);
    if (x <  x_top && y >= y_left) return static_cast<Type>(s2.rho);
    if (x <  x_bot  && y <  y_left) return static_cast<Type>(s3.rho);
    return static_cast<Type>(s4.rho);
}

// u 使用动量波速
template <typename Type>
HostDevice Type u_xyz(Type x, Type y, Type z, Type t) {
    constexpr auto s21 = compute_wave_speeds_x(s2, s1).momentum;
    constexpr auto s34 = compute_wave_speeds_x(s3, s4).momentum;
    constexpr auto s32 = compute_wave_speeds_y(s3, s2).momentum;
    constexpr auto s41 = compute_wave_speeds_y(s4, s1).momentum;
    Type x_top = init_x0y0[0] + s21 * (1.0 - 0.04 * (t/0.8)) * t;
    Type x_bot = init_x0y0[0] + s34 * t;
    Type y_left = init_x0y0[1] + s32 * t;
    Type y_right = init_x0y0[1] + s41 * (1.0 - 0.04 * (t/0.8)) * t;
    if (x >= x_top && y >= y_right) return static_cast<Type>(s1.u);
    if (x <  x_top && y >= y_left) return static_cast<Type>(s2.u);
    if (x <  x_bot  && y <  y_left) return static_cast<Type>(s3.u);
    return static_cast<Type>(s4.u);
}

// v 使用动量波速
template <typename Type>
HostDevice Type v_xyz(Type x, Type y, Type z, Type t) {
    constexpr auto s21 = compute_wave_speeds_x(s2, s1).momentum;
    constexpr auto s34 = compute_wave_speeds_x(s3, s4).momentum;
    constexpr auto s32 = compute_wave_speeds_y(s3, s2).momentum;
    constexpr auto s41 = compute_wave_speeds_y(s4, s1).momentum;
    Type x_top = init_x0y0[0] + s21 * (1.0 - 0.04 * (t/0.8)) * t;
    Type x_bot = init_x0y0[0] + s34 * t;
    Type y_left = init_x0y0[1] + s32 * t;
    Type y_right = init_x0y0[1] + s41 * (1.0 - 0.04 * (t/0.8)) * t;
    if (x >= x_top && y >= y_right) return static_cast<Type>(s1.v);
    if (x <  x_top && y >= y_left) return static_cast<Type>(s2.v);
    if (x <  x_bot  && y <  y_left) return static_cast<Type>(s3.v);
    return static_cast<Type>(s4.v);
}

template <typename Type>
HostDevice Type w_xyz(Type x, Type y, Type z, Type t) {
    return static_cast<Type>(0.0);
}

// p 使用能量波速
template <typename Type>
HostDevice Type p_xyz(Type x, Type y, Type z, Type t) {
    constexpr auto s21 = compute_wave_speeds_x(s2, s1).energy;
    constexpr auto s34 = compute_wave_speeds_x(s3, s4).energy;
    constexpr auto s32 = compute_wave_speeds_y(s3, s2).energy;
    constexpr auto s41 = compute_wave_speeds_y(s4, s1).energy;
    Type x_top = init_x0y0[0] + s21 * (1.0 - 0.04 * (t/0.8)) * t;
    Type x_bot = init_x0y0[0] + s34 * t;
    Type y_left = init_x0y0[1] + s32 * t;
    Type y_right = init_x0y0[1] + s41 * (1.0 - 0.04 * (t/0.8)) * t;
    if (x >= x_top && y >= y_right) return static_cast<Type>(s1.p);
    if (x <  x_top && y >= y_left) return static_cast<Type>(s2.p);
    if (x <  x_bot  && y <  y_left) return static_cast<Type>(s3.p);
    return static_cast<Type>(s4.p);
}

// energy 使用前四者计算得到
template <typename Type>
HostDevice Type e_xyz(Type x, Type y, Type z, Type t) {
    Type rho = rho_xyz(x, y, z, t);
    Type u   = u_xyz(x, y, z, t);
    Type v   = v_xyz(x, y, z, t);
    Type w   = 0.0;
    Type p   = p_xyz(x, y, z, t);
    return p / ((get_gamma() - 1.0) * rho) + 0.5 * (u * u + v * v + w * w);
}

#define Filed_Func(filedname) \
HostDevice Scalar filedname##_xyz(const vector3f& xyz, Scalar t){\
    Scalar x = xyz[0], y = xyz[1], z = xyz[2];\
    return filedname##_xyz(x,y,z,t);\
}\
HostDevice Scalar rho##filedname##_xyz(const vector3f& xyz, Scalar t){\
    Scalar x = xyz[0], y = xyz[1], z = xyz[2];\
    return rho_xyz(x,y,z,t)*filedname##_xyz(x,y,z,t);\
}\
Scalar filedname##_Xi(const CompTetrahedron& cell, const vector3f& Xi, Scalar t){\
    const vector3f& xyz = cell.transform_to_physical(Xi);\
    Scalar x = xyz[0], y = xyz[1], z = xyz[2];\
    return filedname##_xyz(x,y,z,t);\
}\
Scalar rho##filedname##_Xi(const CompTetrahedron& cell, const vector3f& Xi, Scalar t){\
    const vector3f& xyz = cell.transform_to_physical(Xi);\
    Scalar x = xyz[0], y = xyz[1], z = xyz[2];\
    return rho_xyz(x,y,z,t)*filedname##_xyz(x,y,z,t);\
}
Filed_Func(rho);
Filed_Func(u);
Filed_Func(v);
Filed_Func(w);
Filed_Func(p);
Filed_Func(e);

#undef Filed_Func

DenseMatrix<5,1> U_Xi(const CompTetrahedron& cell, const vector3f& Xi, Scalar t){
    return {
        rho_Xi(cell,Xi,t),
        rhou_Xi(cell,Xi,t),
        rhov_Xi(cell,Xi,t),
        rhow_Xi(cell,Xi,t),
        rhoe_Xi(cell,Xi,t)
        };
};

#undef Filed_Func