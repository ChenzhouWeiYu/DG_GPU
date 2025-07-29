#include "base/type.h"
// #include "base/exact.h"
#include "mesh/mesh.h"
#include "problem.h"

// =====================
// 四象限 Riemann 问题：初值 + 边界条件统一函数
// =====================

// ------------------ 变量专用界面 ------------------

// rho 使用质量波速
template <typename Type>
HostDevice Type rho_xyz(Type x, Type y, Type z, Type t) {
    constexpr auto s21 = compute_wave_speeds_x(s2, s1).mass;
    constexpr auto s34 = compute_wave_speeds_x(s3, s4).mass;
    constexpr auto s32 = compute_wave_speeds_y(s3, s2).mass;
    constexpr auto s41 = compute_wave_speeds_y(s4, s1).mass;
    Type x_top = init_x0y0[0] + s21 * t;
    Type x_bot = init_x0y0[0] + s34 * t;
    Type y_left = init_x0y0[1] + s32 * t;
    Type y_right = init_x0y0[1] + s41 * t;
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
    Type x_top = init_x0y0[0] + s21 * t;
    Type x_bot = init_x0y0[0] + s34 * t;
    Type y_left = init_x0y0[1] + s32 * t;
    Type y_right = init_x0y0[1] + s41 * t;
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
    Type x_top = init_x0y0[0] + s21 * t;
    Type x_bot = init_x0y0[0] + s34 * t;
    Type y_left = init_x0y0[1] + s32 * t;
    Type y_right = init_x0y0[1] + s41 * t;
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
    Type x_top = init_x0y0[0] + s21 * t;
    Type x_bot = init_x0y0[0] + s34 * t;
    Type y_left = init_x0y0[1] + s32 * t;
    Type y_right = init_x0y0[1] + s41 * t;
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
    return p / ((param_gamma - 1.0) * rho) + 0.5 * (u * u + v * v + w * w);
}




// constexpr Scalar param_gamma = 1.4;
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

// constexpr Scalar shock_speed_x21 = (init_rho[0]*init_u[0] - init_rho[1]*init_u[1]) / (init_rho[0] - init_rho[1]);
// constexpr Scalar shock_speed_x34 = (init_rho[3]*init_u[3] - init_rho[2]*init_u[2]) / (init_rho[3] - init_rho[2]);
// constexpr Scalar shock_speed_y41 = (init_rho[0]*init_v[0] - init_rho[3]*init_v[3]) / (init_rho[0] - init_rho[3]);
// constexpr Scalar shock_speed_y32 = (init_rho[1]*init_v[1] - init_rho[2]*init_v[2]) / (init_rho[1] - init_rho[2]);

// constexpr Scalar shock_speed_x21 = (init_rho[0]*init_u[0]*init_u[0] + init_p[0] - init_rho[1]*init_u[1]*init_u[1] - init_p[1]) /
//                                    (init_rho[0]*init_u[0] - init_rho[1]*init_u[1]);
// constexpr Scalar shock_speed_x34 = (init_rho[3]*init_u[3]*init_u[3] + init_p[3] - init_rho[2]*init_u[2]*init_u[2] - init_p[2]) /
//                                    (init_rho[3]*init_u[3] - init_rho[2]*init_u[2]);
// constexpr Scalar shock_speed_y41 = (init_rho[0]*init_v[0]*init_v[0] + init_p[0] - init_rho[3]*init_v[3]*init_v[3] - init_p[3]) /
//                                    (init_rho[0]*init_v[0] - init_rho[3]*init_v[3]);
// constexpr Scalar shock_speed_y32 = (init_rho[1]*init_v[1]*init_v[1] + init_p[1] - init_rho[2]*init_v[2]*init_v[2] - init_p[2]) /
//                                    (init_rho[1]*init_v[1] - init_rho[2]*init_v[2]);

// 界面位置计算函数
// HostDevice Scalar shock_interface_x(Scalar x, Scalar y, Scalar t) {
//     return init_x0y0[0] + (y >= init_x0y0[1] ? shock_speed_x21 : shock_speed_x34) * t;
// }
// HostDevice Scalar shock_interface_y(Scalar x, Scalar y, Scalar t) {
//     return init_x0y0[1] + (x >= init_x0y0[0] ? shock_speed_y41 : shock_speed_y32) * t;
// }

// rho
// template<typename Type>
// HostDevice Type rho_xyz(Type x, Type y, Type z, Type t) {
//     Scalar x_ = shock_interface_x(x, y, t);
//     Scalar y_ = shock_interface_y(x, y, t);
//     if (x >= x_ && y >= y_) return static_cast<Type>(init_rho[0]); // Ω1
//     if (x <  x_ && y >= y_) return static_cast<Type>(init_rho[1]); // Ω2
//     if (x <  x_ && y <  y_) return static_cast<Type>(init_rho[2]); // Ω3
//     return static_cast<Type>(init_rho[3]);                          // Ω4
// }

// // u
// template<typename Type>
// HostDevice Type u_xyz(Type x, Type y, Type z, Type t) {
//     Scalar x_ = shock_interface_x(x, y, t);
//     Scalar y_ = shock_interface_y(x, y, t);
//     if (x >= x_ && y >= y_) return static_cast<Type>(init_u[0]);
//     if (x <  x_ && y >= y_) return static_cast<Type>(init_u[1]);
//     if (x <  x_ && y <  y_) return static_cast<Type>(init_u[2]);
//     return static_cast<Type>(init_u[3]);
// }

// // v
// template<typename Type>
// HostDevice Type v_xyz(Type x, Type y, Type z, Type t) {
//     Scalar x_ = shock_interface_x(x, y, t);
//     Scalar y_ = shock_interface_y(x, y, t);
//     if (x >= x_ && y >= y_) return static_cast<Type>(init_v[0]);
//     if (x <  x_ && y >= y_) return static_cast<Type>(init_v[1]);
//     if (x <  x_ && y <  y_) return static_cast<Type>(init_v[2]);
//     return static_cast<Type>(init_v[3]);
// }

// // w
// template<typename Type>
// HostDevice Type w_xyz(Type x, Type y, Type z, Type t) {
//     return static_cast<Type>(0);
// }

// // p
// template<typename Type>
// HostDevice Type p_xyz(Type x, Type y, Type z, Type t) {
//     Scalar x_ = shock_interface_x(x, y, t);
//     Scalar y_ = shock_interface_y(x, y, t);
//     if (x >= x_ && y >= y_) return static_cast<Type>(init_p[0]);
//     if (x <  x_ && y >= y_) return static_cast<Type>(init_p[1]);
//     if (x <  x_ && y <  y_) return static_cast<Type>(init_p[2]);
//     return static_cast<Type>(init_p[3]);
// }

// // total energy
// template<typename Type>
// HostDevice Type e_xyz(Type x, Type y, Type z, Type t) {
//     Type rho = rho_xyz<Type>(x,y,z,t);
//     Type u   = u_xyz<Type>(x,y,z,t);
//     Type v   = v_xyz<Type>(x,y,z,t);
//     Type w   = w_xyz<Type>(x,y,z,t);
//     Type p   = p_xyz<Type>(x,y,z,t);
//     return p / (param_gamma - 1) / rho + Scalar(0.5)*(u*u + v*v + w*w);
// }

// constexpr Scalar param_gamma = 1.4;

// template<typename Type>
// HostDevice Type rho_xyz(Type x, Type y, Type z, Type t) {
//     if (x >= 0.5 && y >= 0.5)
//         return static_cast<Type>(1);      // Ω1
//     else if (x < 0.5 && y >= 0.5)
//         return static_cast<Type>(2);   // Ω2
//     else if (x < 0.5 && y < 0.5)
//         return static_cast<Type>(1);    // Ω3
//     else // x >= 0.5 && y < 0.5
//         return static_cast<Type>(3);   // Ω4
// }

// template<typename Type>
// HostDevice Type u_xyz(Type x, Type y, Type z, Type t) {
//     if (x >= 0.5 && y >= 0.5)
//         return static_cast<Type>(0.75);      // Ω1
//     else if (x < 0.5 && y >= 0.5)
//         return static_cast<Type>(0.75);    // Ω2
//     else if (x < 0.5 && y < 0.5)
//         return static_cast<Type>(-0.75);    // Ω3
//     else
//         return static_cast<Type>(-0.75);      // Ω4
// }

// template<typename Type>
// HostDevice Type v_xyz(Type x, Type y, Type z, Type t) {
//     if (x >= 0.5 && y >= 0.5)
//         return static_cast<Type>(-0.5);      // Ω1
//     else if (x < 0.5 && y >= 0.5)
//         return static_cast<Type>(0.5);      // Ω2
//     else if (x < 0.5 && y < 0.5)
//         return static_cast<Type>(0.5);    // Ω3
//     else
//         return static_cast<Type>(-0.5);    // Ω4
// }

// template<typename Type>
// HostDevice Type w_xyz(Type x, Type y, Type z, Type t) {
//     return static_cast<Type>(0.0); // 二维问题
// }

// template<typename Type>
// HostDevice Type p_xyz(Type x, Type y, Type z, Type t) {
//     if (x >= 0.5 && y >= 0.5)
//         return static_cast<Type>(1);      // Ω1
//     else if (x < 0.5 && y >= 0.5)
//         return static_cast<Type>(1);      // Ω2
//     else if (x < 0.5 && y < 0.5)
//         return static_cast<Type>(1);    // Ω3
//     else
//         return static_cast<Type>(1);      // Ω4
// }



// template<typename Type>
// HostDevice Type rho_xyz(Type x, Type y, Type z, Type t) {
//     if (x >= 0.8 && y >= 0.8)
//         return static_cast<Type>(1.5);      // Ω1
//     else if (x < 0.8 && y >= 0.8)
//         return static_cast<Type>(0.5323);   // Ω2
//     else if (x < 0.8 && y < 0.8)
//         return static_cast<Type>(0.138);    // Ω3
//     else // x >= 0.8 && y < 0.8
//         return static_cast<Type>(0.5323);   // Ω4
// }

// template<typename Type>
// HostDevice Type u_xyz(Type x, Type y, Type z, Type t) {
//     if (x >= 0.8 && y >= 0.8)
//         return static_cast<Type>(0.0);      // Ω1
//     else if (x < 0.8 && y >= 0.8)
//         return static_cast<Type>(1.206);    // Ω2
//     else if (x < 0.8 && y < 0.8)
//         return static_cast<Type>(1.206);    // Ω3
//     else
//         return static_cast<Type>(0.0);      // Ω4
// }

// template<typename Type>
// HostDevice Type v_xyz(Type x, Type y, Type z, Type t) {
//     if (x >= 0.8 && y >= 0.8)
//         return static_cast<Type>(0.0);      // Ω1
//     else if (x < 0.8 && y >= 0.8)
//         return static_cast<Type>(0.0);      // Ω2
//     else if (x < 0.8 && y < 0.8)
//         return static_cast<Type>(1.206);    // Ω3
//     else
//         return static_cast<Type>(1.206);    // Ω4
// }

// template<typename Type>
// HostDevice Type w_xyz(Type x, Type y, Type z, Type t) {
//     return static_cast<Type>(0.0); // 二维问题
// }

// template<typename Type>
// HostDevice Type p_xyz(Type x, Type y, Type z, Type t) {
//     if (x >= 0.8 && y >= 0.8)
//         return static_cast<Type>(1.5);      // Ω1
//     else if (x < 0.8 && y >= 0.8)
//         return static_cast<Type>(0.3);      // Ω2
//     else if (x < 0.8 && y < 0.8)
//         return static_cast<Type>(0.029);    // Ω3
//     else
//         return static_cast<Type>(0.3);      // Ω4
// }

// template<typename Type>
// HostDevice Type e_xyz(Type x, Type y, Type z, Type t) {
//     Type rho = rho_xyz(x, y, z, t);
//     Type u = u_xyz(x, y, z, t);
//     Type v = v_xyz(x, y, z, t);
//     Type w = w_xyz(x, y, z, t);
//     Type p = p_xyz(x, y, z, t);
//     return p / ((param_gamma - 1.0) * rho) + 0.5 * (u*u + v*v + w*w);
// }



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