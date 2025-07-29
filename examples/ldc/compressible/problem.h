#include "base/type.h"
#include "base/exact.h"
#include "mesh/mesh.h"


// 理想气体比热比   
constexpr Scalar param_gamma = 1.4;
// 气体常数
constexpr Scalar param_R = 8.314;
// 空气的气体常数
constexpr Scalar param_Air_R = 287.05;
// 空气的比热比 
constexpr Scalar param_Air_gamma = 1.4;
// 空气的比热容
constexpr Scalar param_Air_cp = param_Air_R * param_Air_gamma / (param_Air_gamma - 1);
// 空气的参考温度
constexpr Scalar param_Air_T0 = 300.0;
// 空气的声速 c = \sqrt{\gamma R T} 大致量级就够了，只是判定 Ma 是否足够小
constexpr Scalar param_Air_c = 347.2; 

// 初始压强
constexpr Scalar param_p0 = 103000.0;


// 顶盖速度
constexpr Scalar param_U_0 = 1.0;
// 马赫数 << 0.1 就行
constexpr Scalar param_Ma = param_U_0 / param_Air_c;
// 参考密度
constexpr Scalar param_rho_0 = param_p0 / (param_Air_R * param_Air_T0);

// 雷诺数
constexpr Scalar param_Re = 1000.0;
// 参考粘度
constexpr Scalar param_mu = param_rho_0 * param_U_0 * 1.0 / param_Re;


// 可压程序，模拟不可压方腔

template<typename Type>
Type rho_xyz(Type x, Type y, Type z, Type t){
    return param_rho_0;
}

// 顶盖速度为 1
template<typename Type>
Type u_xyz(Type x, Type y, Type z, Type t){
    return (y>1-1e-12) ? param_U_0 : 0.0;
}

template<typename Type>
Type v_xyz(Type x, Type y, Type z, Type t){
    return 0.0;
}

// template<typename Type>
// Type psi_left_corner(Type r, Type theta) {
//     // 左上角流函数 (θ ∈ [-π/2, 0])
//     const Type pi = M_PI;
//     Type sin_theta = sin(theta);
//     Type cos_theta = cos(theta);
//     Type term1 = pi * pi * sin_theta;
//     Type term2 = -4.0 * theta * cos_theta;
//     Type term3 = 2.0 * pi * theta * sin_theta;
//     return (term1 + term2 + term3) / (pi * pi - 4.0);
// }

// template<typename Type>
// Type psi_right_corner(Type r, Type theta) {
//     // 右上角流函数 (θ ∈ [0, π/2])
//     const Type pi = M_PI;
//     Type alpha = pi/2 - theta;  // 转换为与左上角对称的形式
//     Type sin_alpha = sin(alpha);
//     Type cos_alpha = cos(alpha);
//     Type term1 = pi * pi * sin_alpha;
//     Type term2 = -4.0 * alpha * cos_alpha;
//     Type term3 = 2.0 * pi * alpha * sin_alpha;
//     return -(term1 + term2 + term3) / (pi * pi - 4.0);  // 负号保持对称性
// }




// const double PI = 3.141592653589793;
// const double EPS = 1e-16;
// const double ALPHA = PI*PI/4.0 - 1.0;
// const double THRESHOLD = 0.5; // 局部展开阈值

// // 左上角渐进解 (x=0, y=1)
// template<typename Type>
// void left_corner_solution(Type x, Type y, Type* u, Type* v, Type* p) {
//     Type r = sqrt(x*x + (1.0-y)*(1.0-y) + EPS);
//     Type theta = -atan2(1.0-y, x); // θ ∈ [-π/2,0]
    
//     Type cos2t = cos(2*theta);
//     Type sin2t = sin(2*theta);
    
//     // u分量
//     *u = -(0.5 - PI*PI/4 - PI/2*theta + 0.5*cos2t - PI/4*sin2t)/ALPHA;
    
//     // v分量
//     *v = -(-theta - PI/4 + 0.5*sin2t + PI/4*cos2t)/ALPHA;
    
//     // p分量 (如果需要)
//     *p = (1e-3*(PI*cos(theta) + 2*sin(theta)))/(ALPHA*(r + EPS));
    
//     // 指数衰减因子
//     Type decay = std::exp(-8*(1-y)*(1-y));
//     *u *= decay;
//     *v *= decay;
//     *p *= decay;
// }

// // 右上角渐进解 (x=1, y=1)
// template<typename Type>
// void right_corner_solution(Type x, Type y, Type* u, Type* v, Type* p) {
//     Type r = sqrt((1.0-x)*(1.0-x) + (1.0-y)*(1.0-y) + EPS);
//     Type theta = - atan2(1.0-y, 1.0-x); // θ ∈ [0,π/2]
    
//     Type cos2t = cos(2*theta);
//     Type sin2t = sin(2*theta);
    
//     // u分量
//     *u = -(0.5 - PI*PI/4 - PI/2*theta + 0.5*cos2t - PI/4*sin2t)/ALPHA;
    
//     // v分量 (符号与左上角不同)
//     *v = (theta + PI/4 - 0.5*sin2t - PI/4*cos2t)/ALPHA;
    
//     // p分量 (如果需要)
//     *p = -(1e-3*(PI*std::cos(theta) + 2*std::sin(theta)))/(ALPHA*(r + EPS));
    
//     // 指数衰减因子
//     Type decay = std::exp(-8*(1-y)*(1-y));
//     *u *= decay;
//     *v *= decay;
//     *p *= decay;
// }

// // 顶盖驱动流速度场 (完整实现)
// template<typename Type>
// Type u_xyz(Type x, Type y, Type z, Type t) {
//     // 检查是否在左上角附近
//     if (x < THRESHOLD && y > 1.0 - THRESHOLD) {
//         Type u, v, p;
//         left_corner_solution(x, y, &u, &v, &p);
//         return u;
//     }
//     // 检查是否在右上角附近
//     if (x > 1.0 - THRESHOLD && y > 1.0 - THRESHOLD) {
//         Type u, v, p;
//         right_corner_solution(x, y, &u, &v, &p);
//         return u;
//     }
//     // 全局解（顶盖速度为1，其余为0）
//     return (y > 1.0 - 1e-12) ? 1.0 : 0.0;
// }

// template<typename Type>
// Type v_xyz(Type x, Type y, Type z, Type t) {
//     // 检查是否在左上角附近
//     if (x < THRESHOLD && y > 1.0 - THRESHOLD) {
//         Type u, v, p;
//         left_corner_solution(x, y, &u, &v, &p);
//         return v;
//     }
//     // 检查是否在右上角附近
//     else if (x > 1.0 - THRESHOLD && y > 1.0 - THRESHOLD) {
//         Type u, v, p;
//         right_corner_solution(x, y, &u, &v, &p);
//         return v;
//     }
//     // 全局解
//     return 0.0;
// }


template<typename Type>
Type w_xyz(Type x, Type y, Type z, Type t){
    return 0.0;
}

template<typename Type>
Type p_xyz(Type x, Type y, Type z, Type t){
    // return (param_gamma-1)*rho_xyz(x,y,z,t)*e_xyz(x,y,z);
    return rho_xyz(x,y,z,t)*param_Air_R*param_Air_T0;
}
template<typename Type>
Type e_xyz(Type x, Type y, Type z, Type t){
    // Scalar r2 = x*x + y*y;
    // constexpr Scalar r_ds2 = 1.0Q/(2.0Q * 0.1Q * 0.1Q);
    // return 1e-12 + 0.979264*M_1_PI*r_ds2* std::exp(-r2*r_ds2);
    Type p = p_xyz(x,y,z,t);
    Type u = u_xyz(x,y,z,t);
    Type v = v_xyz(x,y,z,t);
    Type w = v_xyz(x,y,z,t);
    return p/rho_xyz(x,y,z,t)/(param_gamma-1) + 0.5*(u*u+v*v+w*w);
    
}

#define Filed_Func(filedname) \
Scalar filedname##_xyz(const vector3f& xyz, Scalar t){\
    Scalar x = xyz[0], y = xyz[1], z = xyz[2];\
    return filedname##_xyz(x,y,z,t);\
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

DenseMatrix<5,1> U_Xi(const CompTetrahedron& cell, const vector3f& Xi, Scalar t){
    return {
        rho_Xi(cell,Xi,t),
        rhou_Xi(cell,Xi,t),
        rhov_Xi(cell,Xi,t),
        rhow_Xi(cell,Xi,t),
        rhoe_Xi(cell,Xi,t)
        };
};

DenseMatrix<3,1> uvw_Xi(const CompTetrahedron& cell, const vector3f& Xi, Scalar t){
    return {
        u_Xi(cell,Xi,t),
        v_Xi(cell,Xi,t),
        w_Xi(cell,Xi,t)
        };
};

#undef Filed_Func