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
constexpr Scalar param_Re = 105.0;
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
    Scalar r2 = x*x + y*y;
    return r2 > 0.5*0.5 + 1e-12 ? param_U_0 : 0.0; // 0.5 是半径
}

template<typename Type>
Type v_xyz(Type x, Type y, Type z, Type t){
    return 0.0;
}

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
