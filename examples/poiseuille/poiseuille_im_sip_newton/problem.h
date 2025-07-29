#include "base/type.h"
#include "base/exact.h"
#include "mesh/mesh.h"


constexpr Scalar param_L = 1;
constexpr Scalar param_theta = 0.01;
constexpr Scalar param_Ma = 0.1;
constexpr Scalar param_gamma = 1.4;
constexpr Scalar param_mu = 1.0e-2;
constexpr Scalar param_dpdx = -8*param_mu/(param_L*param_L);


template<typename Type>
Type rho_xyz(Type x, Type y, Type z, Type t){
    return 1.0;
}
template<typename Type>
Type u_xyz(Type x, Type y, Type z, Type t){
    Type yLy = y * (param_L - y);
    Type u = -0.5 / param_mu * param_dpdx * (yLy + param_theta * yLy * yLy);
    return t>0.0? u: 0;
    // return u;
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
    Type p = 1/(param_gamma*param_Ma*param_Ma)+param_dpdx*x;
    return t>0.0?p:1/(param_gamma*param_Ma*param_Ma);
    // return p;
}
template<typename Type>
Type e_xyz(Type x, Type y, Type z, Type t){
    Type p = p_xyz(x,y,z,t);
    Type u = u_xyz(x,y,z,t);
    Type v = v_xyz(x,y,z,t);
    return p/rho_xyz(x,y,z,t)/(param_gamma-1) + 0.5*(u*u+v*v);
}

template<typename Type>
Type fu_xyz(Type x, Type y, Type z, Type t){
    return param_dpdx*param_theta*(param_L*param_L-6*y*param_L+6*y*y);
}

template<typename Type>
Type fe_xyz(Type x, Type y, Type z, Type t){
    Scalar L11 = 2 * 0.25 / param_mu * param_dpdx * param_dpdx * y*(param_L - y);
    Scalar L12 = 1-param_theta*(param_L*param_L-6*y*param_L+6*y*y)-param_gamma/(param_gamma-1);
    Scalar L13 = 1+param_theta*y*(param_L - y);  //把解代进去多了一个这个，L11*L12*L13
    Scalar L21 = -0.25 / param_mu * param_dpdx * param_dpdx * (param_L - 2*y) * (param_L - 2*y);
    Scalar L22 = 2*param_theta*param_L*y-2*param_theta*y*y+1;
    return L11*L12*L13  +  L21*L22*L22;
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
Filed_Func(fu);
Filed_Func(fe);


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