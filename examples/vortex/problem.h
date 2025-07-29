#include "base/type.h"
#include "base/exact.h"
#include "mesh/mesh.h"



constexpr Scalar init_x0 = 4.0;
constexpr Scalar init_y0 = 4.5;

constexpr Scalar param_phi = 5;
constexpr Scalar param_gamma = 1.4;

constexpr Scalar velocity_u = 1.0;
constexpr Scalar velocity_v = 0.5;


template<typename Type>
Type rho_xyz(Type x, Type y, Type z, Type t){
    Type x0 = init_x0 + velocity_u * t;
    Type y0 = init_y0 + velocity_v * t;
    Type r2=(x-x0)*(x-x0)+(y-y0)*(y-y0);
    constexpr Type slope = (param_phi*param_phi*(param_gamma-1))/(16*M_PI*M_PI*param_gamma);
    Type rho = std::pow(1-slope*std::exp(2*(1-r2)), 1.0/(param_gamma-1));
    return rho;
}
template<typename Type>
Type u_xyz(Type x, Type y, Type z, Type t){
    Type x0 = init_x0 + velocity_u * t;
    Type y0 = init_y0 + velocity_v * t;
    Type r2=(x-x0)*(x-x0)+(y-y0)*(y-y0);
    Type u = velocity_u - param_phi*(y-y0)/(2*M_PI)*std::exp(1-r2);
    return u;
}
template<typename Type>
Type v_xyz(Type x, Type y, Type z, Type t){
    Type x0 = init_x0 + velocity_u * t;
    Type y0 = init_y0 + velocity_v * t;
    Type r2=(x-x0)*(x-x0)+(y-y0)*(y-y0);
    Type v = velocity_v + param_phi*(x-x0)/(2*M_PI)*std::exp(1-r2);
    return v;
}
template<typename Type>
Type w_xyz(Type x, Type y, Type z, Type t){
    Type x0 = init_x0 + velocity_u * t;
    Type y0 = init_y0 + velocity_v * t;
    Type r2=(x-x0)*(x-x0)+(y-y0)*(y-y0);
    Type w = 0.0;
    return w;
}
template<typename Type>
Type p_xyz(Type x, Type y, Type z, Type t){
    Type x0 = init_x0 + velocity_u * t;
    Type y0 = init_y0 + velocity_v * t;
    Type r2=(x-x0)*(x-x0)+(y-y0)*(y-y0);
    Type p = std::pow(rho_xyz(x,y,z,t),param_gamma);
    return p;
}
template<typename Type>
Type e_xyz(Type x, Type y, Type z, Type t){
    Type x0 = init_x0 + velocity_u * t;
    Type y0 = init_y0 + velocity_v * t;
    Type r2=(x-x0)*(x-x0)+(y-y0)*(y-y0);
    Type p = p_xyz(x,y,z,t);
    Type u = u_xyz(x,y,z,t);
    Type v = v_xyz(x,y,z,t);
    return p/rho_xyz(x,y,z,t)/(param_gamma-1) + 0.5*(u*u+v*v);
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

DenseMatrix<3,1> uvw_Xi(const CompTetrahedron& cell, const vector3f& Xi, Scalar t){
    return {
        u_Xi(cell,Xi,t),
        v_Xi(cell,Xi,t),
        w_Xi(cell,Xi,t)
        };
};