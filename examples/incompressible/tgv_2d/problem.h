#include "base/type.h"
#include "base/exact.h"
#include "mesh/mesh.h"



constexpr Scalar param_gamma = 1.4;
constexpr Scalar param_delta_0 = 0.01;
constexpr Scalar param_tau = 1;
constexpr Scalar param_mu = 0.01;
constexpr Scalar param_rho_0 = 1.0;  
constexpr Scalar param_U_0 = 1.0;
constexpr Scalar param_k = 2*M_PI;


template<typename Type>
Type rho_xyz(Type x, Type y, Type z, Type t){
    return param_rho_0*1.0;
}
template<typename Type>
Type u_xyz(Type x, Type y, Type z, Type t){
    return std::sin(param_k*x)*std::cos(param_k*y) * std::exp(-(2*param_k*param_k*param_mu)*t);
}
template<typename Type>
Type v_xyz(Type x, Type y, Type z, Type t){
    return -std::cos(param_k*x)*std::sin(param_k*y) * std::exp(-(2*param_k*param_k*param_mu)*t);
}
template<typename Type>
Type w_xyz(Type x, Type y, Type z, Type t){
    return 0.0;
}
template<typename Type>
Type p_xyz(Type x, Type y, Type z, Type t){
    return param_rho_0/4 * (std::cos(2*param_k*x) + std::cos(2*param_k*y)) * std::exp(-(4*param_k*param_k*param_mu)*t);
}
template<typename Type>
Type e_xyz(Type x, Type y, Type z, Type t){
    return 0.0;
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

#undef Filed_Func