#include "base/type.h"
#include "base/exact.h"
#include "mesh/mesh.h"



constexpr Scalar param_gamma = 1.4;
constexpr Scalar param_delta_0 = 0.01;
constexpr Scalar param_tau = 1;
constexpr Scalar param_mu = 0.001;
constexpr Scalar param_rho_0 = 1.0;  
constexpr Scalar param_U_0 = 1.0;


template<typename Type>
Type rho_xyz(Type x, Type y, Type z, Type t){
    return param_rho_0*1.0;
}
template<typename Type>
Type u_xyz(Type x, Type y, Type z, Type t){
    // return -std::sin(2*M_PI*y);
    // return 1;
    // return param_U_0 * std::tanh(y / param_delta_0) * std::exp(-t / param_tau);
    return -cos(x)*sin(y) + (t<1e-16?0.25:0)*(sin(2*M_PI*x)*sin(2*M_PI*y));
}
template<typename Type>
Type v_xyz(Type x, Type y, Type z, Type t){
    // return std::sin(2*M_PI*x);
    // return 0;
    return sin(x)*cos(y) - (t<1e-16?0.25:0)*(sin(2*M_PI*x)*sin(2*M_PI*y));
}
template<typename Type>
Type w_xyz(Type x, Type y, Type z, Type t){
    return 0.0;
}
template<typename Type>
Type p_xyz(Type x, Type y, Type z, Type t){
    // return -std::cos(2*M_PI*x)*std::cos(2*M_PI*y);
    // return 170;
    // return -param_rho_0*param_U_0*param_U_0*\
    //     (std::tanh(y/param_delta_0)*std::tanh(y/param_delta_0)+1) * std::exp(-2*t/param_tau);
    return -0.25*(cos(2*x) + cos(2*y)) + (t<1e-16?0.0625:0)*(sin(4*M_PI*x)*sin(4*M_PI*y));
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