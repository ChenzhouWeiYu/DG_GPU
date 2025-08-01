#include "base/type.h"
#include "mesh/computing_mesh.h"
#include "base/exact.h"

HostDevice Scalar get_gamma() {return 1.4;}


template <typename Type>
HostDevice Type rho_xyz(Type x, Type y, Type z, Type t) {
    return static_cast<Type>(1.0);
}

template <typename Type>
HostDevice Type u_xyz(Type x, Type y, Type z, Type t) {
    return static_cast<Type>(0.0);
}

template <typename Type>
HostDevice Type v_xyz(Type x, Type y, Type z, Type t) {
    return static_cast<Type>(0.0);
}

template <typename Type>
HostDevice Type w_xyz(Type x, Type y, Type z, Type t) {
    return static_cast<Type>(0.0);
}

template <typename Type>
HostDevice Type p_xyz(Type x, Type y, Type z, Type t) {
    return static_cast<Type>((get_gamma()-1) * rho_xyz(x,y,z,t) * e_xyz(x,y,z));
}

template <typename Type>
HostDevice Type e_xyz(Type x, Type y, Type z, Type t) {
    Scalar r2 = x*x + y*y;
    constexpr Scalar r_ds2 = 1.0/(2.0 * 0.1 * 0.1);
    return 1e-12 + 0.979264*M_1_PI*r_ds2* std::exp(-r2*r_ds2);
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