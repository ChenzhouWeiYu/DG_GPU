#include "base/type.h"
#include "mesh/computing_mesh.h"
#include "base/exact.h"

HostDevice Scalar get_gamma() {return 1.4;}

template<typename Type>
HostDevice Type rho_xyz(Type x, Type y, Type z, Type t){
    return 1.0;
}
template<typename Type>
HostDevice Type u_xyz(Type x, Type y, Type z, Type t){
    return 0.0;
}
template<typename Type>
HostDevice Type v_xyz(Type x, Type y, Type z, Type t){
    return 0.0;
}
template<typename Type>
HostDevice Type w_xyz(Type x, Type y, Type z, Type t){
    return 0.0;
}
template<typename Type>
HostDevice Type p_xyz(Type x, Type y, Type z, Type t){
    // return (param_gamma-1)*rho_xyz(x,y,z,t)*e_xyz(x,y,z);
    return x*x + y*y < 1e-4 ? 7e9 : 1e-5;
}

template<typename Type>
HostDevice Type e_xyz(Type x, Type y, Type z, Type t) {
    Type rho = rho_xyz<Type>(x,y,z,t);
    Type u   = u_xyz<Type>(x,y,z,t);
    Type v   = v_xyz<Type>(x,y,z,t);
    Type w   = w_xyz<Type>(x,y,z,t);
    Type p   = p_xyz<Type>(x,y,z,t);
    return p / (get_gamma() - 1) / rho + Scalar(0.5)*(u*u + v*v + w*w);
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