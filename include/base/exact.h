#pragma once
#include "base/type.h"
#include "mesh/mesh.h"


template<typename Type>
HostDevice Type rho_xyz(Type x, Type y, Type z, Type t = 0.0);
template<typename Type>
HostDevice Type u_xyz(Type x, Type y, Type z, Type t = 0.0);
template<typename Type>
HostDevice Type v_xyz(Type x, Type y, Type z, Type t = 0.0);
template<typename Type>
HostDevice Type w_xyz(Type x, Type y, Type z, Type t = 0.0);
template<typename Type>
HostDevice Type p_xyz(Type x, Type y, Type z, Type t = 0.0);
template<typename Type>
HostDevice Type e_xyz(Type x, Type y, Type z, Type t = 0.0);

template<typename Type>
HostDevice Type frho_xyz(Type x, Type y, Type z, Type t = 0.0);
template<typename Type>
HostDevice Type fu_xyz(Type x, Type y, Type z, Type t = 0.0);
template<typename Type>
HostDevice Type fv_xyz(Type x, Type y, Type z, Type t = 0.0);
template<typename Type>
HostDevice Type fw_xyz(Type x, Type y, Type z, Type t = 0.0);
template<typename Type>
HostDevice Type fp_xyz(Type x, Type y, Type z, Type t = 0.0);
template<typename Type>
HostDevice Type fe_xyz(Type x, Type y, Type z, Type t = 0.0);

#define Filed_Func(filedname) \
HostDevice Scalar filedname##_xyz(const vector3f& xyz, Scalar t = 0.0);\
HostDevice Scalar rho##filedname##_xyz(const vector3f& xyz, Scalar t = 0.0);\
Scalar filedname##_Xi(const CompTetrahedron& cell, const vector3f& Xi, Scalar t = 0.0);\
Scalar rho##filedname##_Xi(const CompTetrahedron& cell, const vector3f& Xi, Scalar t = 0.0);

Filed_Func(rho);
Filed_Func(u);
Filed_Func(v);
Filed_Func(w);
Filed_Func(p);
Filed_Func(e);
Filed_Func(frho);
Filed_Func(fu);
Filed_Func(fv);
Filed_Func(fw);
Filed_Func(fp);
Filed_Func(fe);
#undef Filed_Func

DenseMatrix<5,1> U_Xi(const CompTetrahedron& cell, const vector3f& Xi, Scalar t);
DenseMatrix<3,1> uvw_Xi(const CompTetrahedron& cell, const vector3f& Xi, Scalar t);