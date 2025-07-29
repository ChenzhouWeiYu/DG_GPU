#pragma once
#include "dg/dg_schemes/explicit_convection_gpu/explicit_convection_gpu.h"
#include "dg/dg_schemes/explicit_convection_gpu/explicit_convection_gpu_impl.h"
// #include "../examples/Eular2D/DoubleMach_GPU_limiter_GPU/problem.h"



// #include "base/type.h"
// #include "base/exact.h"
// #include "mesh/mesh.h"



// constexpr Scalar param_gamma = 1.4;

// template<typename Type>
// inline HostDevice Type rho_xyz(Type x, Type y, Type z, Type t){
//     // return x>0.8 ? (y>0.8 ? 1.5 : 0.5323) : (y>0.8 ? 0.5323 : 0.138);
//     if (t>0) 
//         return x<1.0/6.0+(y>0.5?(1+20*t)/std::sqrt(3.0):0) ? 8 : 1.4;
//     else 
//         return x<1.0/6.0+y/std::sqrt(3.0) ? 8 : 1.4;
// }
// template<typename Type>
// inline HostDevice Type u_xyz(Type x, Type y, Type z, Type t){
//     // return x>0.8 ? 0 : 1.206;
//     if (t>0) 
//         return x<1.0/6.0+(y>0.5?(1+20*t)/std::sqrt(3.0):0) ? 8.25*std::cos(-M_PI/6.0) : 0;
//     else 
//         return x<1.0/6.0+y/std::sqrt(3.0) ? 8.25*std::cos(-M_PI/6.0) : 0;
// }
// template<typename Type>
// inline HostDevice Type v_xyz(Type x, Type y, Type z, Type t){
//     // return y>0.8 ? 0 : 1.206;
//     if (t>0) 
//         return x<1.0/6.0+(y>0.5?(1+20*t)/std::sqrt(3.0):0) ? 8.25*std::sin(-M_PI/6.0) : 0;
//     else 
//         return x<1.0/6.0+y/std::sqrt(3.0) ? 8.25*std::sin(-M_PI/6.0) : 0;
// }
// template<typename Type>
// inline HostDevice Type w_xyz(Type x, Type y, Type z, Type t){
//     return 0.0;
// }
// template<typename Type>
// inline HostDevice Type p_xyz(Type x, Type y, Type z, Type t){
//     // return (param_gamma-1)*rho_xyz(x,y,z,t)*e_xyz(x,y,z);
//     // return x>0.8 ? (y>0.8 ? 1.5 : 0.3) : (y>0.8 ? 0.3 : 0.029); 
//     if (t>0) 
//         return x<1.0/6.0+(y>0.5?(1+20*t)/std::sqrt(3.0):0) ? 116.5 : 1;
//     else 
//         return x<1.0/6.0+y/std::sqrt(3.0) ? 116.5 : 1;
// }
// template<typename Type>
// inline HostDevice Type e_xyz(Type x, Type y, Type z, Type t){
//     // Scalar r2 = x*x + y*y;
//     // constexpr Scalar r_ds2 = 1.0Q/(2.0Q * 0.1Q * 0.1Q);
//     // return 1e-12 + 0.979264*M_1_PI*r_ds2* std::exp(-r2*r_ds2);
//     Type p = p_xyz(x,y,z,t);
//     Type u = u_xyz(x,y,z,t);
//     Type v = v_xyz(x,y,z,t);
//     Type w = v_xyz(x,y,z,t);
//     return p/rho_xyz(x,y,z,t)/(param_gamma-1) + 0.5*(u*u+v*v+w*w);
// }

// #define Filed_Func(filedname) \
// inline HostDevice Scalar filedname##_xyz(const vector3f& xyz, Scalar t){\
//     Scalar x = xyz[0], y = xyz[1], z = xyz[2];\
//     return filedname##_xyz(x,y,z,t);\
// }\
// inline HostDevice Scalar rho##filedname##_xyz(const vector3f& xyz, Scalar t){\
//     Scalar x = xyz[0], y = xyz[1], z = xyz[2];\
//     return rho_xyz(x,y,z,t)*filedname##_xyz(x,y,z,t);\
// }\
// inline HostDevice Scalar filedname##_Xi(const CompTetrahedron& cell, const vector3f& Xi, Scalar t){\
//     const vector3f& xyz = cell.transform_to_physical(Xi);\
//     Scalar x = xyz[0], y = xyz[1], z = xyz[2];\
//     return filedname##_xyz(x,y,z,t);\
// }\
// inline HostDevice Scalar rho##filedname##_Xi(const CompTetrahedron& cell, const vector3f& Xi, Scalar t){\
//     const vector3f& xyz = cell.transform_to_physical(Xi);\
//     Scalar x = xyz[0], y = xyz[1], z = xyz[2];\
//     return rho_xyz(x,y,z,t)*filedname##_xyz(x,y,z,t);\
// }
// Filed_Func(rho);
// Filed_Func(u);
// Filed_Func(v);
// Filed_Func(w);
// Filed_Func(p);
// Filed_Func(e);

// #undef Filed_Func







// device 函数：Basis、Flux 都是可以直接用的

template<uInt Order, uInt N, typename Flux, typename GaussQuadCell, typename GaussQuadFace>
__global__ void eval_boundarys_kernel(const GPUTriangleFace* mesh_faces, uInt num_faces,
                                    const vector3f* mesh_points, Scalar time,
                                    const DenseMatrix<5*N,1>* U,
                                    DenseMatrix<5*N,1>* rhs){
    using Basis = DGBasisEvaluator<Order>;
    // constexpr uInt N = Basis::NumBasis;
    constexpr uInt num_face_points = GaussQuadFace::num_points;
    constexpr auto Qpoints = GaussQuadFace::get_points();
    constexpr auto Qweights = GaussQuadFace::get_weights();

    uInt fid = blockIdx.x * blockDim.x + threadIdx.x;
    if (fid >= num_faces) return;

    const GPUTriangleFace& face = mesh_faces[fid];
    const uInt cell_L = face.neighbor_cells[0];
    const uInt cell_R = face.neighbor_cells[1];

    if (cell_R != uInt(-1)) return; // 只处理边界面

    // const GPUTetrahedron& cell = mesh_faces[cell_L];
    const vector3f& face_p0 = mesh_points[face.nodes[0]];
    const vector3f& face_p1 = mesh_points[face.nodes[1]];
    const vector3f& face_p2 = mesh_points[face.nodes[2]];
    const DenseMatrix<5*N,1>& coef_L = U[cell_L];
    const DenseMatrix<5,1> U_avg{coef_L[0],coef_L[1],coef_L[2],coef_L[3],coef_L[4]};
    for (uInt g = 0; g < num_face_points; ++g) {
        const vector2f& uv = Qpoints[g];
        const Scalar jac_weight = Qweights[g] * face.area * 2;
        auto xi_L = transform_to_cell(face, uv, 0);
        auto basis_L = Basis::eval_all(xi_L[0], xi_L[1], xi_L[2]);

        DenseMatrix<5,1> U_L = DenseMatrix<5,1>::Zeros();
        for (uInt bid = 0; bid < N; ++bid)
            for (uInt k = 0; k < 5; ++k)
                U_L(k,0) += basis_L[bid] * coef_L(5*bid+k,0);
        
        Scalar x = face_p0[0] * (1 - uv[0] - uv[1]) +
                    face_p1[0] * uv[0] +
                    face_p2[0] * uv[1];
        Scalar y = face_p0[1] * (1 - uv[0] - uv[1]) +
                    face_p1[1] * uv[0] +
                    face_p2[1] * uv[1];
        Scalar z = face_p0[2] * (1 - uv[0] - uv[1]) +
                    face_p1[2] * uv[0] +
                    face_p2[2] * uv[1];
        vector3f xyz = {x, y, z};
        DenseMatrix<5,1> U_R = U_L; // 默认 U_R = U_L
        if (face.boundaryType == BoundaryType::Dirichlet) {
            U_R = DenseMatrix<5,1>({rho_xyz(xyz, time),
                                    rhou_xyz(xyz, time),
                                    rhov_xyz(xyz, time),
                                    rhow_xyz(xyz, time),
                                    rhoe_xyz(xyz, time)});
            // U_R = U_R + 1e-2 * (U_R - U_L);
        }
        else if (face.boundaryType == BoundaryType::Pseudo3DZ) {
            U_R[3] = -U_L[3]; // 只反转 z 速度
            // U_R[3] = 0.0;
        }
        else if (face.boundaryType == BoundaryType::Pseudo3DY) {
            U_R[2] = -U_L[2]; // 只反转 y 速度
            // U_R[2] = 0.0;
        }
        else if (face.boundaryType == BoundaryType::Pseudo3DX) {
            U_R[1] = -U_L[1]; // 只反转 x 速度
            // U_R[1] = 0.0;
        }
        else if (face.boundaryType == BoundaryType::Symmetry) {
            Scalar dot_product = U_L[1]*face.normal[0] + U_L[2]*face.normal[1] + U_L[3]*face.normal[2];
            U_R[0] = U_L[0];
            U_R[1] = U_L[1] - 2.0 * dot_product * face.normal[0]; // 反转 x 速度
            U_R[2] = U_L[2] - 2.0 * dot_product * face.normal[1]; // 反转 y 速度
            U_R[3] = U_L[3] - 2.0 * dot_product * face.normal[2]; // 反转 z 速度
            U_R[4] = U_L[4];
        }
        else if (face.boundaryType == BoundaryType::Neumann){
            // U_R = U_avg + 0.0*(U_avg - U_L);
        }
        else {
            continue; // 其他类型暂时跳过
        }

        // auto FU_L = Flux::computeFlux(U_L);
        // auto FU_R = Flux::computeFlux(U_R);
        // auto FUn_L = FU_L.multiply(DenseMatrix<3,1>(face.normal));
        // auto FUn_R = FU_R.multiply(DenseMatrix<3,1>(face.normal));

        // Scalar lambda = Flux::computeWaveSpeed(U_L, U_R);
        // auto LF_flux = 0.5 * (FUn_L + FUn_R + lambda * (U_L - U_R));

        
        // auto LF_flux = Flux::computeLaxFriedrichsFlux(U_L,U_R,face.normal);
        // auto LF_flux = Flux::computeHLLFlux(U_L,U_R,face.normal);
        auto LF_flux = Flux::computeNumericalFlux(U_L,U_R,face.normal);
        
        // if (face.boundaryType == BoundaryType::Pseudo3DZ) {
        //     LF_flux[3] = 0.0; // 只保留 x, y 速度分量
        // }
        // else if (face.boundaryType == BoundaryType::Pseudo3DY) {
        //     LF_flux[2] = 0.0; // 只保留 x, z 速度分量
        // }
        // else if (face.boundaryType == BoundaryType::Pseudo3DX) {
        //     LF_flux[1] = 0.0; // 只保留 y, z 速度分量  
        // }
        // if (face.boundaryType == BoundaryType::Pseudo3DY)
        // printf("Boundary face %u: xyz=(%.3f, %.3f, %.3f), normal=(%.3f, %.3f, %.3f), "
        //        "U_L=(%.3f, %.3f, %.3f, %.3f, %.3f), U_R=(%.3f, %.3f, %.3f, %.3f, %.3f), "
        //        "FUn_L=(%.3f, %.3f, %.3f), FUn_R=(%.3f, %.3f, %.3f), LF_flux=(%.3f, %.3f, %.3f, %.3f, %.3f)\n",
        //        fid,
        //        xyz[0], xyz[1], xyz[2],
        //        face.normal[0], face.normal[1], face.normal[2],
        //        U_L(0,0), U_L(1,0), U_L(2,0), U_L(3,0), U_L(4,0),
        //        U_R(0,0), U_R(1,0), U_R(2,0), U_R(3,0), U_R(4,0),
        //        FUn_L(0,0), FUn_L(1,0), FUn_L(2,0),
        //        FUn_R(0,0), FUn_R(1,0), FUn_R(2,0),
        //        LF_flux(0,0), LF_flux(1,0), LF_flux(2,0), LF_flux(3,0), LF_flux(4,0));
        for (uInt j = 0; j < N; ++j) {
            Scalar phi_jL = basis_L[j];

            atomicAdd(&rhs[cell_L](5*j+0,0), LF_flux(0,0) * phi_jL * jac_weight);
            atomicAdd(&rhs[cell_L](5*j+1,0), LF_flux(1,0) * phi_jL * jac_weight);
            atomicAdd(&rhs[cell_L](5*j+2,0), LF_flux(2,0) * phi_jL * jac_weight);
            atomicAdd(&rhs[cell_L](5*j+3,0), LF_flux(3,0) * phi_jL * jac_weight);
            atomicAdd(&rhs[cell_L](5*j+4,0), LF_flux(4,0) * phi_jL * jac_weight);
        }
    }
}

// Kernel launcher
template<uInt Order, typename Flux, typename GaussQuadCell, typename GaussQuadFace>
void ExplicitConvectionGPU<Order, Flux, GaussQuadCell, GaussQuadFace>::eval_boundarys(
    const DeviceMesh& mesh, const LongVectorDevice<5*N>& U, LongVectorDevice<5*N>& rhs, Scalar time)
{
    dim3 block(32);
    dim3 grid((mesh.num_faces() + block.x - 1) / block.x);
    eval_boundarys_kernel<Order, N, Flux, QuadC, QuadF><<<grid, block>>>(mesh.device_faces(), mesh.num_faces(), mesh.device_points(), time, U.d_blocks, rhs.d_blocks);
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    // }
    // cudaDeviceSynchronize();
}