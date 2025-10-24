#pragma once
#include "base/type.h"
#include "matrix/dense_matrix.h"

template<typename Physics>
class FluxSchemeBase {
public:
    static constexpr uInt NEQN = Physics::NEQN;

    // HostDevice
    // static DenseMatrix<NEQN, 1> compute(
    //     const Physics& physics,
    //     const DenseMatrix<NEQN, 1>& UL,
    //     const DenseMatrix<NEQN, 1>& UR,
    //     Scalar nx, Scalar ny, Scalar nz);

    
    // HostDevice
    // static DenseMatrix<NEQN, 1> compute(
    //     const Physics& physics,
    //     const DenseMatrix<NEQN, 1>& UL,
    //     const DenseMatrix<NEQN, 1>& UR,
    //     vector3f vec) {
    //         return compute(physics, UL, UR, vec[0], vec[1], vec[2]);
    //     }
        

protected:
    HostDevice inline 
    static Scalar positive(Scalar val) { 
        return fmax(val, 1e-16); 
    }

    // 构建旋转矩阵 Q (法向 → x 轴)
    HostDevice inline 
    static DenseMatrix<3, 3> build_rotation_matrix(Scalar nx, Scalar ny, Scalar nz) {
        DenseMatrix<3, 3> Q;
        // 法向 (假设已归一化)
        Q(0, 0) = nx; Q(0, 1) = ny; Q(0, 2) = nz;

        // 切向 t1
        vector3f ref = (fabs(nz) > 0.9) ? vector3f{0.0, 1.0, 0.0} : vector3f{0.0, 0.0, 1.0};
        vector3f t1;
        t1[0] = ny * ref[2] - nz * ref[1];
        t1[1] = nz * ref[0] - nx * ref[2];
        t1[2] = nx * ref[1] - ny * ref[0];
        
        Scalar t1_norm = sqrt(t1[0]*t1[0] + t1[1]*t1[1] + t1[2]*t1[2]);
        if (t1_norm > 1e-12) {
            t1[0] /= t1_norm; t1[1] /= t1_norm; t1[2] /= t1_norm;
        } else {
            t1 = {1.0, 0.0, 0.0};
        }
        Q(1, 0) = t1[0]; Q(1, 1) = t1[1]; Q(1, 2) = t1[2];

        // 切向 t2 = n × t1
        vector3f t2;
        t2[0] = ny * t1[2] - nz * t1[1];
        t2[1] = nz * t1[0] - nx * t1[2];
        t2[2] = nx * t1[1] - ny * t1[0];
        Q(2, 0) = t2[0]; Q(2, 1) = t2[1]; Q(2, 2) = t2[2];

        return Q;
    }

    // 旋转守恒变量 (仅动量)
    HostDevice inline 
    static void rotate_conserved(DenseMatrix<NEQN, 1>& U, const DenseMatrix<3, 3>& Q) {
        Scalar rho_u = U[1], rho_v = U[2], rho_w = U[3];
        U[1] = Q(0,0)*rho_u + Q(0,1)*rho_v + Q(0,2)*rho_w; // 法向动量
        U[2] = Q(1,0)*rho_u + Q(1,1)*rho_v + Q(1,2)*rho_w; // 切向1
        U[3] = Q(2,0)*rho_u + Q(2,1)*rho_v + Q(2,2)*rho_w; // 切向2
    }

    // 逆旋转通量 (仅动量)
    HostDevice inline 
    static void inverse_rotate_flux(DenseMatrix<NEQN, 1>& F, const DenseMatrix<3, 3>& Q) {
        Scalar F_n = F[1], F_t1 = F[2], F_t2 = F[3];
        F[1] = Q(0,0)*F_n + Q(1,0)*F_t1 + Q(2,0)*F_t2; // x-momentum
        F[2] = Q(0,1)*F_n + Q(1,1)*F_t1 + Q(2,1)*F_t2; // y-momentum
        F[3] = Q(0,2)*F_n + Q(1,2)*F_t1 + Q(2,2)*F_t2; // z-momentum
    }
};