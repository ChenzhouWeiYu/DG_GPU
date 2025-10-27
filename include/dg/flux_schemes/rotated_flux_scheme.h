// rotated_flux_scheme.h
#pragma once
#include "dg/flux_schemes/flux_scheme_base.h"
template<typename FluxImpl, typename Physics>
class RotatedFluxScheme : public FluxSchemeBase<Physics> {
public:
    using PhysicsType = Physics;
    using Base = FluxSchemeBase<Physics>;
    using Base::NEQN;

    HostDevice __forceinline__
    static DenseMatrix<NEQN, 1> compute(
        const Physics& physics,
        const DenseMatrix<NEQN, 1>& U_L,
        const DenseMatrix<NEQN, 1>& U_R,
        Scalar nx, Scalar ny, Scalar nz) {
        
        auto Q = build_rotation_matrix(nx, ny, nz);
        DenseMatrix<NEQN, 1> QU_L = U_L;
        DenseMatrix<NEQN, 1> QU_R = U_R;
        rotate_conserved(QU_L, Q);
        rotate_conserved(QU_R, Q);

        // 关键：直接调用 FluxImpl::compute_1d
        DenseMatrix<NEQN, 1> F_prime = FluxImpl::compute_1d(physics, QU_L, QU_R);

        inverse_rotate_flux(F_prime, Q);
        return F_prime;
    }

    HostDevice __forceinline__
    static DenseMatrix<NEQN, 1> compute(
        const Physics& physics,
        const DenseMatrix<NEQN, 1>& UL,
        const DenseMatrix<NEQN, 1>& UR,
        vector3f vec) {
        return compute(physics, UL, UR, vec[0], vec[1], vec[2]);
    }

    // 构建旋转矩阵 Q (法向 → x 轴)
    HostDevice __forceinline__ 
    static DenseMatrix<3, 3> build_rotation_matrix(Scalar nx, Scalar ny, Scalar nz) {
        DenseMatrix<3, 3> Q;
        // 法向 (假设已归一化)
        Scalar n_norm = sqrt(nx*nx + ny*ny + nz*nz);
        Q(0, 0) = nx/n_norm; Q(0, 1) = ny/n_norm; Q(0, 2) = nz/n_norm;

        // 切向 t1
        vector3f ref = (fabs(nz) > 0.9) ? vector3f{0.0, 1.0, 0.0} : vector3f{0.0, 0.0, 1.0};
        vector3f t1;
        t1[0] = ny * ref[2] - nz * ref[1];
        t1[1] = nz * ref[0] - nx * ref[2];
        t1[2] = nx * ref[1] - ny * ref[0];
        
        Scalar t1_norm = sqrt(t1[0]*t1[0] + t1[1]*t1[1] + t1[2]*t1[2]);
        // if (t1_norm > 1e-12) {
        //     t1[0] /= t1_norm; t1[1] /= t1_norm; t1[2] /= t1_norm;
        // } else {
        //     t1 = {1.0, 0.0, 0.0};
        // }
        Q(1, 0) = t1[0]/t1_norm; Q(1, 1) = t1[1]/t1_norm; Q(1, 2) = t1[2]/t1_norm;

        // 切向 t2 = n × t1
        vector3f t2;
        t2[0] = ny * t1[2] - nz * t1[1];
        t2[1] = nz * t1[0] - nx * t1[2];
        t2[2] = nx * t1[1] - ny * t1[0];
        Scalar t2_norm = sqrt(t2[0]*t2[0] + t2[1]*t2[1] + t2[2]*t2[2]);
        Q(2, 0) = t2[0]/t2_norm; Q(2, 1) = t2[1]/t2_norm; Q(2, 2) = t2[2]/t2_norm;

        return Q;
    }

    // 旋转守恒变量 (仅动量)
    HostDevice __forceinline__ 
    static void rotate_conserved(DenseMatrix<NEQN, 1>& U, const DenseMatrix<3, 3>& Q) {
        Scalar rho_u = U[1], rho_v = U[2], rho_w = U[3];
        U[1] = Q(0,0)*rho_u + Q(0,1)*rho_v + Q(0,2)*rho_w; // 法向动量
        U[2] = Q(1,0)*rho_u + Q(1,1)*rho_v + Q(1,2)*rho_w; // 切向1
        U[3] = Q(2,0)*rho_u + Q(2,1)*rho_v + Q(2,2)*rho_w; // 切向2
    }

    // 逆旋转通量 (仅动量)
    HostDevice __forceinline__ 
    static void inverse_rotate_flux(DenseMatrix<NEQN, 1>& F, const DenseMatrix<3, 3>& Q) {
        Scalar F_n = F[1], F_t1 = F[2], F_t2 = F[3];
        F[1] = Q(0,0)*F_n + Q(1,0)*F_t1 + Q(2,0)*F_t2; // x-momentum
        F[2] = Q(0,1)*F_n + Q(1,1)*F_t1 + Q(2,1)*F_t2; // y-momentum
        F[3] = Q(0,2)*F_n + Q(1,2)*F_t1 + Q(2,2)*F_t2; // z-momentum
    }
};