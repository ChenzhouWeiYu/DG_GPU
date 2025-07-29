#pragma once
#include "base/type.h"
#include "dg/dg_basis/dg_basis.h"
#include "mesh/mesh.h"
#include "matrix/matrix.h"
#include "base/exact.h"
#include "dg/dg_flux/diffusion_physical_flux.h"

template<uInt Order=3, typename Flux = AirFluxD, 
        typename GaussQuadCell = GaussLegendreTet::Auto, 
        typename GaussQuadFace = GaussLegendreTri::Auto>
class ImplicitDiffusion {
private:
    using BlockMat = DenseMatrix<5,5>;
    using Basis = DGBasisEvaluator<Order>;

    using QuadC = typename std::conditional_t<
        std::is_same_v<GaussQuadCell, GaussLegendreTet::Auto>,
        typename AutoQuadSelector<Order, GaussLegendreTet::Auto>::type,
        GaussQuadCell
    >;
    using QuadF = typename std::conditional_t<
        std::is_same_v<GaussQuadFace, GaussLegendreTri::Auto>,
        typename AutoQuadSelector<Order, GaussLegendreTri::Auto>::type,
        GaussQuadFace
    >;

    static constexpr uInt N = Basis::NumBasis;
    vector3f transform_to_cell(const CompTriangleFace& face, const vector2f& uv, uInt side) const ;
    Scalar m_mu;

public:
    ImplicitDiffusion(Scalar mu = 1e-2):m_mu(mu){};
    void assemble(const ComputingMesh& mesh, 
                 const LongVector<5*N>& old_solution,
                 const Scalar curr_time,
                 BlockSparseMatrix<5*N,5*N>& sparse_mat,
                 LongVector<5*N>& sparse_rhs);
    void assemble_cells(const ComputingMesh& mesh, 
                 const LongVector<5*N>& old_solution,
                 const Scalar curr_time,
                 BlockSparseMatrix<5*N,5*N>& sparse_mat,
                 LongVector<5*N>& sparse_rhs);
    void assemble_internals(const ComputingMesh& mesh, 
                 const LongVector<5*N>& old_solution,
                 const Scalar curr_time,
                 BlockSparseMatrix<5*N,5*N>& sparse_mat,
                 LongVector<5*N>& sparse_rhs);
    void assemble_boundarys(const ComputingMesh& mesh, 
                 const LongVector<5*N>& old_solution,
                 const Scalar curr_time,
                 BlockSparseMatrix<5*N,5*N>& sparse_mat,
                 LongVector<5*N>& sparse_rhs);
};






#define Explicit_For_Flux(Order) \
extern template class ImplicitDiffusion<Order,AirFluxD>;\
extern template class ImplicitDiffusion<Order,MonatomicFluxD>;\
extern template class ImplicitDiffusion<Order+1,AirFluxD>;\
extern template class ImplicitDiffusion<Order+1,MonatomicFluxD>;\
extern template class ImplicitDiffusion<Order+2,AirFluxD>;\
extern template class ImplicitDiffusion<Order+2,MonatomicFluxD>;

Explicit_For_Flux(0)
Explicit_For_Flux(3)

#undef Explicit_For_Flux









// template<uInt Order=3, typename GaussQuadCell = GaussLegendreTet::Auto, typename GaussQuadFace = GaussLegendreTri::Auto>
// class ImplicitDiffusion_NewtonConvection {
//     using BlockMat = DenseMatrix<5,5>;
//     using Basis = DGBasisEvaluator<Order>;

//     using QuadC = typename std::conditional_t<
//         std::is_same_v<GaussQuadCell, GaussLegendreTet::Auto>,
//         typename AutoQuadSelector<Order, GaussLegendreTet::Auto>::type,
//         GaussQuadCell
//     >;
//     using QuadF = typename std::conditional_t<
//         std::is_same_v<GaussQuadFace, GaussLegendreTri::Auto>,
//         typename AutoQuadSelector<Order, GaussLegendreTri::Auto>::type,
//         GaussQuadFace
//     >;

//     static constexpr uInt N = Basis::NumBasis;


//     Scalar m_mu, m_Pr;
// public:
//     ImplicitDiffusion_NewtonConvection(Scalar mu = 1e-2, Scalar Pr = 0.72):m_mu(mu),m_Pr(Pr){};


//     std::array<DenseMatrix<5,5>,3> assemble_jacobian(const DenseMatrix<5,1>& U, const DenseMatrix<5,3>& grad_U) {
//         Scalar rho = U[0], rho_u = U[1], rho_v = U[2], rho_w = U[3], rho_e = U[4];
//         Scalar u = U[1]/U[0], v = U[2]/U[0], w = U[3]/U[0], e = U[4]/U[0];
//         Scalar drho_dx  = grad_U(0,0), drho_dy  = grad_U(0,1), drho_dz  = grad_U(0,2);
//         Scalar drhou_dx = grad_U(1,0), drhou_dy = grad_U(1,1), drhou_dz = grad_U(1,2);
//         Scalar drhov_dx = grad_U(2,0), drhov_dy = grad_U(2,1), drhov_dz = grad_U(2,2);
//         Scalar drhow_dx = grad_U(3,0), drhow_dy = grad_U(3,1), drhow_dz = grad_U(3,2);
//         Scalar drhoe_dx = grad_U(4,0), drhoe_dy = grad_U(4,1), drhoe_dz = grad_U(4,2);
//         Scalar gamma = 1.4;
//         Scalar u2 = u*u + v*v + w*w;
//         Scalar mu = m_mu, Pr = m_Pr;     // 以后改成接口传入
//         Scalar Const = gamma/Pr;   // 以后改成接口传入
//         Scalar scale_factor = mu / std::pow(rho, 4.0);
//         const auto& F_x = scale_factor*DenseMatrix<5,5>{0,0,0,0,0,(2.0/3.0)*rho*(4*drho_dx*rho_u - 2*drho_dy*rho_v - 2*drho_dz*rho_w + rho*(-2*drhou_dx + drhov_dy + drhow_dz)),-4.0/3.0*drho_dx*pow(rho, 2),(2.0/3.0)*drho_dy*pow(rho, 2),(2.0/3.0)*drho_dz*pow(rho, 2),0,rho*(2*drho_dx*rho_v + 2*drho_dy*rho_u - rho*(drhou_dy + drhov_dx)),-drho_dy*pow(rho, 2),-drho_dx*pow(rho, 2),0,0,rho*(2*drho_dx*rho_w + 2*drho_dz*rho_u - rho*(drhou_dz + drhow_dx)),-drho_dz*pow(rho, 2),0,-drho_dx*pow(rho, 2),0,-Const*drhoe_dx*pow(rho, 2) - drho_dx*pow(rho_u, 2)*(3*Const - 4) - 3*drho_dx*(Const - 1)*(pow(rho_v, 2) + pow(rho_w, 2)) - 2*rho*(-Const*drho_dx*rho_e + rho_v*(drhou_dy - drhov_dx*(Const - 1)) + rho_w*(drhou_dz - drhow_dx*(Const - 1))) + (1.0/3.0)*rho_u*(3*drho_dy*rho_v + 3*drho_dz*rho_w + 2*rho*(drhou_dx*(3*Const - 4) + 2*drhov_dy + 2*drhow_dz)),-1.0/3.0*rho*(-6*Const*drho_dx*rho_u + 8*drho_dx*rho_u + drho_dy*rho_v + drho_dz*rho_w + rho*(drhou_dx*(3*Const - 4) + 2*drhov_dy + 2*drhow_dz)),(1.0/3.0)*rho*(6*drho_dx*rho_v*(Const - 1) - drho_dy*rho_u + 3*rho*(drhou_dy - drhov_dx*(Const - 1))),(1.0/3.0)*rho*(6*drho_dx*rho_w*(Const - 1) - drho_dz*rho_u + 3*rho*(drhou_dz - drhow_dx*(Const - 1))),-Const*drho_dx*pow(rho, 2)};
//         const auto& F_y = scale_factor*DenseMatrix<5,5>{0,0,0,0,0,rho*(2*drho_dx*rho_v + 2*drho_dy*rho_u - rho*(drhou_dy + drhov_dx)),-drho_dy*pow(rho, 2),-drho_dx*pow(rho, 2),0,0,-2.0/3.0*rho*(2*drho_dx*rho_u - 4*drho_dy*rho_v + 2*drho_dz*rho_w - rho*(drhou_dx - 2*drhov_dy + drhow_dz)),(2.0/3.0)*drho_dx*pow(rho, 2),-4.0/3.0*drho_dy*pow(rho, 2),(2.0/3.0)*drho_dz*pow(rho, 2),0,rho*(2*drho_dy*rho_w + 2*drho_dz*rho_v - rho*(drhov_dz + drhow_dy)),0,-drho_dz*pow(rho, 2),-drho_dy*pow(rho, 2),0,-Const*drhoe_dy*pow(rho, 2) - drho_dy*pow(rho_v, 2)*(3*Const - 4) - 3*drho_dy*(Const - 1)*(pow(rho_u, 2) + pow(rho_w, 2)) + 2*rho*(Const*drho_dy*rho_e + rho_u*(drhou_dy*(Const - 1) - drhov_dx) - rho_w*(drhov_dz - drhow_dy*(Const - 1))) + (1.0/3.0)*rho_v*(3*drho_dx*rho_u + 3*drho_dz*rho_w + 2*rho*(2*drhou_dx + drhov_dy*(3*Const - 4) + 2*drhow_dz)),-1.0/3.0*rho*(drho_dx*rho_v - 6*drho_dy*rho_u*(Const - 1) + 3*rho*(drhou_dy*(Const - 1) - drhov_dx)),-1.0/3.0*rho*(-6*Const*drho_dy*rho_v + drho_dx*rho_u + 8*drho_dy*rho_v + drho_dz*rho_w + rho*(2*drhou_dx + drhov_dy*(3*Const - 4) + 2*drhow_dz)),(1.0/3.0)*rho*(6*drho_dy*rho_w*(Const - 1) - drho_dz*rho_v + 3*rho*(drhov_dz - drhow_dy*(Const - 1))),-Const*drho_dy*pow(rho, 2)};
//         const auto& F_z = scale_factor*DenseMatrix<5,5>{0,0,0,0,0,rho*(2*drho_dx*rho_w + 2*drho_dz*rho_u - rho*(drhou_dz + drhow_dx)),-drho_dz*pow(rho, 2),0,-drho_dx*pow(rho, 2),0,rho*(2*drho_dy*rho_w + 2*drho_dz*rho_v - rho*(drhov_dz + drhow_dy)),0,-drho_dz*pow(rho, 2),-drho_dy*pow(rho, 2),0,-2.0/3.0*rho*(2*drho_dx*rho_u + 2*drho_dy*rho_v - 4*drho_dz*rho_w - rho*(drhou_dx + drhov_dy - 2*drhow_dz)),(2.0/3.0)*drho_dx*pow(rho, 2),(2.0/3.0)*drho_dy*pow(rho, 2),-4.0/3.0*drho_dz*pow(rho, 2),0,-Const*drhoe_dz*pow(rho, 2) + drho_dx*rho_u*rho_w + drho_dy*rho_v*rho_w - 3*drho_dz*pow(rho_u, 2)*(Const - 1) - 3*drho_dz*pow(rho_v, 2)*(Const - 1) - drho_dz*pow(rho_w, 2)*(3*Const - 4) + (2.0/3.0)*rho*(3*Const*drho_dz*rho_e + 3*Const*drhov_dz*rho_v + 3*Const*drhow_dz*rho_w + 2*drhou_dx*rho_w + 2*drhov_dy*rho_w - 3*drhov_dz*rho_v - 3*drhow_dy*rho_v - 4*drhow_dz*rho_w + 3*rho_u*(drhou_dz*(Const - 1) - drhow_dx)),-1.0/3.0*rho*(drho_dx*rho_w - 6*drho_dz*rho_u*(Const - 1) + 3*rho*(drhou_dz*(Const - 1) - drhow_dx)),-1.0/3.0*rho*(drho_dy*rho_w - 6*drho_dz*rho_v*(Const - 1) + 3*rho*(drhov_dz*(Const - 1) - drhow_dy)),-1.0/3.0*rho*(drho_dx*rho_u + drho_dy*rho_v - 2*drho_dz*rho_w*(3*Const - 4) + rho*(2*drhou_dx + 2*drhov_dy + drhow_dz*(3*Const - 4))),-Const*drho_dz*pow(rho, 2)};
//         return {F_x,F_y,F_z};
//     }
// public:
//     vector3f transform_to_cell(const CompTriangleFace& face, 
//                               const vector2f& uv, uInt side) const 
//     {
//         const auto& nc = face.m_natural_coords[side];
//         return nc[0]*(1-uv[0]-uv[1]) + nc[1]*uv[0] + nc[2]*uv[1];
//     }

//     Scalar compute_max_wave_speed(const DenseMatrix<5,1>& U_L,
//                                  const DenseMatrix<5,1>& U_R) const 
//     {
//         const Scalar a_L = compute_sound_speed(U_L);
//         const Scalar a_R = compute_sound_speed(U_R);
//         const Scalar rho_L = std::max(U_L[0],1e-1);
//         const Scalar rho_R = std::max(U_R[0],1e-1);
//         const vector3f vel_L{U_L[1]/rho_L, U_L[2]/rho_L, U_L[3]/rho_L};
//         const vector3f vel_R{U_R[1]/rho_R, U_R[2]/rho_R, U_R[3]/rho_R};
//         // debug(vector4f{vec_length(vel_L),vec_length(vel_R),a_L,a_R});
//         return std::max(vec_length(vel_L) + a_L, vec_length(vel_R) + a_R)*0.0;
//         // return 4;
//     }

//     Scalar compute_sound_speed(const DenseMatrix<5,1>& U) const {
//         const Scalar gamma = 1.4;
//         const Scalar rho = std::max(U[0],1e-1);
        
//         const Scalar p = (gamma-1)*(U[4] - 0.5*(U[1]*U[1]+U[2]*U[2]+U[3]*U[3])/rho);
//         // debug(vector2f{rho,p});
//         return std::sqrt(std::max(gamma*p/rho, static_cast<Scalar>(1e-5)));
//     }
// public:
//     void assemble(const ComputingMesh& mesh, 
//                  const LongVector<5*N>& old_solution,
//                  const Scalar curr_time,
//                  BlockSparseMatrix<5*N,5*N>& sparse_mat,
//                  LongVector<5*N>& sparse_rhs) {
//         // int F(U) \cdot ( grad\phi_j  or  mathbf{n}\phi_j )
//         // F(U) = V(U)U = V(U) sum_i C_i\phi_i = sum_i V(U)C_i \phi_i
//         // F(U) \cdot grad\phi_j = sum_i (Fx*gx+Fy*gy+Fz*gz) C_i \phi_i
//         // int F(U) \cdot grad\phi_j = sum_i int (Fx*gx+Fy*gy+Fz*gz)\phi_i C_i 
//         // int (Fx*gx+Fy*gy+Fz*gz)\phi_i 为一个 5*5 矩阵
//         // sum_i (5*5矩阵)_{i,j} C_i，这里的(i,j)是选择两个基函数，i试探函数，j测试函数 


//         // 预计算体积积分点数据
//         constexpr uInt num_vol_points = QuadC::num_points;
//         std::array<std::array<Scalar,3>, num_vol_points> vol_points;
//         std::array<Scalar, num_vol_points> vol_weights;
//         std::array<std::array<Scalar, Basis::NumBasis>, num_vol_points> vol_basis;
//         std::array<std::array<vector3f, Basis::NumBasis>, num_vol_points> vol_grads;
//         for(uInt g=0; g<num_vol_points; ++g) {
//             vol_points[g] = QuadC::points[g];
//             vol_weights[g] = QuadC::weights[g];
//             vol_basis[g] = Basis::eval_all(vol_points[g][0],vol_points[g][1],vol_points[g][2]);
//             vol_grads[g] = Basis::grad_all(vol_points[g][0],vol_points[g][1],vol_points[g][2]);
//         }
        

//         #pragma omp parallel for schedule(dynamic)
//         for(uInt cid=0;cid<mesh.m_cells.size();cid++){
//             // 存储的模式为：
//             // 每个单元有100个自由度（系数）
//             // [单元1,100自由度][单元2,100自由度]......
//             // 其中，5个物理量，各有20个系数
//             // [5个自由度，系数1][5个自由度，系数2][5个自由度，系数3]....
//             const auto& cell = mesh.m_cells[cid];
//             const auto& coef = old_solution[cid];

//             // 或许可以拿到外面？  不确定编译器会不会重复分配内存
//             DenseMatrix<5*N,5*N> cell_matrix;
            
//             const auto& J = cell.get_invJacMat();

//             // 体积分部分
//             for(uInt g=0; g<num_vol_points; ++g) {
//                 std::array<DenseMatrix<3,1>, Basis::NumBasis> J_grad_phi_k;
//                 for(uInt k=0; k<Basis::NumBasis; ++k) {
//                     const auto& grad_phi_j = DenseMatrix<3,1>(vol_grads[g][k]);
//                     J_grad_phi_k[k] = J.multiply(grad_phi_j);
//                     // debug(J_grad_phi_k[k].transpose());
//                 }

//                 DenseMatrix<5,1> U = DenseMatrix<5,1>::Zeros();
//                 DenseMatrix<5,3> grad_U = DenseMatrix<5,3>::Zeros();
//                 for(uInt bid=0; bid<Basis::NumBasis; ++bid) {
//                     const Scalar phi = vol_basis[g][bid];
//                     for(uInt k=0; k<5; ++k){
//                         U[k] += phi * coef[5*bid + k];
//                         grad_U(k,0) += J_grad_phi_k[bid][0] * coef[5*bid + k];
//                         grad_U(k,1) += J_grad_phi_k[bid][1] * coef[5*bid + k];
//                         grad_U(k,2) += J_grad_phi_k[bid][2] * coef[5*bid + k];
//                     }
//                 }


//                 // 三组 J(F_{x,y,z},U)，每一个 5x5
//                 const auto JFU = assemble_jacobian(U,grad_U);
//                 // 已验证：F(U) 与 JFU dot U 完全一致（打印出来的部分肉眼检测）
//                 // const auto FU = assemble_FU(U);
//                 // debug(JFU[0].multiply(U));
//                 // debug(JFU[1].multiply(U));
//                 // debug(JFU[2].multiply(U));
//                 // debug(FU);
//                 const Scalar jac_weight = vol_weights[g] * cell.compute_jacobian_det();

//                 for(uInt i=0; i<Basis::NumBasis; ++i) {
//                     // phi_i，计算 
//                     // ( J(F,U) sum_i c_i phi_i ) grad_phi_j
//                     // sum_i phi_i (grad_phi_j phi_J(F_,U) c_i) 
//                     // c_i 的系数为 phi_i(grad_phi_j J(F,U)) 
//                     const Scalar phi_i = vol_basis[g][i];
//                     for(uInt j=0; j<Basis::NumBasis; ++j) {
//                         // const auto& J = cell.get_invJacMat();
//                         // const auto& grad_phi_j = J.multiply(DenseMatrix<3,1>(vol_grads[g][j]));
//                         // sum_{x,y,z}  J(F_x,U) phi_x_j
//                         DenseMatrix<5,5> flux = JFU[0]*J_grad_phi_k[j][0] + JFU[1]*J_grad_phi_k[j][1] + JFU[2]*J_grad_phi_k[j][2];
//                         // debug(FU.multiply(grad_phi_j));
//                         // debug(flux.multiply(U));
//                         // const auto e = FU.multiply(grad_phi_j)-flux.multiply(U);
//                         // if(e.dot(e)>1e-20) debug(e.dot(e));
//                         flux *= phi_i * jac_weight;

//                         MatrixView<5*N,5*N,5,5> block(cell_matrix,5*j,5*i);
//                         block -= flux;
//                     }
//                 }
//             }
//             // debug(cell_matrix.multiply(coef));
//             #pragma omp critical
//             sparse_mat.add_block(cid, cid, (-1.0)*cell_matrix);
//         }


//         #pragma omp parallel for schedule(dynamic)
//         for(uInt fid=0; fid<mesh.m_faces.size(); ++fid) {
//             const auto& face = mesh.m_faces[fid];
//             const auto& cells = face.m_neighbor_cells;
            
            
//             if(cells[1] == uInt(-1)) {
//                 // continue;
//                 // debug("11111");
//                 const auto bc = mesh.m_boundaryTypes[fid];

//                 // 内部单元
//                 const auto& coef = old_solution[cells[0]];
//                 DenseMatrix<5*N,5*N> face_matrix;
//                 DenseMatrix<5*N,1> face_rhs;

//                 for(uInt g=0; g<QuadF::num_points; ++g) {
//                     // continue;
//                     // 转换到单元自然坐标
//                     const auto& xi = transform_to_cell(face, QuadF::points[g], 0);
//                     const auto& basis = Basis::eval_all(xi[0], xi[1], xi[2]);
//                     const auto& grad_basis = Basis::grad_all(xi[0], xi[1], xi[2]);
//                     const Scalar jac_weight = QuadF::weights[g] * face.compute_jacobian_det();
//                     // 预计算所有的grad_phi_i/j
//                     std::array<DenseMatrix<3,1>, Basis::NumBasis> J_grad_phi_k;
//                     const auto& J = mesh.m_cells[cells[0]].get_invJacMat();
//                     for(uInt k=0; k<Basis::NumBasis; ++k) {
//                         const auto& grad_phi_j = DenseMatrix<3,1>(grad_basis[k]);
//                         J_grad_phi_k[k] = J.multiply(grad_phi_j);
//                     }
//                     DenseMatrix<5,1> U_inner;
//                     DenseMatrix<5,3> grad_U_inner = DenseMatrix<5,3>::Zeros();
//                     DenseMatrix<5,3> grad_U_ghost = DenseMatrix<5,3>::Zeros();
//                     for(uInt bid=0; bid<Basis::NumBasis; ++bid) {
//                         for(uInt k=0; k<5; ++k){
//                             U_inner[k] += basis[bid] * coef[5*bid + k];
//                             grad_U_inner(k,0) += J_grad_phi_k[bid][0] * coef[5*bid + k];
//                             grad_U_inner(k,1) += J_grad_phi_k[bid][1] * coef[5*bid + k];
//                             grad_U_inner(k,2) += J_grad_phi_k[bid][2] * coef[5*bid + k];
//                             // grad_U_ghost(k,0) += J_grad_phi_k[bid][0] * coef[5*bid + k];
//                             // grad_U_ghost(k,1) += J_grad_phi_k[bid][1] * coef[5*bid + k];
//                         }
//                     }

//                     // // 外部用类似于ghost的方法去做，
//                     // auto U_ghost = (bc==BoundaryType::Dirichlet)?
//                     //                 2*0.0-U_inner:   //Dirichlet     下面Neumann
//                     //                 U_inner + 2*0.0*vec_dot(vector3f{0,0,0},face.m_normal);

//                     // 根据边界类型计算通量贡献
//                     DenseMatrix<5UL, 5UL> dF;
//                     DenseMatrix<5UL, 1UL> drhs;
//                     if (bc==BoundaryType::Dirichlet) {
//                         // 使用与内部面完全相同
//                         const auto& U_D = DenseMatrix<5,1>({rho_Xi(mesh.m_cells[cells[0]],xi,curr_time),
//                                                             rhou_Xi(mesh.m_cells[cells[0]],xi,curr_time),
//                                                             rhov_Xi(mesh.m_cells[cells[0]],xi,curr_time),
//                                                             rhow_Xi(mesh.m_cells[cells[0]],xi,curr_time),
//                                                             rhoe_Xi(mesh.m_cells[cells[0]],xi,curr_time)});
//                         // const auto& U_D = DenseMatrix<5,1>({0,0,0,0,0});

//                         auto U_ghost = U_D + (U_D - U_inner);
//                         U_ghost = 0.5*(U_inner+U_ghost);
//                         grad_U_ghost = grad_U_inner;
//                         // (J(F,U) sum_i c_i phi_i) dot n phi_j
//                         // sum_i  phi_i (phi_i J(F_{xyz},U) c_i) phi_j
//                         auto F_inner = assemble_jacobian(U_inner,grad_U_inner);
//                         auto F_ghost = assemble_jacobian(U_ghost,grad_U_ghost);
                        
//                         DenseMatrix<5,5> Fn_inner, Fn_ghost;
//                         for(int d=0; d<3; ++d){
//                             Fn_inner += F_inner[d] * face.m_normal[d];
//                             Fn_ghost += F_ghost[d] * face.m_normal[d];
//                         }
                        
//                         const Scalar lambda = compute_max_wave_speed(U_inner, U_ghost);
//                         // debug(lambda);
//                         // dF = 0.5*(Fn_inner + Fn_ghost + lambda*DenseMatrix<5,5>::Identity());
//                         // drhs = -0.5*lambda*U_ghost;
//                         dF = 0.5*(Fn_inner + lambda*DenseMatrix<5,5>::Identity());
//                         drhs = 0.5*(Fn_ghost.multiply(U_ghost) - lambda*U_ghost);
//                     }
//                     else if(bc==BoundaryType::Pseudo3DZ) {
//                         // ComputingMesh::BType::Neumann
//                         auto U_ghost = U_inner * DenseMatrix<5,1>({1,1,1,-1,1});
//                         // U_ghost = 0.5*(U_inner+U_ghost);
                        
//                         grad_U_ghost = grad_U_inner;
//                         // const auto& U_D = DenseMatrix<5,1>({rho_Xi(mesh.m_cells[cells[0]],xi,curr_time),
//                                                             // rhou_Xi(mesh.m_cells[cells[0]],xi,curr_time),
//                                                             // rhov_Xi(mesh.m_cells[cells[0]],xi,curr_time),
//                                                             // rhow_Xi(mesh.m_cells[cells[0]],xi,curr_time),
//                                                             // rhoe_Xi(mesh.m_cells[cells[0]],xi,curr_time)});
//                         // const auto& U_D = DenseMatrix<5,1>({0,0,0,0,0});

//                         // const auto& U_ghost = U_D;// + (U_D - U_inner);
//                         auto F_inner = assemble_jacobian(U_inner,grad_U_inner);
//                         auto F_ghost = assemble_jacobian(U_ghost,grad_U_ghost);
                        
//                         DenseMatrix<5,5> Fn_inner, Fn_ghost;
//                         for(int d=0; d<3; ++d){
//                             Fn_inner += F_inner[d] * face.m_normal[d];
//                             Fn_ghost += F_ghost[d] * face.m_normal[d];
//                         }
                        
//                         const Scalar lambda = compute_max_wave_speed(U_inner, U_ghost);
//                         // debug(lambda);
//                         // dF = 0.5*(Fn_inner + Fn_ghost + lambda*DenseMatrix<5,5>::Identity());
//                         // drhs = -0.5*lambda*U_ghost;
//                         dF = 0.5*(Fn_inner + lambda*DenseMatrix<5,5>::Identity());
//                         drhs = 0.5*(Fn_ghost.multiply(U_ghost) - lambda*U_ghost);
//                         // // 直接使用内部通量，好像，应该，不需要跳了
//                         // auto F = assemble_jacobian(U_inner);
//                         // DenseMatrix<5,5> Fn;
//                         // for(int d=0; d<3; ++d)
//                         //     Fn += F[d] * face.m_normal[d];
                        
//                         // // Deepseek说我需要加一个对称
//                         // dF = 0.5*(Fn + Fn.transpose());
//                     }

//                     // 组装到单元矩阵
//                     for(uInt i=0; i<Basis::NumBasis; ++i){
//                         for(uInt j=0; j<Basis::NumBasis; ++j){
//                             auto contrib = dF * (basis[i] * basis[j]) * jac_weight;
//                             MatrixView<5*N,5*N,5,5> block(face_matrix,5*j,5*i);
//                             block += contrib;
                            
//                         }
//                     }
//                     for(uInt j=0; j<Basis::NumBasis; ++j){
//                         auto contrib = drhs * (basis[j]) * jac_weight;
//                         MatrixView<5*N,1,5,1> block(face_rhs,5*j,0);
//                         block += contrib;
//                     }
//                 }
//                 // debug("22222");
//                 // debug(face_matrix);

//                 #pragma omp critical
//                 {
//                 sparse_mat.add_block(cells[0], cells[0], (-1.0)*face_matrix);
//                 sparse_rhs[cells[0]] -= (-1.0)*face_rhs;
//                 // debug(vector2u{cells[0], cells[0]});
//                 }
//             }
//             else{

                

//                 // 获取左右单元系数
//                 const auto& coef_L = old_solution[cells[0]];
//                 const auto& coef_R = old_solution[cells[1]];

//                 // 左右两个面的分块矩阵
//                 DenseMatrix<5*N,5*N> face_matrix_LL, face_matrix_LR;
//                 DenseMatrix<5*N,5*N> face_matrix_RL, face_matrix_RR;

//                 for(uInt g=0; g<QuadF::num_points; ++g) {

//                     // 转换到左右单元自然坐标
//                     const auto& xi_L = transform_to_cell(face, QuadF::points[g], 0);
//                     const auto& xi_R = transform_to_cell(face, QuadF::points[g], 1);
//                     const auto& basis_L = Basis::eval_all(xi_L[0], xi_L[1], xi_L[2]);
//                     const auto& basis_R = Basis::eval_all(xi_R[0], xi_R[1], xi_R[2]);
//                     const auto& grad_basis_L = Basis::grad_all(xi_L[0], xi_L[1], xi_L[2]);
//                     const auto& grad_basis_R = Basis::grad_all(xi_R[0], xi_R[1], xi_R[2]);
//                     const Scalar jac_weight = QuadF::weights[g] * face.compute_jacobian_det();

//                     // 预计算所有的grad_phi_i/j
//                     std::array<DenseMatrix<3,1>, Basis::NumBasis> J_grad_phi_k_L,J_grad_phi_k_R;
//                     const auto& J_L = mesh.m_cells[cells[0]].get_invJacMat();
//                     const auto& J_R = mesh.m_cells[cells[1]].get_invJacMat();
//                     for(uInt k=0; k<Basis::NumBasis; ++k) {
//                         const auto& grad_phi_j_L = DenseMatrix<3,1>(grad_basis_L[k]);
//                         const auto& grad_phi_j_R = DenseMatrix<3,1>(grad_basis_R[k]);
//                         J_grad_phi_k_L[k] = J_L.multiply(grad_phi_j_L);
//                         J_grad_phi_k_R[k] = J_R.multiply(grad_phi_j_R);
//                     }

//                     DenseMatrix<5,1> U_L, U_R;
//                     DenseMatrix<5,3> grad_U_L = DenseMatrix<5,3>::Zeros();
//                     DenseMatrix<5,3> grad_U_R = DenseMatrix<5,3>::Zeros();
//                     for(uInt bid=0; bid<Basis::NumBasis; ++bid) {
//                         for(uInt k=0; k<5; ++k) {
//                             U_L[k] += basis_L[bid] * coef_L[5*bid + k];
//                             U_R[k] += basis_R[bid] * coef_R[5*bid + k];
//                             grad_U_L(k,0) += J_grad_phi_k_L[bid][0] * coef_L[5*bid + k];
//                             grad_U_L(k,1) += J_grad_phi_k_L[bid][1] * coef_L[5*bid + k];
//                             grad_U_L(k,2) += J_grad_phi_k_L[bid][2] * coef_L[5*bid + k];
//                             grad_U_R(k,0) += J_grad_phi_k_R[bid][0] * coef_R[5*bid + k];
//                             grad_U_R(k,1) += J_grad_phi_k_R[bid][1] * coef_R[5*bid + k];
//                             grad_U_R(k,2) += J_grad_phi_k_R[bid][2] * coef_R[5*bid + k];
//                         }
//                     }

//                     // 计算通量雅可比
//                     auto F_L = assemble_jacobian(U_L,grad_U_L);
//                     auto F_R = assemble_jacobian(U_R,grad_U_R);

//                     // 计算法向通量雅可比
//                     DenseMatrix<5,5> Fn_L, Fn_R;
//                     for(int d=0; d<3; ++d) {
//                         Fn_L += F_L[d] * face.m_normal[d];
//                         Fn_R += F_R[d] * face.m_normal[d];
//                     }
//                     // debug(Fn_L);

//                     // Lax-Friedrichs参数
//                     const Scalar lambda = compute_max_wave_speed(U_L, U_R);
//                     //  debug(lambda);
//                     // 通量雅可比贡献
//                     const DenseMatrix<5,5> dF_L = 0.5*(Fn_L + lambda*DenseMatrix<5,5>::Identity());
//                     const DenseMatrix<5,5> dF_R = 0.5*(Fn_R - lambda*DenseMatrix<5,5>::Identity());
//                     // debug(dF_L);
//                     // 组装到左右单元矩阵
//                     for(uInt i=0; i<Basis::NumBasis; ++i) {
//                         for(uInt j=0; j<Basis::NumBasis; ++j) {
//                             const Scalar phi_iL = basis_L[i];
//                             const Scalar phi_jL = basis_L[j];
//                             const Scalar phi_iR = basis_R[i];
//                             const Scalar phi_jR = basis_R[j];

//                             // auto contrib_L = dF_L * (phi_iL * phi_jL) * jac_weight;
//                             // auto contrib_R = dF_R * (phi_iR * phi_jR) * jac_weight;

//                             MatrixView<5*N,5*N,5,5> block_LL(face_matrix_LL, 5*j, 5*i);
//                             MatrixView<5*N,5*N,5,5> block_LR(face_matrix_LR, 5*j, 5*i);
//                             MatrixView<5*N,5*N,5,5> block_RL(face_matrix_RL, 5*j, 5*i);
//                             MatrixView<5*N,5*N,5,5> block_RR(face_matrix_RR, 5*j, 5*i);
//                             // debug(contrib_L);
//                             block_LL += dF_L * (phi_iL * phi_jL) * jac_weight;
//                             block_LR += dF_R * (phi_iR * phi_jL) * jac_weight;
//                             block_RL += dF_L * (phi_iL * phi_jR) * jac_weight;
//                             block_RR += dF_R * (phi_iR * phi_jR) * jac_weight;
//                         }
//                     }
//                 }
//                 // 这四个矩阵。。。。。终于。。。。。。
//                 #pragma omp critical
//                 {   
//                     sparse_mat.add_block(cells[0], cells[0], (-1.0)*face_matrix_LL);
//                     sparse_mat.add_block(cells[0], cells[1], (-1.0)*face_matrix_LR);
//                     sparse_mat.add_block(cells[1], cells[0], (-1.0)*(-1.)*face_matrix_RL); 
//                     sparse_mat.add_block(cells[1], cells[1], (-1.0)*(-1.)*face_matrix_RR);
//                     // debug(vector2u{cells[0], cells[1]});
//                 }
//             }
//         }

        
//     }

// };


