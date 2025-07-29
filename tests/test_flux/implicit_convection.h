#pragma once
#include "base/type.h"
#include "mesh/mesh.h"
#include "matrix/matrix.h"
#include "dg/dg_basis/dg_basis.h"
#include "base/exact.h"


template<uInt Order=3, typename GaussQuadCell = GaussLegendreTet::Auto, typename GaussQuadFace = GaussLegendreTri::Auto>
class OldImplicitConvection {
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



public:
    std::array<DenseMatrix<5,5>,3> assemble_jacobian(const DenseMatrix<5,1>& U) {
        Scalar rho = std::max(U[0],1e-16);
        Scalar u = U[1]/U[0], v = U[2]/U[0], w = U[3]/U[0], e = U[4]/U[0];
        Scalar gamma = 1.4;
        Scalar u2 = u*u + v*v + w*w;
        std::array<DenseMatrix<5,5>,3> F;
        F[0] = {0,1,0,0,0,  
                        -u*u+0.5*(gamma-1)*u2, 2*u-(gamma-1)*u, -(gamma-1)*v,-(gamma-1)*w,gamma-1,
                        -u*v,v,u,0,0,
                        -u*w,w,0,u,0,
                        u*(-gamma*e+(gamma-1)*u2),gamma*e-0.5*(gamma-1)*(2*u*u+u2),-(gamma-1)*u*v,-(gamma-1)*u*w,gamma*u};
        F[1] = {0,0,1,0,0,  
                        -u*v,v,u,0,0,
                        -v*v+0.5*(gamma-1)*u2, -(gamma-1)*u, 2*v-(gamma-1)*v,-(gamma-1)*w,gamma-1,
                        -v*w,0,w,v,0,
                        v*(-gamma*e+(gamma-1)*u2),-(gamma-1)*u*v,gamma*e-0.5*(gamma-1)*(2*v*v+u2),-(gamma-1)*v*w,gamma*v};
        F[2] = {0,0,0,1,0,  
                        -u*w,w,0,u,0,
                        -v*w,0,w,v,0,
                        -w*w+0.5*(gamma-1)*u2,-(gamma-1)*u,-(gamma-1)*v,2*w-(gamma-1)*w,gamma-1,
                        w*(-gamma*e+(gamma-1)*u2),-(gamma-1)*u*w,-(gamma-1)*v*w,gamma*e-0.5*(gamma-1)*(2*w*w+u2),gamma*w};
        return F;
    }
    DenseMatrix<5,3> assemble_FU(const DenseMatrix<5,1>& U){
        Scalar rho = std::max(U[0],1e-16);
        Scalar u = U[1]/U[0], v = U[2]/U[0], w = U[3]/U[0], e = U[4]/U[0];
        Scalar gamma = 1.4;
        Scalar u2 = u*u + v*v + w*w;
        Scalar p = (gamma-1)*rho*(e-0.5*u2);
        return {rho*u,rho*v,rho*w,
                rho*u*u+p,rho*v*u  ,rho*w*u  ,
                rho*u*v  ,rho*v*v+p,rho*w*v  ,
                rho*u*w  ,rho*v*w  ,rho*w*w+p,
                u*(rho*e+p),v*(rho*e+p),w*(rho*e+p)};
    }
public:
    vector3f transform_to_cell(const CompTriangleFace& face, 
                              const vector2f& uv, uInt side) const 
    {
        const auto& nc = face.m_natural_coords[side];
        return nc[0]*(1-uv[0]-uv[1]) + nc[1]*uv[0] + nc[2]*uv[1];
    }

    Scalar compute_max_wave_speed(const DenseMatrix<5,1>& U_L,
                                 const DenseMatrix<5,1>& U_R) const 
    {
        const Scalar a_L = compute_sound_speed(U_L);
        const Scalar a_R = compute_sound_speed(U_R);
        const Scalar rho_L = std::max(U_L[0],1e-16);
        const Scalar rho_R = std::max(U_R[0],1e-16);
        const vector3f vel_L{U_L[1]/rho_L, U_L[2]/rho_L, U_L[3]/rho_L};
        const vector3f vel_R{U_R[1]/rho_R, U_R[2]/rho_R, U_R[3]/rho_R};
        // debug(vector4f{vec_length(vel_L),vec_length(vel_R),a_L,a_R});
        return std::max(vec_length(vel_L) + a_L, vec_length(vel_R) + a_R)*1.00;
        // return 4;
    }

    Scalar compute_sound_speed(const DenseMatrix<5,1>& U) const {
        const Scalar gamma = 1.4;
        const Scalar rho = std::max(U[0],1e-16);
        
        const Scalar p = (gamma-1)*(U[4] - 0.5*(U[1]*U[1]+U[2]*U[2]+U[3]*U[3])/rho);
        // debug(vector2f{rho,p});
        return std::sqrt(std::max(gamma*p/rho, static_cast<Scalar>(1e-5)));
    }
public:
    // void assemble(const ComputingMesh& mesh, 
    //              const LongVector<5*N>& old_solution,
    //              const Scalar curr_time,
    //              BlockSparseMatrix<5*N,5*N>& sparse_mat,
    //              LongVector<5*N>& sparse_rhs) {
    //     // int F(U) \cdot ( grad\phi_j  or  mathbf{n}\phi_j )
    //     // F(U) = V(U)U = V(U) sum_i C_i\phi_i = sum_i V(U)C_i \phi_i
    //     // F(U) \cdot grad\phi_j = sum_i (Fx*gx+Fy*gy+Fz*gz) C_i \phi_i
    //     // int F(U) \cdot grad\phi_j = sum_i int (Fx*gx+Fy*gy+Fz*gz)\phi_i C_i 
    //     // int (Fx*gx+Fy*gy+Fz*gz)\phi_i 为一个 5*5 矩阵
    //     // sum_i (5*5矩阵)_{i,j} C_i，这里的(i,j)是选择两个基函数，i试探函数，j测试函数 


    //     // 预计算体积积分点数据
    //     constexpr uInt num_vol_points = QuadC::num_points;
    //     std::array<std::array<Scalar,3>, num_vol_points> vol_points;
    //     std::array<Scalar, num_vol_points> vol_weights;
    //     std::array<std::array<Scalar, Basis::NumBasis>, num_vol_points> vol_basis;
    //     std::array<std::array<vector3f, Basis::NumBasis>, num_vol_points> vol_grads;
    //     for(uInt g=0; g<num_vol_points; ++g) {
    //         vol_points[g] = QuadC::points[g];
    //         vol_weights[g] = QuadC::weights[g];
    //         vol_basis[g] = Basis::eval_all(vol_points[g][0],vol_points[g][1],vol_points[g][2]);
    //         vol_grads[g] = Basis::grad_all(vol_points[g][0],vol_points[g][1],vol_points[g][2]);
    //     }

    //     #pragma omp parallel for schedule(dynamic)
    //     for(uInt cid=0;cid<mesh.m_cells.size();cid++){
    //         // 存储的模式为：
    //         // 每个单元有100个自由度（系数）
    //         // [单元1,100自由度][单元2,100自由度]......
    //         // 其中，5个物理量，各有20个系数
    //         // [5个自由度，系数1][5个自由度，系数2][5个自由度，系数3]....
    //         const auto& cell = mesh.m_cells[cid];
    //         const auto& coef = old_solution[cid];

    //         // 或许可以拿到外面？  不确定编译器会不会重复分配内存
    //         DenseMatrix<5*N,5*N> cell_matrix;

    //         // 体积分部分
    //         for(uInt g=0; g<num_vol_points; ++g) {
    //             DenseMatrix<5,1> U = DenseMatrix<5,1>::Zeros();
    //             for(uInt bid=0; bid<Basis::NumBasis; ++bid) {
    //                 const Scalar phi = vol_basis[g][bid];
    //                 for(uInt k=0; k<5; ++k) 
    //                     U[k] += phi * coef[5*bid + k];
    //             }

    //             // 三组 J(F_{x,y,z},U)，每一个 5x5
    //             const auto JFU = assemble_jacobian(U);
    //             // 已验证：F(U) 与 JFU dot U 完全一致（打印出来的部分肉眼检测）
    //             // const auto FU = assemble_FU(U);
    //             // debug(JFU[0].multiply(U));
    //             // debug(JFU[1].multiply(U));
    //             // debug(JFU[2].multiply(U));
    //             // debug(FU);
    //             const Scalar jac_weight = vol_weights[g] * cell.compute_jacobian_det();

    //             for(uInt i=0; i<Basis::NumBasis; ++i) {
    //                 // phi_i，计算 
    //                 // ( J(F,U) sum_i c_i phi_i ) grad_phi_j
    //                 // sum_i phi_i (grad_phi_j phi_J(F_,U) c_i) 
    //                 // c_i 的系数为 phi_i(grad_phi_j J(F,U)) 
    //                 const Scalar phi_i = vol_basis[g][i];
    //                 for(uInt j=0; j<Basis::NumBasis; ++j) {
    //                     const auto& J = inverse_3x3(DenseMatrix<3,3>(cell.compute_jacobian_mat()));
    //                     const auto& grad_phi_j = J.multiply(DenseMatrix<3,1>(vol_grads[g][j]));
    //                     // sum_{x,y,z}  J(F_x,U) phi_x_j
    //                     DenseMatrix<5,5> flux = JFU[0]*grad_phi_j[0] + JFU[1]*grad_phi_j[1] + JFU[2]*grad_phi_j[2];
    //                     // debug(FU.multiply(grad_phi_j));
    //                     // debug(flux.multiply(U));
    //                     // const auto e = FU.multiply(grad_phi_j)-flux.multiply(U);
    //                     // if(e.dot(e)>1e-20) debug(e.dot(e));
    //                     flux *= phi_i * jac_weight;

    //                     MatrixView<5*N,5*N,5,5> block(cell_matrix,5*j,5*i);
    //                     block -= flux;
    //                 }
    //             }
    //         }
    //         // debug(cell_matrix.multiply(coef));
    //         #pragma omp critical
    //         sparse_mat.add_block(cid, cid, cell_matrix);
    //     }


    //     #pragma omp parallel for schedule(dynamic)
    //     for(uInt fid=0; fid<mesh.m_faces.size(); ++fid) {
    //         const auto& face = mesh.m_faces[fid];
    //         const auto& cells = face.m_neighbor_cells;
            
            
    //         if(cells[1] == uInt(-1)) {
    //             // debug("11111");
    //             const auto bc = mesh.m_boundary[fid];

    //             // 内部单元
    //             const auto& coef = old_solution[cells[0]];
    //             DenseMatrix<5*N,5*N> face_matrix;
    //             DenseMatrix<5*N,1> face_rhs;

    //             for(uInt g=0; g<QuadF::num_points; ++g) {
    //                 // 转换到单元自然坐标
    //                 const auto& xi = transform_to_cell(face, QuadF::points[g], 0);
    //                 const auto& basis = Basis::eval_all(xi[0], xi[1], xi[2]);
    //                 const Scalar jac_weight = QuadF::weights[g] * face.compute_jacobian_det();
                    
    //                 // 重建内部状态
    //                 DenseMatrix<5,1> U_inner;
    //                 for(uInt bid=0; bid<Basis::NumBasis; ++bid) {
    //                     for(uInt k=0; k<5; ++k)
    //                         U_inner[k] += basis[bid] * coef[5*bid + k];
    //                 }

    //                 // // 外部用类似于ghost的方法去做，
    //                 // auto U_ghost = (bc==1)?
    //                 //                 2*0.0-U_inner:   //Dirichlet     下面Neumann
    //                 //                 U_inner + 2*0.0*vec_dot(vector3f{0,0,0},face.m_normal);

    //                 // 根据边界类型计算通量贡献
    //                 DenseMatrix<5UL, 5UL> dF;
    //                 DenseMatrix<5UL, 1UL> drhs;
    //                 if (bc==1) {
    //                     // 使用与内部面完全相同
    //                     const auto& U_D = DenseMatrix<5,1>({rho_Xi(mesh.m_cells[cells[0]],xi,curr_time),
    //                                                         rhou_Xi(mesh.m_cells[cells[0]],xi,curr_time),
    //                                                         rhov_Xi(mesh.m_cells[cells[0]],xi,curr_time),
    //                                                         rhow_Xi(mesh.m_cells[cells[0]],xi,curr_time),
    //                                                         rhoe_Xi(mesh.m_cells[cells[0]],xi,curr_time)});
    //                     // const auto& U_D = DenseMatrix<5,1>({0,0,0,0,0});

    //                     auto U_ghost = U_D + (U_D - U_inner);
    //                     U_ghost = 0.5*(U_inner+U_ghost);
    //                     // (J(F,U) sum_i c_i phi_i) dot n phi_j
    //                     // sum_i  phi_i (phi_i J(F_{xyz},U) c_i) phi_j
    //                     auto F_inner = assemble_jacobian(U_inner);
    //                     auto F_ghost = assemble_jacobian(U_ghost);
    //                     F_inner = F_ghost;
                        
    //                     DenseMatrix<5,5> Fn_inner, Fn_ghost;
    //                     for(int d=0; d<3; ++d){
    //                         Fn_inner += F_inner[d] * face.m_normal[d];
    //                         Fn_ghost += F_ghost[d] * face.m_normal[d];
    //                     }
                        
    //                     const Scalar lambda = compute_max_wave_speed(U_inner, U_ghost);
    //                     // debug(lambda);
    //                     // dF = 0.5*(Fn_inner + Fn_ghost + lambda*DenseMatrix<5,5>::Identity());
    //                     // drhs = -0.5*lambda*U_ghost;
    //                     dF = 0.5*(Fn_inner + lambda*DenseMatrix<5,5>::Identity());
    //                     drhs = 0.5*(Fn_ghost.multiply(U_ghost) - lambda*U_ghost);
    //                 }
    //                 else if(bc==3){
    //                     auto U_ghost = U_inner * DenseMatrix<5,1>({1,1,1,1,1});
    //                     auto F_inner = assemble_jacobian(0.5*(U_inner+U_ghost));
    //                     auto F_ghost = assemble_jacobian(0.5*(U_inner+U_ghost));
                        
    //                     DenseMatrix<5,5> Fn_inner, Fn_ghost;
    //                     for(int d=0; d<3; ++d){
    //                         Fn_inner += F_inner[d] * face.m_normal[d];
    //                         Fn_ghost += F_ghost[d] * face.m_normal[d];
    //                     }
                        
    //                     const Scalar lambda = compute_max_wave_speed(U_inner, U_ghost);
    //                     dF = 0.5*(Fn_inner + lambda*DenseMatrix<5,5>::Identity());
    //                     drhs = 0.5*(Fn_ghost.multiply(U_ghost) - lambda*U_ghost);
    //                 }
    //                 else {
    //                     // ComputingMesh::BType::Neumann
    //                     auto U_ghost = U_inner * DenseMatrix<5,1>({1,1,1,-1,1});
    //                     // U_ghost = 0.5*(U_inner+U_ghost);
    //                     // const auto& U_D = DenseMatrix<5,1>({rho_Xi(mesh.m_cells[cells[0]],xi,curr_time),
    //                                                         // rhou_Xi(mesh.m_cells[cells[0]],xi,curr_time),
    //                                                         // rhov_Xi(mesh.m_cells[cells[0]],xi,curr_time),
    //                                                         // rhow_Xi(mesh.m_cells[cells[0]],xi,curr_time),
    //                                                         // rhoe_Xi(mesh.m_cells[cells[0]],xi,curr_time)});
    //                     // const auto& U_D = DenseMatrix<5,1>({0,0,0,0,0});

    //                     // const auto& U_ghost = U_D;// + (U_D - U_inner);
    //                     auto F_inner = assemble_jacobian(0.5*(U_inner+U_ghost));
    //                     auto F_ghost = assemble_jacobian(0.5*(U_inner+U_ghost));
                        
    //                     DenseMatrix<5,5> Fn_inner, Fn_ghost;
    //                     for(int d=0; d<3; ++d){
    //                         Fn_inner += F_inner[d] * face.m_normal[d];
    //                         Fn_ghost += F_ghost[d] * face.m_normal[d];
    //                     }
                        
    //                     const Scalar lambda = compute_max_wave_speed(U_inner, U_ghost);
    //                     // debug(lambda);
    //                     // dF = 0.5*(Fn_inner + Fn_ghost + lambda*DenseMatrix<5,5>::Identity());
    //                     // drhs = -0.5*lambda*U_ghost;
    //                     dF = 0.5*(Fn_inner + lambda*DenseMatrix<5,5>::Identity());
    //                     drhs = 0.5*(Fn_ghost.multiply(U_ghost) - lambda*U_ghost);
    //                     // // 直接使用内部通量，好像，应该，不需要跳了
    //                     // auto F = assemble_jacobian(U_inner);
    //                     // DenseMatrix<5,5> Fn;
    //                     // for(int d=0; d<3; ++d)
    //                     //     Fn += F[d] * face.m_normal[d];
                        
    //                     // // Deepseek说我需要加一个对称
    //                     // dF = 0.5*(Fn + Fn.transpose());
    //                 }

    //                 // 组装到单元矩阵
    //                 for(uInt i=0; i<Basis::NumBasis; ++i){
    //                     for(uInt j=0; j<Basis::NumBasis; ++j){
    //                         auto contrib = dF * (basis[i] * basis[j]) * jac_weight;
    //                         MatrixView<5*N,5*N,5,5> block(face_matrix,5*j,5*i);
    //                         block += contrib;
                            
    //                     }
    //                 }
    //                 for(uInt j=0; j<Basis::NumBasis; ++j){
    //                     auto contrib = drhs * (basis[j]) * jac_weight;
    //                     MatrixView<5*N,1,5,1> block(face_rhs,5*j,0);
    //                     block += contrib;
    //                 }
    //             }
    //             // debug("22222");
    //             // debug(face_matrix);

    //             #pragma omp critical
    //             {
    //             sparse_mat.add_block(cells[0], cells[0], face_matrix);
    //             sparse_rhs[cells[0]] -= face_rhs;
    //             // debug(vector2u{cells[0], cells[0]});
    //             }
    //         }
    //         else{

                

    //             // 获取左右单元系数
    //             const auto& coef_L = old_solution[cells[0]];
    //             const auto& coef_R = old_solution[cells[1]];

    //             // 左右两个面的分块矩阵
    //             DenseMatrix<5*N,5*N> face_matrix_LL, face_matrix_LR;
    //             DenseMatrix<5*N,5*N> face_matrix_RL, face_matrix_RR;

    //             for(uInt g=0; g<QuadF::num_points; ++g) {
    //                 const auto& uv = QuadF::points[g];
    //                 const Scalar weight = QuadF::weights[g] * face.compute_jacobian_det();

    //                 // 转换到左右单元自然坐标
    //                 auto xi_L = transform_to_cell(face, uv, 0);
    //                 auto xi_R = transform_to_cell(face, uv, 1);

    //                 // 计算左右单元基函数值
    //                 auto basis_L = Basis::eval_all(xi_L[0], xi_L[1], xi_L[2]);
    //                 auto basis_R = Basis::eval_all(xi_R[0], xi_R[1], xi_R[2]);

    //                 // 重建左右状态
    //                 DenseMatrix<5,1> U_L, U_R;
    //                 for(uInt bid=0; bid<Basis::NumBasis; ++bid) {
    //                     for(uInt k=0; k<5; ++k) {
    //                         U_L[k] += basis_L[bid] * coef_L[5*bid + k];
    //                         U_R[k] += basis_R[bid] * coef_R[5*bid + k];
    //                     }
    //                 }

    //                 // 计算通量雅可比
    //                 auto F_L = assemble_jacobian(U_L);
    //                 auto F_R = assemble_jacobian(U_R);

    //                 // 计算法向通量雅可比
    //                 DenseMatrix<5,5> Fn_L, Fn_R;
    //                 for(int d=0; d<3; ++d) {
    //                     Fn_L += F_L[d] * face.m_normal[d];
    //                     Fn_R += F_R[d] * face.m_normal[d];
    //                 }
    //                 // debug(Fn_L);

    //                 // Lax-Friedrichs参数
    //                 const Scalar lambda = compute_max_wave_speed(U_L, U_R);
    //                 //  debug(lambda);
    //                 // 通量雅可比贡献
    //                 const DenseMatrix<5,5> dF_L = 0.5*(Fn_L + lambda*DenseMatrix<5,5>::Identity());
    //                 const DenseMatrix<5,5> dF_R = 0.5*(Fn_R - lambda*DenseMatrix<5,5>::Identity());
    //                 // debug(dF_L);
    //                 // 组装到左右单元矩阵
    //                 for(uInt i=0; i<Basis::NumBasis; ++i) {
    //                     for(uInt j=0; j<Basis::NumBasis; ++j) {
    //                         const Scalar phi_iL = basis_L[i];
    //                         const Scalar phi_jL = basis_L[j];
    //                         const Scalar phi_iR = basis_R[i];
    //                         const Scalar phi_jR = basis_R[j];

    //                         auto contrib_L = dF_L * (phi_iL * phi_jL) * weight;
    //                         auto contrib_R = dF_R * (phi_iR * phi_jR) * weight;

    //                         MatrixView<5*N,5*N,5,5> block_LL(face_matrix_LL, 5*j, 5*i);
    //                         MatrixView<5*N,5*N,5,5> block_LR(face_matrix_LR, 5*j, 5*i);
    //                         MatrixView<5*N,5*N,5,5> block_RL(face_matrix_RL, 5*j, 5*i);
    //                         MatrixView<5*N,5*N,5,5> block_RR(face_matrix_RR, 5*j, 5*i);
    //                         // debug(contrib_L);
    //                         block_LL += dF_L * (phi_iL * phi_jL) * weight;
    //                         block_LR += dF_R * (phi_iR * phi_jL) * weight;
    //                         block_RL += dF_L * (phi_iL * phi_jR) * weight;
    //                         block_RR += dF_R * (phi_iR * phi_jR) * weight;
    //                     }
    //                 }
    //             }
    //             // 这四个矩阵。。。。。终于。。。。。。
    //             #pragma omp critical
    //             {   
    //                 sparse_mat.add_block(cells[0], cells[0], face_matrix_LL);
    //                 sparse_mat.add_block(cells[0], cells[1], face_matrix_LR);
    //                 sparse_mat.add_block(cells[1], cells[0], (-1.)*face_matrix_RL); 
    //                 sparse_mat.add_block(cells[1], cells[1], (-1.)*face_matrix_RR);
    //                 // debug(vector2u{cells[0], cells[1]});
    //             }
    //         }
    //     }

        
    // }

};


