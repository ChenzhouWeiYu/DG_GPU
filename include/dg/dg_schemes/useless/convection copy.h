#pragma once
#include "Type.h"
#include "DG_Basis.h"
#include "../OptimizedMesh/OptimizedMesh.h"
#include "../Matrix/Matrix.h"

template<uInt Order=3>
class ImplicitConvection {
    using BlockMat = DenseMatrix<5,5>;
    using Basis = DGBasisEvaluator<Order>;
    using QuadC = GaussLegendreTet::Degree5Points15;
    using QuadF = GaussLegendreTri::Degree4Points7;

    static constexpr uInt N = Basis::NumBasis;

    // struct PrecomputedData {
    //     std::array<Scalar, QuadC::num_points> weights;
    //     std::array<std::array<Scalar,3>, QuadC::num_points> points;
    //     std::array<std::array<Scalar, Basis::NumBasis>, QuadC::num_points> basis_values;
    //     std::array<std::array<vector3f, Basis::NumBasis>, QuadC::num_points> basis_grads;
        
    //     PrecomputedData() {
    //         for(uInt g=0; g<QuadC::num_points; ++g){
    //             const auto& p = QuadC::points[g];
    //             basis_values[g] = Basis::eval_all(p[0], p[1], p[2]);
    //             basis_grads[g] = Basis::grad_all(p[0], p[1], p[2]);
    //             weights[g] = QuadC::weights[g];
    //         }
    //     }
    // };

    // static const PrecomputedData precomputed; // 静态预计算的，基函数at积分点



    std::array<DenseMatrix<5,5>,3> assemble_jacobian(const DenseMatrix<5,1>& U) {
        Scalar rho = U[0];
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
                        v*(-gamma*e+(gamma-1)*u2),-(gamma-1)*u*v,gamma*e-0.5*(gamma-1)*(2*v*u+u2),-(gamma-1)*v*w,gamma*v};
        F[2] = {0,0,0,1,0,  
                        -u*w,w,0,u,0,
                        -v*w,0,w,v,0,
                        -w*w+0.5*(gamma-1)*u2,-(gamma-1)*u,-(gamma-1)*v,2*w-(gamma-1)*w,gamma-1,
                        w*(-gamma*e+(gamma-1)*u2),-(gamma-1)*u*w,-(gamma-1)*v*w,gamma*e-0.5*(gamma-1)*(2*w*w+u2),gamma*w};
        return F;
    }
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
        const vector3f vel_L{U_L[1]/U_L[0], U_L[2]/U_L[0], U_L[3]/U_L[0]};
        const vector3f vel_R{U_R[1]/U_R[0], U_R[2]/U_R[0], U_R[3]/U_R[0]};
        // debug(vector4f{vec_length(vel_L),vec_length(vel_R),a_L,a_R});
        return std::max(vec_length(vel_L) + a_L, vec_length(vel_R) + a_R);
    }

    Scalar compute_sound_speed(const DenseMatrix<5,1>& U) const {
        const Scalar gamma = 1.4;
        const Scalar rho = U[0];
        
        const Scalar p = (gamma-1)*(U[4] - 0.5*(U[1]*U[1]+U[2]*U[2]+U[3]*U[3])/rho);
        // debug(vector2f{rho,p});
        return std::sqrt(gamma*std::max(p,1e-47)/rho);
    }
public:
    void assemble(const ComputingMesh& mesh, 
                 const LongVector<5*N>& old_solution,
                 BlockSparseMatrix<5*N,5*N>& sparse_mat) {
        // int F(U) \cdot ( grad\phi_j  or  mathbf{n}\phi_j )
        // F(U) = V(U)U = V(U) sum_i C_i\phi_i = sum_i V(U)C_i \phi_i
        // F(U) \cdot grad\phi_j = sum_i (Fx*gx+Fy*gy+Fz*gz) C_i \phi_i
        // int F(U) \cdot grad\phi_j = sum_i int (Fx*gx+Fy*gy+Fz*gz)\phi_i C_i 
        // int (Fx*gx+Fy*gy+Fz*gz)\phi_i 为一个 5*5 矩阵
        // sum_i (5*5矩阵)_{i,j} C_i，这里的(i,j)是选择两个基函数，i试探函数，j测试函数 

        // 预计算体积积分点数据
        constexpr uInt num_vol_points = QuadC::num_points;
        std::array<std::array<Scalar,3>, num_vol_points> vol_points;
        std::array<Scalar, num_vol_points> vol_weights;
        std::array<std::array<Scalar, Basis::NumBasis>, num_vol_points> vol_basis;
        std::array<std::array<vector3f, Basis::NumBasis>, num_vol_points> vol_grads;
        for(uInt g=0; g<num_vol_points; ++g) {
            vol_points[g] = QuadC::points[g];
            vol_weights[g] = QuadC::weights[g];
            vol_basis[g] = Basis::eval_all(vol_points[g][0],vol_points[g][1],vol_points[g][2]);
            vol_grads[g] = Basis::grad_all(vol_points[g][0],vol_points[g][1],vol_points[g][2]);
        }

        #pragma omp parallel for schedule(dynamic)
        for(uInt cid=0;cid<mesh.m_cells.size();cid++){
            const auto& cell = mesh.m_cells[cid];
            // 存储的模式为：
            // 每个单元有100个自由度（系数）
            // [单元1,100自由度][单元2,100自由度]......
            // 其中，5个物理量，各有20个系数
            // [5个自由度，系数1][5个自由度，系数2][5个自由度，系数3]....
            const auto& coef = old_solution[cid];
            DenseMatrix<5*N,5*N> cell_matrix;

            // 体积分部分
            for(uInt g=0; g<num_vol_points; ++g) {
                DenseMatrix<5,1> U = DenseMatrix<5,1>::Zeros();
                for(uInt bid=0; bid<Basis::NumBasis; ++bid) {
                    const Scalar phi = vol_basis[g][bid];
                    for(uInt k=0; k<5; ++k) 
                        U[k] += phi * coef[5*bid + k];
                }

                const auto F = assemble_jacobian(U);
                const Scalar jac_weight = vol_weights[g] * cell.compute_jacobian_det();

                for(uInt i=0; i<Basis::NumBasis; ++i) {
                    const Scalar phi_i = vol_basis[g][i];
                    for(uInt j=0; j<Basis::NumBasis; ++j) {
                        const vector3f& grad_phi_j = vol_grads[g][j];
                        DenseMatrix<5,5> flux = F[0]*grad_phi_j[0] + F[1]*grad_phi_j[1] + F[2]*grad_phi_j[2];
                        flux *= phi_i * jac_weight;

                        MatrixView<5*N,5*N,5,5> block(cell_matrix,5*i,5*j);
                        block += flux;
                    }
                }
            }
            
#pragma omp critical
            sparse_mat.add_block(cid, cid, cell_matrix);
        }


        #pragma omp parallel for schedule(dynamic)
        for(uInt fid=0; fid<mesh.m_faces.size(); ++fid) {
            const auto& face = mesh.m_faces[fid];
            const auto& cells = face.m_neighbor_cells;
            
            
            if(cells[1] == uInt(-1)) {
                // debug("11111");
                const auto bc = mesh.m_boundary[fid];

                // 内部单元
                const auto& coef = old_solution[cells[0]];
                DenseMatrix<5*N,5*N> face_matrix;

                for(uInt g=0; g<QuadF::num_points; ++g) {
                    // 转换到单元自然坐标
                    auto xi = transform_to_cell(face, QuadF::points[g], 0);
                    auto basis = Basis::eval_all(xi[0], xi[1], xi[2]);

                    // 重建内部状态
                    DenseMatrix<5,1> U_inner;
                    for(uInt bid=0; bid<Basis::NumBasis; ++bid) {
                        for(uInt k=0; k<5; ++k)
                            U_inner[k] += basis[bid] * coef[5*bid + k];
                    }

                    // 外部用类似于ghost的方法去做，
                    auto U_ghost = (bc==1)?
                                    2*0.0-U_inner:   //Dirichlet     下面Neumann
                                    U_inner + 2*0.0*vec_dot(vector3f{0,0,0},face.m_normal);

                    // 根据边界类型计算通量贡献
                    DenseMatrix<5UL, 5UL> dF;
                    if (bc==1) {
                        // 使用与内部面完全相同
                        auto F_inner = assemble_jacobian(U_inner);
                        auto F_ghost = assemble_jacobian(U_ghost);
                        
                        DenseMatrix<5,5> Fn_inner, Fn_ghost;
                        for(int d=0; d<3; ++d){
                            Fn_inner += F_inner[d] * face.m_normal[d];
                            Fn_ghost += F_ghost[d] * face.m_normal[d];
                        }
                        
                        const Scalar lambda = compute_max_wave_speed(U_inner, U_ghost);
                        // debug(lambda);
                        dF = 0.5*(Fn_inner + Fn_ghost + lambda*DenseMatrix<5,5>::Identity());
                    }
                    else {
                        // ComputingMesh::BType::Neumann
                        // 直接使用内部通量，好像，应该，不需要跳了
                        auto F = assemble_jacobian(U_inner);
                        DenseMatrix<5,5> Fn;
                        for(int d=0; d<3; ++d)
                            Fn += F[d] * face.m_normal[d];
                        
                        // Deepseek说我需要加一个对称
                        dF = 0.5*(Fn + Fn.transpose());
                    }

                    // 组装到单元矩阵
                    for(uInt i=0; i<Basis::NumBasis; ++i){
                        for(uInt j=0; j<Basis::NumBasis; ++j){
                            auto contrib = dF * (basis[i] * basis[j]) * QuadF::weights[g];
                            MatrixView<5*N,5*N,5,5> block(face_matrix,5*i,5*j);
                            block += contrib;
                        }
                    }
                }
                // debug("22222");
                // debug(face_matrix);

#pragma omp critical
                {
                sparse_mat.add_block(cells[0], cells[0], face_matrix);
                // debug(vector2u{cells[0], cells[0]});
                }
            }
            else{

                

                // 获取左右单元系数
                const auto& coef_L = old_solution[cells[0]];
                const auto& coef_R = old_solution[cells[1]];

                // 左右两个面的分块矩阵
                DenseMatrix<5*N,5*N> face_matrix_L, face_matrix_R;

                for(uInt g=0; g<QuadF::num_points; ++g) {
                    const auto& uv = QuadF::points[g];
                    const Scalar weight = QuadF::weights[g] * face.compute_jacobian_det();

                    // 转换到左右单元自然坐标
                    auto xi_L = transform_to_cell(face, uv, 0);
                    auto xi_R = transform_to_cell(face, uv, 1);

                    // 计算左右单元基函数值
                    auto basis_L = Basis::eval_all(xi_L[0], xi_L[1], xi_L[2]);
                    auto basis_R = Basis::eval_all(xi_R[0], xi_R[1], xi_R[2]);

                    // 重建左右状态
                    DenseMatrix<5,1> U_L, U_R;
                    for(uInt bid=0; bid<Basis::NumBasis; ++bid) {
                        for(uInt k=0; k<5; ++k) {
                            U_L[k] += basis_L[bid] * coef_L[5*bid + k];
                            U_R[k] += basis_R[bid] * coef_R[5*bid + k];
                        }
                    }

                    // 计算通量雅可比
                    auto F_L = assemble_jacobian(U_L);
                    auto F_R = assemble_jacobian(U_R);

                    // 计算法向通量雅可比
                    DenseMatrix<5,5> Fn_L, Fn_R;
                    for(int d=0; d<3; ++d) {
                        Fn_L += F_L[d] * face.m_normal[d];
                        Fn_R += F_R[d] * face.m_normal[d];
                    }
                    // debug(Fn_L);

                    // Lax-Friedrichs参数
                    const Scalar lambda = compute_max_wave_speed(U_L, U_R);
    //  debug(lambda);
                    // 通量雅可比贡献
                    const DenseMatrix<5,5> dF_L = 0.5*(Fn_L + lambda*DenseMatrix<5,5>::Identity());
                    const DenseMatrix<5,5> dF_R = 0.5*(Fn_R - lambda*DenseMatrix<5,5>::Identity());
                    // debug(dF_L);
                    // 组装到左右单元矩阵
                    for(uInt i=0; i<Basis::NumBasis; ++i) {
                        for(uInt j=0; j<Basis::NumBasis; ++j) {
                            const Scalar phi_iL = basis_L[i];
                            const Scalar phi_jL = basis_L[j];
                            const Scalar phi_iR = basis_R[i];
                            const Scalar phi_jR = basis_R[j];

                            auto contrib_L = dF_L * (phi_iL * phi_jL) * weight;
                            auto contrib_R = dF_R * (phi_iR * phi_jR) * weight;

                            MatrixView<5*N,5*N,5,5> block_L(face_matrix_L, 5*i, 5*j);
                            MatrixView<5*N,5*N,5,5> block_R(face_matrix_R, 5*i, 5*j);
                            // debug(contrib_L);
                            block_L += contrib_L;
                            block_R += contrib_R;
                        }
                    }
                }

#pragma omp critical
                {   
                    sparse_mat.add_block(cells[0], cells[0], face_matrix_L);
                    sparse_mat.add_block(cells[1], cells[1], face_matrix_R);
                    sparse_mat.add_block(cells[0], cells[1], (-1.)*face_matrix_L); 
                    sparse_mat.add_block(cells[1], cells[0], (-1.)*face_matrix_R);
                    // debug(vector2u{cells[0], cells[1]});
                }
            }
        }
        // debug("11");
        // debug(mesh.m_cells.size());
        // #pragma omp parallel for schedule(dynamic)
        // for(uInt cid=0;cid<mesh.m_cells.size();cid++){
        //     const auto& cell = mesh.m_cells[cid];
        //     // 存储的模式为：
        //     // 每个单元有100个自由度（系数）
        //     // [单元1,100自由度][单元2,100自由度]......
        //     // 其中，5个物理量，各有20个系数
        //     // [5个自由度，系数1][5个自由度，系数2][5个自由度，系数3]....
        //     const auto& coef = old_solution[cid];
        //     DenseMatrix<5*N,5*N> cell_matrix;
        //     // debug("222");

        //     std::array<DenseMatrix<5,1>, QuadC::num_points> U_at_gauss;
        //     for(uInt g=0; g<QuadC::num_points; ++g) {
        //         for(uInt bid=0; bid<Basis::NumBasis; ++bid) {
        //             const Scalar phi = precomputed.basis_values[g][bid];
        //             U_at_gauss[g][0] += phi * coef[5*bid + 0];
        //             U_at_gauss[g][1] += phi * coef[5*bid + 1];
        //             U_at_gauss[g][2] += phi * coef[5*bid + 2];
        //             U_at_gauss[g][3] += phi * coef[5*bid + 3];
        //             U_at_gauss[g][4] += phi * coef[5*bid + 4];
        //         }
        //     }


        //     for(uInt g=0; g<QuadC::num_points; ++g) {
        //         const auto& weight = precomputed.weights[g];
        //         const auto& basis = precomputed.basis_values[g];
        //         const auto& grads = precomputed.basis_grads[g];
        //         const auto& U = U_at_gauss[g];

        //         const auto F = assemble_jacobian(U);
        //         const Scalar jac_weight = weight * cell.m_volume;

        //         for(uInt i=0; i<Basis::NumBasis; ++i) {
        //             const Scalar phi_i = basis[i];
        //             const uInt row_offset = 5*i;

        //             for(uInt j=0; j<Basis::NumBasis; ++j) {
        //                 const vector3f& grad_phi_j = grads[j];
        //                 const uInt col_offset = 5*j;

        //                 // 计算通量贡献：Σ(Fx*∂xφj + Fy*∂yφj + Fz*∂zφj)
        //                 DenseMatrix<5,5> flux_contribution = 
        //                     F[0]*grad_phi_j[0] + 
        //                     F[1]*grad_phi_j[1] + 
        //                     F[2]*grad_phi_j[2];
                        
        //                 flux_contribution *= phi_i * jac_weight;

        //                 // 累加到单元矩阵块
        //                 MatrixView<5*N,5*N,5,5> block(cell_matrix,5*i,5*j);

        //                 block += flux_contribution;
        //             }
        //         }
        //     }

            // #pragma omp critical
            // sparse_mat.add_block(cid, cid, cell_matrix);


            // for(uInt i=0;i<Basis::NumBasis;i++){
            //     for(uInt j=0;j<Basis::NumBasis;j++){
            //         // 在这里组装一个 5*5 的矩阵，涉及到Gauss积分
            //         // debug("3333");
            //         MatrixView<5*N,5*N,5,5> submat(cell_matrix,5*i,5*j);
            //         for(uInt gauss=0;gauss<QuadC::num_points;gauss++){
            //             // debug("44444");
            //             const auto& points = QuadC::points;
            //             const auto& basis = Basis::eval_all(points[gauss][0],points[gauss][1],points[gauss][2]);
            //             const auto& grads = Basis::grad_all(points[gauss][0],points[gauss][1],points[gauss][2]);
            //             // 一大堆基函数的值都拿到了
            //             DenseMatrix<5,1> U;
            //             // debug("555555");
            //             for(uInt bid=0;bid<basis.size();bid++){
            //                 // [5个自由度，系数1][5个自由度，系数2][5个自由度，系数3]....
            //                 // debug(bid);
            //                 // debug(basis[bid]);
            //                 // debug(coef[5*bid+0]);
            //                 // debug(U[0]);
            //                 U[0] += basis[bid]*coef[5*bid+0];
            //                 U[1] += basis[bid]*coef[5*bid+1];
            //                 U[2] += basis[bid]*coef[5*bid+2];
            //                 U[3] += basis[bid]*coef[5*bid+3];
            //                 U[4] += basis[bid]*coef[5*bid+4];
            //             }
            //             // debug("6666666");
            //             const auto F = assemble_jacobian(U);
            //             const auto& R = (F[0]*grads[j][0]+F[1]*grads[j][1]+F[2]*grads[j][2])*basis[i];
            //             // debug(R);
            //             // debug("77777777");

            //             // 采用的高斯求解公式，其权重精确表示为 weight*Volume，
            //             submat += R * QuadC::weights[gauss]*cell.m_volume;  
            //             // debug("888888888");
            //             // debug(R);
            //         }
            //     }
            // }
            // // debug(vector2u{cid,cid});
            // // 存储的模式为：5个物理量，各有20个系数
            // // [5个自由度，系数1][5个自由度，系数2][5个自由度，系数3]....
            // // 矩阵sparse_mat的模式：row,vol,Mat<100,100>，分块存储
            // sparse_mat.add_block(cid,cid,cell_matrix);
        // }
        // 缺少面积分，但暂时忽略，先处理体积分



        
    }

};


// template<uInt Order>
// const typename ImplicitConvection<Order>::PrecomputedData 
// ImplicitConvection<Order>::precomputed;










template<uInt Order=3>
class ExplicitDiffusion {
    using BlockMat = DenseMatrix<5,5>;
    using Basis = DGBasisEvaluator<Order>;
    using QuadC = GaussLegendreTet::Degree10Points74;
    using QuadF = GaussLegendreTri::Degree10Points24;

    static constexpr uInt N = Basis::NumBasis;
    DenseMatrix<5,3> assemble_FU(const DenseMatrix<5,1>& U){
        Scalar rho = U[0];
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

    std::array<std::array<DenseMatrix<5,5>,3>,3> assemble_jacobian(const DenseMatrix<5,1>& U) const{
        Scalar rho = U[0];
        Scalar u = U[1]/U[0], v = U[2]/U[0], w = U[3]/U[0], e = U[4]/U[0];
        Scalar gamma = 1.4;
        Scalar u2 = u*u + v*v + w*w;
        Scalar mu = 1, Pr = 1;
        Scalar Const = gamma/Pr; // 
        const auto& G_fx_ux = mu/rho * DenseMatrix<5,5>({0, 0, 0, 0, 0, -4.0/3.0*u, 4.0/3.0, 0, 0, 0, -v, 0, 1, 0, 0, -w, 0, 0, 1, 0, -Const*e + std::pow(u, 2)*(Const - 4.0/3.0) + (Const - 1)*(std::pow(v, 2) + std::pow(w, 2)), u*(4.0/3.0 - Const), v*(1 - Const), w*(1 - Const), Const});
        const auto& G_fx_uy = mu/rho * DenseMatrix<5,5>({0, 0, 0, 0, 0, (2.0/3.0)*v, 0, -2.0/3.0, 0, 0, -u, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1.0/3.0*u*v, v, -2.0/3.0*u, 0, 0});
        const auto& G_fx_uz = mu/rho * DenseMatrix<5,5>({0, 0, 0, 0, 0, (2.0/3.0)*w, 0, 0, -2.0/3.0, 0, 0, 0, 0, 0, 0, -u, 1, 0, 0, 0, -1.0/3.0*u*w, w, 0, -2.0/3.0*u, 0});
        const auto& G_fy_ux = mu/rho * DenseMatrix<5,5>({0, 0, 0, 0, 0, -v, 0, 1, 0, 0, (2.0/3.0)*u, -2.0/3.0, 0, 0, 0, 0, 0, 0, 0, 0, -1.0/3.0*u*v, -2.0/3.0*v, u, 0, 0});
        const auto& G_fy_uy = mu/rho * DenseMatrix<5,5>({0, 0, 0, 0, 0, -u, 1, 0, 0, 0, -4.0/3.0*v, 0, 4.0/3.0, 0, 0, -w, 0, 0, 1, 0, -Const*e + Const*std::pow(v, 2) + Const*std::pow(w, 2) + std::pow(u, 2)*(Const - 1) - 4.0/3.0*std::pow(v, 2) - std::pow(w, 2), u*(1 - Const), v*(4.0/3.0 - Const), w*(1 - Const), Const});
        const auto& G_fy_uz = mu/rho * DenseMatrix<5,5>({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (2.0/3.0)*w, 0, 0, -2.0/3.0, 0, -v, 0, 1, 0, 0, -1.0/3.0*v*w, 0, w, -2.0/3.0*v, 0});
        const auto& G_fz_ux = mu/rho * DenseMatrix<5,5>({0, 0, 0, 0, 0, -w, 0, 0, 1, 0, 0, 0, 0, 0, 0, (2.0/3.0)*u, -2.0/3.0, 0, 0, 0, -1.0/3.0*u*w, -2.0/3.0*w, 0, u, 0});
        const auto& G_fz_uy = mu/rho * DenseMatrix<5,5>({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -w, 0, 0, 1, 0, (2.0/3.0)*v, 0, -2.0/3.0, 0, 0, -1.0/3.0*v*w, 0, -2.0/3.0*w, v, 0});
        const auto& G_fz_uz = mu/rho * DenseMatrix<5,5>({0, 0, 0, 0, 0, -u, 1, 0, 0, 0, -v, 0, 1, 0, 0, -4.0/3.0*w, 0, 0, 4.0/3.0, 0, -Const*e + Const*std::pow(v, 2) + Const*std::pow(w, 2) + std::pow(u, 2)*(Const - 1) - std::pow(v, 2) - 4.0/3.0*std::pow(w, 2), u*(1 - Const), v*(1 - Const), w*(4.0/3.0 - Const), Const});
        return {{G_fx_ux, G_fx_uy, G_fx_uz},  {G_fy_ux, G_fy_uy, G_fy_uz},  {G_fz_ux, G_fz_uy, G_fz_uz}};
    }
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
        const vector3f vel_L{U_L[1]/U_L[0], U_L[2]/U_L[0], U_L[3]/U_L[0]};
        const vector3f vel_R{U_R[1]/U_R[0], U_R[2]/U_R[0], U_R[3]/U_R[0]};
        // debug(vector4f{vec_length(vel_L),vec_length(vel_R),a_L,a_R});
        return std::max(vec_length(vel_L) + a_L, vec_length(vel_R) + a_R)*1;
        // return 4;
    }

    Scalar compute_sound_speed(const DenseMatrix<5,1>& U) const {
        const Scalar gamma = 1.4;
        const Scalar rho = U[0];
        
        const Scalar p = (gamma-1)*(U[4] - 0.5*(U[1]*U[1]+U[2]*U[2]+U[3]*U[3])/rho);
        // debug(vector2f{rho,p});
        return std::sqrt(gamma*std::max(p,1e-47)/rho);
    }
public:
    LongVector<5*N> eval(const ComputingMesh& mesh, 
                 const LongVector<5*N>& old_solution){
        constexpr uInt num_vol_points = QuadC::num_points;
        std::array<std::array<Scalar,3>, num_vol_points> vol_points;
        std::array<Scalar, num_vol_points> vol_weights;
        std::array<std::array<Scalar, Basis::NumBasis>, num_vol_points> vol_basis;
        std::array<std::array<vector3f, Basis::NumBasis>, num_vol_points> vol_grads;
        // Scalar val=0;



        #define is_debug
        #undef is_debug

        #ifdef is_debug
        uInt target = 2860;
        #endif
        
        


        for(uInt g=0; g<num_vol_points; ++g) {
            vol_points[g] = QuadC::points[g];
            vol_weights[g] = QuadC::weights[g];
            vol_basis[g] = Basis::eval_all(vol_points[g][0],vol_points[g][1],vol_points[g][2]);
            vol_grads[g] = Basis::grad_all(vol_points[g][0],vol_points[g][1],vol_points[g][2]);
            // val+=vol_weights[g];
        }
        // debug(val);

        
        LongVector<5*N> result(old_solution.size());
        #ifdef is_debug
        auto print_cell_rhow = [&](uInt cell_id, uInt type){
            #pragma omp critical
            if(cell_id == target){
                debug(vector2u{cell_id,type});
                debug(vector4f{result[cell_id](5*0+3,0),result[cell_id](5*1+3,0),result[cell_id](5*2+3,0),result[cell_id](5*3+3,0)});
            }
        };
        #endif

        #pragma omp parallel for schedule(dynamic)
        for(uInt cid=0;cid<mesh.m_cells.size();cid++){
            // print_cell_rhow(cid,0);
            std::array<Scalar,5> cumsum = {0,0,0,0,0};

            const auto& cell = mesh.m_cells[cid];
            const auto& coef = old_solution[cid];
            for(uInt g=0; g<num_vol_points; ++g) {
                DenseMatrix<5,1> U = DenseMatrix<5,1>::Zeros();
                for(uInt bid=0; bid<Basis::NumBasis; ++bid) {
                    const Scalar phi = vol_basis[g][bid];
                    for(uInt k=0; k<5; ++k) 
                        U[k] += phi * coef[5*bid + k];
                }
                const auto& FU = assemble_FU(U);
                // debug(FU);
                const Scalar jac_weight = vol_weights[g] * cell.compute_jacobian_det();
                for(uInt j=0; j<Basis::NumBasis; ++j) {
                    // [p1-p0,p2-p0,p3-p0]已经自带转置，只需求逆
                    const auto& J = inverse_3x3(DenseMatrix<3,3>(cell.compute_jacobian_mat()));
                    const auto& grad_phi_j = DenseMatrix<3,1>(vol_grads[g][j]);
                    const auto& flux = FU.multiply(J.multiply(grad_phi_j));
                    // if(j==Basis::NumBasis-1){
                    //     // debug(DenseMatrix<3,3>(cell.compute_jacobian_mat()));
                    //     // debug(J.multiply(grad_phi_j));
                    //     // debug(flux);
                    //     // debug(jac_weight);
                    //     // debug(flux.transpose() * jac_weight);
                    // }
                    #ifdef is_debug
                    if(cid == target){
                        cumsum[0] = 1;
                        cumsum[1+j] += -flux[3] * jac_weight;
                    }
                    #endif
                    // #pragma omp critical
                    // MatrixView<5*N,1,5,1>(result[cid],5*j,0) -= flux * jac_weight;
                    #pragma omp atomic update
                    result[cid](5*j+0,0) -= flux(0,0) * jac_weight;
                    #pragma omp atomic update
                    result[cid](5*j+1,0) -= flux(1,0) * jac_weight;
                    #pragma omp atomic update
                    result[cid](5*j+2,0) -= flux(2,0) * jac_weight;
                    #pragma omp atomic update
                    result[cid](5*j+3,0) -= flux(3,0) * jac_weight;
                    #pragma omp atomic update
                    result[cid](5*j+4,0) -= flux(4,0) * jac_weight;
                    
                }
                // print_cell_rhow(cid,1);
                
                // cumsum += 
            }
            #ifdef is_debug
            #pragma omp critical
            if(cid == target) debug(cumsum);
            #endif
            // debug(std::array<Scalar,5>{result[cid](5*(Basis::NumBasis-1)+0,0),result[cid](5*(Basis::NumBasis-1)+1,0),result[cid](5*(Basis::NumBasis-1)+2,0),result[cid](5*(Basis::NumBasis-1)+3,0),result[cid](5*(Basis::NumBasis-1)+4,0)});
        }

        #pragma omp parallel for schedule(dynamic)
        for(uInt fid=0; fid<mesh.m_faces.size(); ++fid) {
            std::array<Scalar,5> cumsum = {0,0,0,0,0};
            const auto& face = mesh.m_faces[fid];
            const auto& cells = face.m_neighbor_cells;
            if(cells[1] == uInt(-1)) {
                const auto bc = mesh.m_boundary[fid];
                const auto& coef = old_solution[cells[0]];
                for(uInt g=0; g<QuadF::num_points; ++g) {
                    const auto& xi = transform_to_cell(face, QuadF::points[g], 0);
                    const auto& basis = Basis::eval_all(xi[0], xi[1], xi[2]);
                    const Scalar jac_weight = QuadF::weights[g] * face.compute_jacobian_det();
                    
                    DenseMatrix<5,1> U_inner;
                    for(uInt bid=0; bid<Basis::NumBasis; ++bid) {
                        for(uInt k=0; k<5; ++k)
                            U_inner[k] += basis[bid] * coef[5*bid + k];
                    }
                    // auto U_ghost = (bc==1)?
                    //                 2*DenseMatrix<5,1>({1,1,0.5,0,2.5+0.625})-U_inner:   //Dirichlet     下面Neumann
                    //                 U_inner + 2*0.0*vec_dot(vector3f{0,0,0},face.m_normal);
                    
                    if (bc==1) {
                        const auto& U_D = DenseMatrix<5,1>({1,1,0.5,0,2.5+0.625});
                        const auto& U_ghost = U_D + (U_D - U_inner);
                        const auto& FU_inner = assemble_FU(U_inner);
                        const auto& FU_ghost = assemble_FU(U_ghost);
                        const auto& FUn_inner = FU_inner.multiply(DenseMatrix<3,1>(face.m_normal));
                        const auto& FUn_ghost = FU_ghost.multiply(DenseMatrix<3,1>(face.m_normal));
                        
                        
                        const Scalar lambda = compute_max_wave_speed(U_inner, U_ghost);
                        // debug(lambda);
                        const auto& LF_flux = 0.5*(FUn_inner + FUn_ghost + lambda*(U_inner-U_ghost));
                        for(uInt j=0; j<Basis::NumBasis; ++j) {
                            const Scalar phi_j = basis[j];
                            #ifdef is_debug
                            if(cells[0] == target){
                                cumsum[0] = 2;
                                cumsum[1+j] += LF_flux[3] * phi_j * jac_weight;
                            }
                            #endif
                            // #pragma omp critical
                            // MatrixView<5*N,1,5,1>(result[cells[0]],5*j,0) += LF_flux * phi_j * jac_weight;
                           
                            #pragma omp atomic update
                            result[cells[0]](5*j+0,0) += LF_flux(0,0) * phi_j * jac_weight;
                            #pragma omp atomic update
                            result[cells[0]](5*j+1,0) += LF_flux(1,0) * phi_j * jac_weight;
                            #pragma omp atomic update
                            result[cells[0]](5*j+2,0) += LF_flux(2,0) * phi_j * jac_weight;
                            #pragma omp atomic update
                            result[cells[0]](5*j+3,0) += LF_flux(3,0) * phi_j * jac_weight;
                            #pragma omp atomic update
                            result[cells[0]](5*j+4,0) += LF_flux(4,0) * phi_j * jac_weight;
                        }
                        // print_cell_rhow(cells[0],2);
                    }
                    else {
                        if(0){
                            // 把伪三维问题，视为是上下边界 U cdot n = 0 去做
                            const auto& Un_N = vec_dot(vector3f{0,0,0},face.m_normal);
                            const auto& U_ghost = U_inner + 2 * 0.0 * Un_N;
                            const auto& FU_inner = assemble_FU(U_inner);
                            const auto& FU_ghost = assemble_FU(U_ghost);
                            const auto& FUn_inner = FU_inner.multiply(DenseMatrix<3,1>(face.m_normal));
                            const auto& FUn_ghost = FU_ghost.multiply(DenseMatrix<3,1>(face.m_normal));
                            // 因为 0 Neumann，所以这里直接 U_ghost = U_inner，F(U_inner)=F(U_ghost)
                            const auto& LF_flux = FUn_inner;
                            for(uInt j=0; j<Basis::NumBasis; ++j) {
                                const Scalar phi_j = basis[j];
                                #ifdef is_debug
                                if(cells[0] == target){
                                    cumsum[0] = 3;
                                    cumsum[1+j] += LF_flux[3] * phi_j * jac_weight;
                                }
                                #endif
                                // #pragma omp critical
                                // MatrixView<5*N,1,5,1>(result[cells[0]],5*j,0) += LF_flux * phi_j * jac_weight;
                                
                                #pragma omp atomic update
                                result[cells[0]](5*j+0,0) += LF_flux(0,0) * phi_j * jac_weight;
                                #pragma omp atomic update
                                result[cells[0]](5*j+1,0) += LF_flux(1,0) * phi_j * jac_weight;
                                #pragma omp atomic update
                                result[cells[0]](5*j+2,0) += LF_flux(2,0) * phi_j * jac_weight;
                                #pragma omp atomic update
                                result[cells[0]](5*j+3,0) += LF_flux(3,0) * phi_j * jac_weight;
                                #pragma omp atomic update
                                result[cells[0]](5*j+4,0) += LF_flux(4,0) * phi_j * jac_weight;
                            }
                            // print_cell_rhow(cells[0],3);
                        }
                        else{
                            // 把伪三维问题，视为是上下边界 (rho w)_{ghost} = -(rho w)_{inner}
                            const auto& U_ghost = U_inner * DenseMatrix<5,1>({1,1,1,-1,1});
                            const auto& FU_inner = assemble_FU(U_inner);
                            const auto& FU_ghost = assemble_FU(U_ghost);
                            const auto& FUn_inner = FU_inner.multiply(DenseMatrix<3,1>(face.m_normal));
                            const auto& FUn_ghost = FU_ghost.multiply(DenseMatrix<3,1>(face.m_normal));
                            
                            const Scalar lambda = compute_max_wave_speed(U_inner, U_ghost);
                            // debug(lambda);
                            const auto& LF_flux = 0.5*(FUn_inner + FUn_ghost + lambda*(U_inner-U_ghost));
                            // if(cells[0] == target){
                            //     debug(face.m_normal);
                            //     // debug(std::array<Scalar,5>{FUn_inner[0],FUn_inner[1],FUn_inner[2],FUn_inner[3],FUn_inner[4]});
                            //     // debug(std::array<Scalar,5>{FUn_ghost[0],FUn_ghost[1],FUn_ghost[2],FUn_ghost[3],FUn_ghost[4]});
                            //     debug(std::array<Scalar,5>{LF_flux[3],FUn_inner[3],FUn_ghost[3],U_inner[3],U_ghost[3]});
                            // }
                            for(uInt j=0; j<Basis::NumBasis; ++j) {
                                const Scalar phi_j = basis[j];
                                #ifdef is_debug
                                #pragma omp critical
                                if(cells[0] == target){
                                    cumsum[0] = 3;
                                    cumsum[1+j] += LF_flux[3] * phi_j * jac_weight;
                                    // if(j==1)debug(vector4f{LF_flux[3] , phi_j , jac_weight, LF_flux[3] * phi_j * jac_weight});
                                }
                                #endif
                                // #pragma omp critical
                                // MatrixView<5*N,1,5,1>(result[cells[0]],5*j,0) += LF_flux * phi_j * jac_weight;
                                
                                #pragma omp atomic update
                                result[cells[0]](5*j+0,0) += LF_flux(0,0) * phi_j * jac_weight;
                                #pragma omp atomic update
                                result[cells[0]](5*j+1,0) += LF_flux(1,0) * phi_j * jac_weight;
                                #pragma omp atomic update
                                result[cells[0]](5*j+2,0) += LF_flux(2,0) * phi_j * jac_weight;
                                #pragma omp atomic update
                                result[cells[0]](5*j+3,0) += LF_flux(3,0) * phi_j * jac_weight;
                                #pragma omp atomic update
                                result[cells[0]](5*j+4,0) += LF_flux(4,0) * phi_j * jac_weight;
                                
                            }
                            // print_cell_rhow(cells[0],3);
                        }
                    }
                }
            }
            else{
                const auto& coef_L = old_solution[cells[0]];
                const auto& coef_R = old_solution[cells[1]];
                for(uInt g=0; g<QuadF::num_points; ++g) {
                    const auto& uv = QuadF::points[g];
                    const Scalar jac_weight = QuadF::weights[g] * face.compute_jacobian_det();

                    auto xi_L = transform_to_cell(face, uv, 0);
                    auto xi_R = transform_to_cell(face, uv, 1);
                    auto basis_L = Basis::eval_all(xi_L[0], xi_L[1], xi_L[2]);
                    auto basis_R = Basis::eval_all(xi_R[0], xi_R[1], xi_R[2]);

                    DenseMatrix<5,1> U_L, U_R;
                    for(uInt bid=0; bid<Basis::NumBasis; ++bid) {
                        for(uInt k=0; k<5; ++k) {
                            U_L[k] += basis_L[bid] * coef_L[5*bid + k];
                            U_R[k] += basis_R[bid] * coef_R[5*bid + k];
                        }
                    }

                    const auto& FU_L = assemble_FU(U_L);
                    const auto& FU_R = assemble_FU(U_R);
                    const auto& FUn_L = FU_L.multiply(DenseMatrix<3,1>(face.m_normal));
                    const auto& FUn_R = FU_R.multiply(DenseMatrix<3,1>(face.m_normal));

                    // Lax-Friedrichs参数
                    const Scalar lambda = compute_max_wave_speed(U_L, U_R);
                    // debug(fid);
                    // debug(vector4f{fid,lambda,U_L[0],U_R[0]});
                    const auto& LF_flux = 0.5*(FUn_L + FUn_R + lambda*(U_L-U_R));
                    // if(cells[0] == target || cells[1] == target){
                    //     debug(face.m_normal);
                    //     // debug(std::array<Scalar,5>{FUn_L[0],FUn_L[1],FUn_L[2],FUn_L[3],FUn_L[4]});
                    //     // debug(std::array<Scalar,5>{FUn_R[0],FUn_R[1],FUn_R[2],FUn_R[3],FUn_R[4]});
                    //     debug(std::array<Scalar,5>{LF_flux[3],FUn_L[3],FUn_R[3],U_L[3],U_R[3]});
                    // }
                    // const auto& dF_L = 0.5*(FUn_L + lambda*U_L);
                    // const auto& dF_R = 0.5*(FUn_R - lambda*U_R);
                    for(uInt j=0; j<Basis::NumBasis; ++j) {
                        const Scalar phi_jL = basis_L[j];
                        const Scalar phi_jR = basis_R[j];
                        // if(j==Basis::NumBasis-1){
                        //     // debug(LF_flux * phi_jL *  jac_weight);
                        // }
                        #ifdef is_debug
                        #pragma omp critical
                        if(cells[0] == target){
                                    cumsum[0] = 4;
                            cumsum[1+j] += LF_flux[3] * phi_jL *  jac_weight;
                                    // if(j==1)debug(vector4f{LF_flux[3] , phi_jL, jac_weight,LF_flux[3] * phi_jL * jac_weight});
                        }
                        #pragma omp critical
                        if(cells[1] == target){
                                    cumsum[0] = 5;
                            cumsum[1+j] += LF_flux[3] * -phi_jR *  jac_weight;
                                    // if(j==1)debug(vector3f{LF_flux[3] , -phi_jR , jac_weight});
                        }
                        #endif
                        // #pragma omp critical
                        // {
                        //     MatrixView<5*N,1,5,1>(result[cells[0]],5*j,0) += LF_flux * phi_jL *  jac_weight;
                        //     MatrixView<5*N,1,5,1>(result[cells[1]],5*j,0) -= LF_flux * phi_jR *  jac_weight;
                        // }
                        
                        #pragma omp atomic update
                        result[cells[0]](5*j+0,0) += LF_flux(0,0) * phi_jL * jac_weight;
                        #pragma omp atomic update
                        result[cells[0]](5*j+1,0) += LF_flux(1,0) * phi_jL * jac_weight;
                        #pragma omp atomic update
                        result[cells[0]](5*j+2,0) += LF_flux(2,0) * phi_jL * jac_weight;
                        #pragma omp atomic update
                        result[cells[0]](5*j+3,0) += LF_flux(3,0) * phi_jL * jac_weight;
                        #pragma omp atomic update
                        result[cells[0]](5*j+4,0) += LF_flux(4,0) * phi_jL * jac_weight;
                        #pragma omp atomic update
                        result[cells[1]](5*j+0,0) -= LF_flux(0,0) * phi_jR * jac_weight;
                        #pragma omp atomic update
                        result[cells[1]](5*j+1,0) -= LF_flux(1,0) * phi_jR * jac_weight;
                        #pragma omp atomic update
                        result[cells[1]](5*j+2,0) -= LF_flux(2,0) * phi_jR * jac_weight;
                        #pragma omp atomic update
                        result[cells[1]](5*j+3,0) -= LF_flux(3,0) * phi_jR * jac_weight;
                        #pragma omp atomic update
                        result[cells[1]](5*j+4,0) -= LF_flux(4,0) * phi_jR * jac_weight;
                        
                    }
                    // print_cell_rhow(cells[0],4);
                    // print_cell_rhow(cells[1],5);
                }
            }
            #ifdef is_debug
            #pragma omp critical
            if(cells[0] == target || cells[1] == target) debug(cumsum);
            #endif
        }
        // debug(result);
        // for(uInt cid = 0;cid<result.size();cid++){
        //     const auto& V = result[cid];
        //     if(std::abs(V[3])>1e-8 || std::abs(V[8])>1e-8 || 
        //         std::abs(V[13])>1e-8 || std::abs(V[18])>1e-8){
        //         debug(cid);
        //         debug(mesh.m_cells[cid].m_centroid);
        //         debug(V.transpose());
        //     }
        // }
        return result;
    }

};