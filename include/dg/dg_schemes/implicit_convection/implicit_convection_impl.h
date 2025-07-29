#include "dg/dg_schemes/implicit_convection/implicit_convection.h"


template<uInt Order, typename Flux, typename GaussQuadCell, typename GaussQuadFace>
void ImplicitConvection<Order, Flux, GaussQuadCell, GaussQuadFace>::assemble_boundarys(const ComputingMesh& mesh, 
                 const LongVector<5*N>& old_solution,
                 const Scalar curr_time,
                 BlockSparseMatrix<5*N,5*N>& sparse_mat,
                 LongVector<5*N>& sparse_rhs){
    constexpr uInt num_face_points = QuadF::num_points;
    constexpr auto Qpoints = QuadF::get_points();
    constexpr auto Qweights = QuadF::get_weights();
    // 只计算边界的
    #pragma omp parallel for schedule(dynamic)
    for(uInt fid=0; fid<mesh.m_faces.size(); ++fid) {
        const auto& face = mesh.m_faces[fid];
        const auto& cells = face.m_neighbor_cells;

        // 跳过内部单元
        if(cells[1] != uInt(-1)) continue;
        // 只要边界单元

        const auto bc = mesh.m_boundaryTypes[fid];

        // 内部单元
        const auto& coef = old_solution[cells[0]];
        DenseMatrix<5*N,5*N> face_matrix;
        DenseMatrix<5*N,1> face_rhs;

        for(uInt g=0; g<num_face_points; ++g) {
            // 转换到单元自然坐标
            const auto& xi = transform_to_cell(face, Qpoints[g], 0);
            const auto& basis = Basis::eval_all(xi[0], xi[1], xi[2]);
            const Scalar jac_weight = Qweights[g] * face.compute_jacobian_det();
            
            // 重建内部状态
            DenseMatrix<5,1> U_inner;
            DenseMatrix<5,1> U_ghost;
            for(uInt bid=0; bid<Basis::NumBasis; ++bid) {
                for(uInt k=0; k<5; ++k)
                    U_inner[k] += basis[bid] * coef[5*bid + k];
            }

            // // 外部用类似于ghost的方法去做，
            // auto U_ghost = (bc==BoundaryType::Dirichlet)?
            //                 2*0.0-U_inner:   //Dirichlet     下面Neumann
            //                 U_inner + 2*0.0*vec_dot(vector3f{0,0,0},face.m_normal);

            // 根据边界类型计算通量贡献
            DenseMatrix<5UL, 5UL> dF;
            DenseMatrix<5UL, 1UL> drhs;
            if (bc==BoundaryType::Dirichlet) {
                // 使用与内部面完全相同
                const auto& U_D = U_Xi(mesh.m_cells[cells[0]],xi,curr_time);
                // const auto& U_D = DenseMatrix<5,1>({0,0,0,0,0});

                U_ghost = U_D + (U_D - U_inner);
                U_ghost = 0.5*(U_inner+U_ghost);
                // (J(F,U) sum_i c_i phi_i) dot n phi_j
                // sum_i  phi_i (phi_i J(F_{xyz},U) c_i) phi_j
                auto F_inner = Flux::computeJacobian(U_inner);
                auto F_ghost = Flux::computeJacobian(U_ghost);
                F_inner = F_ghost;
                
                DenseMatrix<5,5> Fn_inner, Fn_ghost;
                for(uInt d=0; d<3; ++d){
                    Fn_inner += F_inner[d] * face.m_normal[d];
                    Fn_ghost += F_ghost[d] * face.m_normal[d];
                }
                
                const Scalar lambda = Flux::computeWaveSpeed(U_inner, U_ghost);
                U_ghost = 0.5*(U_inner+U_ghost);
                // debug(lambda);
                // dF = 0.5*(Fn_inner + Fn_ghost + lambda*DenseMatrix<5,5>::Identity());
                // drhs = -0.5*lambda*U_ghost;
                dF = 0.5*(Fn_inner + lambda*DenseMatrix<5,5>::Identity());
                drhs = 0.5*(Fn_ghost.multiply(U_ghost) - lambda*U_ghost);
            }
            else if (bc==BoundaryType::WallTD) {
                // 使用与内部面完全相同
                const auto& U_D = U_Xi(mesh.m_cells[cells[0]],xi,curr_time);
                U_ghost = U_D + (U_D - U_inner);
                // 密度齐次Neumann
                U_ghost[0] = U_inner[0]; 
                Scalar cvT_inner = U_inner[4]-0.5*(U_inner[1]*U_inner[1]+U_inner[2]*U_inner[2]+U_inner[3]*U_inner[3])/U_inner[0];
                // 温度Dirichlet
                Scalar cvT_ghost = 2*300 - cvT_inner;
                U_ghost[4] = cvT_ghost + 0.5*(U_ghost[1]*U_ghost[1]+U_ghost[2]*U_ghost[2]+U_ghost[3]*U_ghost[3])/U_ghost[0];
                
                U_ghost = 0.5*(U_inner+U_ghost);
                auto F_inner = Flux::computeJacobian(U_inner);
                auto F_ghost = Flux::computeJacobian(U_ghost);
                F_inner = F_ghost;
                
                DenseMatrix<5,5> Fn_inner, Fn_ghost;
                for(uInt d=0; d<3; ++d){
                    Fn_inner += F_inner[d] * face.m_normal[d];
                    Fn_ghost += F_ghost[d] * face.m_normal[d];
                }
                
                const Scalar lambda = Flux::computeWaveSpeed(U_inner, U_ghost);
                U_ghost = 0.5*(U_inner+U_ghost);
                dF = 0.5*(Fn_inner + lambda*DenseMatrix<5,5>::Identity());
                drhs = 0.5*(Fn_ghost.multiply(U_ghost) - lambda*U_ghost);
            }
            else if (bc==BoundaryType::WallTN) {
                // 使用与内部面完全相同
                const auto& U_D = U_Xi(mesh.m_cells[cells[0]],xi,curr_time);
                U_ghost = U_D + (U_D - U_inner);
                U_ghost[0] = U_inner[0];
                Scalar cvT_inner = U_inner[4]-0.5*(U_inner[1]*U_inner[1]+U_inner[2]*U_inner[2]+U_inner[3]*U_inner[3])/U_inner[0];
                Scalar cvT_ghost = cvT_inner;
                U_ghost[4] = cvT_ghost + 0.5*(U_ghost[1]*U_ghost[1]+U_ghost[2]*U_ghost[2]+U_ghost[3]*U_ghost[3])/U_ghost[0];
                
                U_ghost = 0.5*(U_inner+U_ghost);
                auto F_inner = Flux::computeJacobian(U_inner);
                auto F_ghost = Flux::computeJacobian(U_ghost);
                F_inner = F_ghost;
                
                DenseMatrix<5,5> Fn_inner, Fn_ghost;
                for(uInt d=0; d<3; ++d){
                    Fn_inner += F_inner[d] * face.m_normal[d];
                    Fn_ghost += F_ghost[d] * face.m_normal[d];
                }
                
                const Scalar lambda = Flux::computeWaveSpeed(U_inner, U_ghost);
                U_ghost = 0.5*(U_inner+U_ghost);
                dF = 0.5*(Fn_inner + lambda*DenseMatrix<5,5>::Identity());
                drhs = 0.5*(Fn_ghost.multiply(U_ghost) - lambda*U_ghost);
            }
            else if(bc==BoundaryType::Pseudo3DZ){
                // ComputingMesh::BType::Neumann
                U_ghost = U_inner * DenseMatrix<5,1>({1,1,1,-1,1});
                // U_ghost = 0.5*(U_inner+U_ghost);
                // const auto& U_D = DenseMatrix<5,1>({rho_Xi(mesh.m_cells[cells[0]],xi,curr_time),
                                                    // rhou_Xi(mesh.m_cells[cells[0]],xi,curr_time),
                                                    // rhov_Xi(mesh.m_cells[cells[0]],xi,curr_time),
                                                    // rhow_Xi(mesh.m_cells[cells[0]],xi,curr_time),
                                                    // rhoe_Xi(mesh.m_cells[cells[0]],xi,curr_time)});
                // const auto& U_D = DenseMatrix<5,1>({0,0,0,0,0});

                // const auto& U_ghost = U_D;// + (U_D - U_inner);
                auto F_inner = Flux::computeJacobian(0.5*(U_inner+U_ghost));
                auto F_ghost = Flux::computeJacobian(0.5*(U_inner+U_ghost));
                
                DenseMatrix<5,5> Fn_inner, Fn_ghost;
                for(uInt d=0; d<3; ++d){
                    Fn_inner += F_inner[d] * face.m_normal[d];
                    Fn_ghost += F_ghost[d] * face.m_normal[d];
                }
                
                const Scalar lambda = Flux::computeWaveSpeed(U_inner, U_ghost);
                U_ghost = 0.5*(U_inner+U_ghost);
                // debug(lambda);
                // dF = 0.5*(Fn_inner + Fn_ghost + lambda*DenseMatrix<5,5>::Identity());
                // drhs = -0.5*lambda*U_ghost;
                dF = 0.5*(Fn_inner + lambda*DenseMatrix<5,5>::Identity());
                drhs = 0.5*(Fn_ghost.multiply(U_ghost) - lambda*U_ghost);
                // // 直接使用内部通量，好像，应该，不需要跳了
                // auto F = Flux::computeJacobian(U_inner);
                // DenseMatrix<5,5> Fn;
                // for(uInt d=0; d<3; ++d)
                //     Fn += F[d] * face.m_normal[d];
                
                // // Deepseek说我需要加一个对称
                // dF = 0.5*(Fn + Fn.transpose());
            }
            else if(bc==BoundaryType::Pseudo3DY){
                U_ghost = U_inner * DenseMatrix<5,1>({1,1,-1,1,1});
                auto F_inner = Flux::computeJacobian(0.5*(U_inner+U_ghost));
                auto F_ghost = Flux::computeJacobian(0.5*(U_inner+U_ghost));
                
                DenseMatrix<5,5> Fn_inner, Fn_ghost;
                for(uInt d=0; d<3; ++d){
                    Fn_inner += F_inner[d] * face.m_normal[d];
                    Fn_ghost += F_ghost[d] * face.m_normal[d];
                }
                
                const Scalar lambda = Flux::computeWaveSpeed(U_inner, U_ghost);
                U_ghost = 0.5*(U_inner+U_ghost);
                dF = 0.5*(Fn_inner + lambda*DenseMatrix<5,5>::Identity());
                drhs = 0.5*(Fn_ghost.multiply(U_ghost) - lambda*U_ghost);
            }
            else if(bc==BoundaryType::Pseudo3DX){
                U_ghost = U_inner * DenseMatrix<5,1>({1,-1,1,1,1});
                auto F_inner = Flux::computeJacobian(0.5*(U_inner+U_ghost));
                auto F_ghost = Flux::computeJacobian(0.5*(U_inner+U_ghost));
                
                DenseMatrix<5,5> Fn_inner, Fn_ghost;
                for(uInt d=0; d<3; ++d){
                    Fn_inner += F_inner[d] * face.m_normal[d];
                    Fn_ghost += F_ghost[d] * face.m_normal[d];
                }
                
                const Scalar lambda = Flux::computeWaveSpeed(U_inner, U_ghost);
                U_ghost = 0.5*(U_inner+U_ghost);
                dF = 0.5*(Fn_inner + lambda*DenseMatrix<5,5>::Identity());
                drhs = 0.5*(Fn_ghost.multiply(U_ghost) - lambda*U_ghost);
            }
            

            // 组装到单元矩阵
            for(uInt i=0; i<Basis::NumBasis; ++i){
                for(uInt j=0; j<Basis::NumBasis; ++j){
                    Mat5x5 contrib = dF * (basis[i] * basis[j]) * jac_weight;
                    MatrixView<5*N,5*N,5,5> block(face_matrix,5*j,5*i);
                    block += contrib;
                    
                }
            }
            for(uInt j=0; j<Basis::NumBasis; ++j){
                Mat5x1 contrib = drhs * (basis[j]) * jac_weight;
                MatrixView<5*N,1,5,1> block(face_rhs,5*j,0);
                block += contrib;
            }
        }
        // debug("22222");
        // debug(face_matrix);

        #pragma omp critical
        {
        sparse_mat.add_block(cells[0], cells[0], face_matrix);
        sparse_rhs[cells[0]] -= face_rhs;
        // debug(vector2u{cells[0], cells[0]});
        }
    }
}










template<uInt Order, typename Flux, typename GaussQuadCell, typename GaussQuadFace>
void ImplicitConvection<Order, Flux, GaussQuadCell, GaussQuadFace>::assemble(const ComputingMesh& mesh, 
                 const LongVector<5*N>& old_solution,
                 const Scalar curr_time,
                 BlockSparseMatrix<5*N,5*N>& sparse_mat,
                 LongVector<5*N>& sparse_rhs){
    // print("123");
    assemble_cells(mesh,old_solution,curr_time,sparse_mat,sparse_rhs);
    // print("234");
    assemble_internals(mesh,old_solution,curr_time,sparse_mat,sparse_rhs);
    // print("345");
    assemble_boundarys(mesh,old_solution,curr_time,sparse_mat,sparse_rhs);
    // print("456");
}

template<uInt Order, typename Flux, typename GaussQuadCell, typename GaussQuadFace>
vector3f ImplicitConvection<Order, Flux, GaussQuadCell, GaussQuadFace>::transform_to_cell(const CompTriangleFace& face, 
                              const vector2f& uv, uInt side) const {
    const auto& nc = face.m_natural_coords[side];
    return nc[0]*(1-uv[0]-uv[1]) + nc[1]*uv[0] + nc[2]*uv[1];
}

template<uInt Order, typename Flux, typename GaussQuadCell, typename GaussQuadFace>
void ImplicitConvection<Order, Flux, GaussQuadCell, GaussQuadFace>::assemble_cells(const ComputingMesh& mesh, 
                 const LongVector<5*N>& old_solution,
                 const Scalar curr_time,
                 BlockSparseMatrix<5*N,5*N>& sparse_mat,
                 LongVector<5*N>& sparse_rhs) {
    // 预计算体积积分点数据
    constexpr uInt num_vol_points = QuadC::num_points;
    constexpr auto Qpoints = QuadC::get_points();
    constexpr auto Qweights = QuadC::get_weights();
    std::array<std::array<Scalar,3>, num_vol_points> vol_points;
    std::array<Scalar, num_vol_points> vol_weights;
    std::array<std::array<Scalar, Basis::NumBasis>, num_vol_points> vol_basis;
    std::array<std::array<vector3f, Basis::NumBasis>, num_vol_points> vol_grads;
    for(uInt g=0; g<num_vol_points; ++g) {
        vol_points[g] = Qpoints[g];
        vol_weights[g] = Qweights[g];
        vol_basis[g] = Basis::eval_all(vol_points[g][0],vol_points[g][1],vol_points[g][2]);
        vol_grads[g] = Basis::grad_all(vol_points[g][0],vol_points[g][1],vol_points[g][2]);
    }
    // uInt F(U) \cdot ( grad\phi_j  or  mathbf{n}\phi_j )
    // F(U) = V(U)U = V(U) sum_i C_i\phi_i = sum_i V(U)C_i \phi_i
    // F(U) \cdot grad\phi_j = sum_i (Fx*gx+Fy*gy+Fz*gz) C_i \phi_i
    // uInt F(U) \cdot grad\phi_j = sum_i uInt (Fx*gx+Fy*gy+Fz*gz)\phi_i C_i 
    // uInt (Fx*gx+Fy*gy+Fz*gz)\phi_i 为一个 5*5 矩阵
    // sum_i (5*5矩阵)_{i,j} C_i，这里的(i,j)是选择两个基函数，i试探函数，j测试函数 

    #pragma omp parallel for schedule(dynamic)
    for(uInt cid=0;cid<mesh.m_cells.size();cid++){
        // 存储的模式为：
        // 每个单元有100个自由度（系数）
        // [单元1,100自由度][单元2,100自由度]......
        // 其中，5个物理量，各有20个系数
        // [5个自由度，系数1][5个自由度，系数2][5个自由度，系数3]....
        const auto& cell = mesh.m_cells[cid];
        const auto& coef = old_solution[cid];

        // 或许可以拿到外面？  不确定编译器会不会重复分配内存
        DenseMatrix<5*N,5*N> cell_matrix;

        // 体积分部分
        for(uInt g=0; g<num_vol_points; ++g) {
            DenseMatrix<5,1> U = DenseMatrix<5,1>::Zeros();
            for(uInt bid=0; bid<Basis::NumBasis; ++bid) {
                const Scalar phi = vol_basis[g][bid];
                for(uInt k=0; k<5; ++k) 
                    U[k] += phi * coef[5*bid + k];
            }

            // 三组 J(F_{x,y,z},U)，每一个 5x5
            const auto JFU = Flux::computeJacobian(U);
            const Scalar jac_weight = vol_weights[g] * cell.compute_jacobian_det();

            for(uInt i=0; i<Basis::NumBasis; ++i) {
                // phi_i，计算 
                // ( J(F,U) sum_i c_i phi_i ) grad_phi_j
                // sum_i phi_i (grad_phi_j phi_J(F_,U) c_i) 
                // c_i 的系数为 phi_i(grad_phi_j J(F,U)) 
                const Scalar phi_i = vol_basis[g][i];
                for(uInt j=0; j<Basis::NumBasis; ++j) {
                    const auto& J = cell.get_invJacMat(); //DenseMatrix<3,3>(cell.compute_jacobian_mat()).inverse();
                    const auto& grad_phi_j = J.multiply(DenseMatrix<3,1>(vol_grads[g][j]));
                    // sum_{x,y,z}  J(F_x,U) phi_x_j
                    DenseMatrix<5,5> flux = JFU[0]*grad_phi_j[0] + JFU[1]*grad_phi_j[1] + JFU[2]*grad_phi_j[2];
                    // debug(FU.multiply(grad_phi_j));
                    // debug(Flux::multiply(U));
                    // const auto e = FU.multiply(grad_phi_j)-Flux::multiply(U);
                    // if(e.dot(e)>1e-20) debug(e.dot(e));
                    flux *= phi_i * jac_weight;

                    // MatrixView<5*N,5*N,5,5> block(cell_matrix,5*j,5*i);
                    // block -= flux;
                    // 遍历 5x5 的单元贡献
                    for (uInt row = 0; row < 5; ++row) {
                        for (uInt col = 0; col < 5; ++col) {
                            // 块左上角在全局矩阵中的偏移
                            uInt global_row_LL = 5 * j + row;
                            uInt global_col_LL = 5 * i + col;
                            // LL 块
                            cell_matrix(global_row_LL, global_col_LL) = flux(row, col);
                        }
                    }
                }
            }
        }
        // debug(cell_matrix.multiply(coef));
        #pragma omp critical
        sparse_mat.add_block(cid, cid, cell_matrix);
    }
}


template<uInt Order, typename Flux, typename GaussQuadCell, typename GaussQuadFace>
void ImplicitConvection<Order, Flux, GaussQuadCell, GaussQuadFace>::assemble_internals(const ComputingMesh& mesh, 
                 const LongVector<5*N>& old_solution,
                 const Scalar curr_time,
                 BlockSparseMatrix<5*N,5*N>& sparse_mat,
                 LongVector<5*N>& sparse_rhs){
    constexpr uInt num_face_points = QuadF::num_points;
    constexpr auto Qpoints = QuadF::get_points();
    constexpr auto Qweights = QuadF::get_weights();
    // std::vector<std::array<BlkMat,4>> face_mat_list(mesh.m_faces.size());
    // 只计算内部的
    #pragma omp parallel for schedule(dynamic)
    for(uInt fid=0; fid<mesh.m_faces.size(); ++fid) {
        const auto& face = mesh.m_faces[fid];
        const auto& cells = face.m_neighbor_cells;
        
        // 跳过边界单元
        if(cells[1] == uInt(-1)) continue;
        // 后面就是只有内部面

        // 获取左右单元系数
        const auto& coef_L = old_solution[cells[0]];
        const auto& coef_R = old_solution[cells[1]];

        // 左右两个面的分块矩阵
        BlkMat face_matrix_LL, face_matrix_LR;
        BlkMat face_matrix_RL, face_matrix_RR;

        for(uInt g=0; g<num_face_points; ++g) {
            const auto& uv = Qpoints[g];
            const Scalar weight = Qweights[g] * face.compute_jacobian_det();

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
            auto F_L = Flux::computeJacobian(U_L);
            auto F_R = Flux::computeJacobian(U_R);

            // 计算法向通量雅可比
            DenseMatrix<5,5> Fn_L, Fn_R;
            for(uInt d=0; d<3; ++d) {
                Fn_L += F_L[d] * face.m_normal[d];
                Fn_R += F_R[d] * face.m_normal[d];
            }
            // debug(Fn_L);

            // Lax-Friedrichs参数
            const Scalar lambda = Flux::computeWaveSpeed(U_L, U_R);
            //  debug(lambda);
            // 通量雅可比贡献
            const DenseMatrix<5,5> dF_L = 0.5*(Fn_L + lambda*DenseMatrix<5,5>::Identity());
            const DenseMatrix<5,5> dF_R = 0.5*(Fn_R - lambda*DenseMatrix<5,5>::Identity());
            // debug(dF_L);
            // 组装到左右单元矩阵
            // for(uInt i=0; i<Basis::NumBasis; ++i) {
            //     for(uInt j=0; j<Basis::NumBasis; ++j) {
            //         const Scalar phi_iL = basis_L[i];
            //         const Scalar phi_jL = basis_L[j];
            //         const Scalar phi_iR = basis_R[i];
            //         const Scalar phi_jR = basis_R[j];

            //         // auto contrib_L = dF_L * (phi_iL * phi_jL) * weight;
            //         // auto contrib_R = dF_R * (phi_iR * phi_jR) * weight;

            //         MatrixView<5*N,5*N,5,5> block_LL(face_matrix_LL, 5*j, 5*i);
            //         MatrixView<5*N,5*N,5,5> block_LR(face_matrix_LR, 5*j, 5*i);
            //         MatrixView<5*N,5*N,5,5> block_RL(face_matrix_RL, 5*j, 5*i);
            //         MatrixView<5*N,5*N,5,5> block_RR(face_matrix_RR, 5*j, 5*i);
            //         // debug(contrib_L);
            //         block_LL += dF_L * (phi_iL * phi_jL) * weight;
            //         block_LR += dF_R * (phi_iR * phi_jL) * weight;
            //         block_RL += dF_L * (phi_iL * phi_jR) * weight;
            //         block_RR += dF_R * (phi_iR * phi_jR) * weight;
            //     }
            // }
            for(uInt i = 0; i < Basis::NumBasis; ++i) {
                for(uInt j = 0; j < Basis::NumBasis; ++j) {
                    const Scalar phi_iL = basis_L[i];
                    const Scalar phi_jL = basis_L[j];
                    const Scalar phi_iR = basis_R[i];
                    const Scalar phi_jR = basis_R[j];

                    // 计算系数
                    const Scalar coeff_LL = (phi_iL * phi_jL * weight);
                    const Scalar coeff_LR = (phi_iR * phi_jL * weight);
                    const Scalar coeff_RL = (phi_iL * phi_jR * weight);
                    const Scalar coeff_RR = (phi_iR * phi_jR * weight);
                    
                    // 遍历 5x5 的单元贡献
                    for (uInt row = 0; row < 5; ++row) {
                        for (uInt col = 0; col < 5; ++col) {
                            // 块左上角在全局矩阵中的偏移
                            uInt global_row_LL = 5 * j + row;
                            uInt global_col_LL = 5 * i + col;
                            // LL 块
                            face_matrix_LL(global_row_LL, global_col_LL) = coeff_LL * dF_L(row, col);
                        }
                    }
                    // 遍历 5x5 的单元贡献
                    for (uInt row = 0; row < 5; ++row) {
                        for (uInt col = 0; col < 5; ++col) {
                            // 块左上角在全局矩阵中的偏移
                            uInt global_row_LL = 5 * j + row;
                            uInt global_col_LL = 5 * i + col;
                            // LR 块
                            face_matrix_LR(global_row_LL, global_col_LL) = coeff_LR * dF_R(row, col);
                        }
                    }
                    // 遍历 5x5 的单元贡献
                    for (uInt row = 0; row < 5; ++row) {
                        for (uInt col = 0; col < 5; ++col) {
                            // 块左上角在全局矩阵中的偏移
                            uInt global_row_LL = 5 * j + row;
                            uInt global_col_LL = 5 * i + col;
                            // RL 块
                            face_matrix_RL(global_row_LL, global_col_LL) = coeff_RL * dF_L(row, col);
                        }
                    }
                    // 遍历 5x5 的单元贡献
                    for (uInt row = 0; row < 5; ++row) {
                        for (uInt col = 0; col < 5; ++col) {
                            // 块左上角在全局矩阵中的偏移
                            uInt global_row_LL = 5 * j + row;
                            uInt global_col_LL = 5 * i + col;
                            // RR 块
                            face_matrix_RR(global_row_LL, global_col_LL) = coeff_RR * dF_R(row, col);
                        }
                    }
                }
            }
        }
        // 这四个矩阵。。。。。终于。。。。。。
        // face_mat_list[fid] = {face_matrix_LL,face_matrix_LR,-face_matrix_RL,-face_matrix_RR};
        #pragma omp critical
        {   
            sparse_mat.add_block(cells[0], cells[0], face_matrix_LL);
            sparse_mat.add_block(cells[0], cells[1], face_matrix_LR);
            sparse_mat.add_block(cells[1], cells[0], (-1.)*face_matrix_RL); 
            sparse_mat.add_block(cells[1], cells[1], (-1.)*face_matrix_RR);
            // debug(vector2u{cells[0], cells[1]});
        }
    }
    // for(uInt fid=0; fid<mesh.m_faces.size(); ++fid) {
    //     const auto& face = mesh.m_faces[fid];
    //     const auto& cells = face.m_neighbor_cells;
        
    //     // 跳过边界单元
    //     if(cells[1] == uInt(-1)) continue;
    //     const auto& face_matrix = face_mat_list[fid];
    //     sparse_mat.add_block(cells[0], cells[0], face_matrix[0]);
    //     sparse_mat.add_block(cells[0], cells[1], face_matrix[1]);
    //     sparse_mat.add_block(cells[1], cells[0], face_matrix[2]); 
    //     sparse_mat.add_block(cells[1], cells[1], face_matrix[3]);
    // }
}

