#include "dg/dg_schemes/implicit_diffusion/implicit_diffusion.h"


template<uInt Order, typename Flux, typename GaussQuadCell, typename GaussQuadFace>
void ImplicitDiffusion<Order, Flux, GaussQuadCell, GaussQuadFace>::assemble_boundarys(const ComputingMesh& mesh, 
                 const LongVector<5*N>& old_solution,
                 const Scalar curr_time,
                 BlockSparseMatrix<5*N,5*N>& sparse_mat,
                 LongVector<5*N>& sparse_rhs){
    constexpr uInt num_face_points = QuadF::num_points;
    constexpr auto Qpoints = QuadF::get_points();
    constexpr auto Qweights = QuadF::get_weights();
    #pragma omp parallel for schedule(dynamic)
    for(uInt fid=0; fid<mesh.m_faces.size(); ++fid) {
        // std::array<Scalar,5> cumsum = {0,0,0,0,0};
        const auto& face = mesh.m_faces[fid];
        const auto& cells = face.m_neighbor_cells;
        if(cells[1] != uInt(-1)) continue;
        const auto bc = mesh.m_boundaryTypes[fid];
        const auto& coef = old_solution[cells[0]];
        DenseMatrix<5*N,5*N> face_matrix;
        DenseMatrix<5*N,1> face_rhs;
        for(uInt g=0; g<num_face_points; ++g) {
            const auto& xi = transform_to_cell(face, Qpoints[g], 0);
            const auto& basis = Basis::eval_all(xi[0], xi[1], xi[2]);
            const auto& grad_basis = Basis::grad_all(xi[0], xi[1], xi[2]);
            const Scalar jac_weight = Qweights[g] * face.compute_jacobian_det();
            // 预计算所有的grad_phi_i/j
            std::array<DenseMatrix<3,1>, Basis::NumBasis> J_grad_phi_k;
            const auto& J = mesh.m_cells[cells[0]].get_invJacMat();
            for(uInt k=0; k<Basis::NumBasis; ++k) {
                J_grad_phi_k[k] = J.multiply(grad_basis[k]);
            }
            DenseMatrix<5,1> U_inner;
            DenseMatrix<5,3> grad_U_inner = DenseMatrix<5,3>::Zeros();
            for(uInt bid=0; bid<Basis::NumBasis; ++bid) {
                for(uInt k=0; k<5; ++k){
                    U_inner[k] += basis[bid] * coef[5*bid + k];
                    grad_U_inner(k,0) += J_grad_phi_k[bid][0] * coef[5*bid + k];
                    grad_U_inner(k,1) += J_grad_phi_k[bid][1] * coef[5*bid + k];
                    grad_U_inner(k,2) += J_grad_phi_k[bid][2] * coef[5*bid + k];
                }
            }
            DenseMatrix<5,1> U_ghost;
            DenseMatrix<3,1> normal = DenseMatrix<3,1>(face.m_normal);
            Scalar beta_p = 0.5*(Order+1)*(Order+1)/mesh.m_cells[cells[0]].m_h;
            if (bc==BoundaryType::Dirichlet) {
                const auto& U_D = U_Xi(mesh.m_cells[cells[0]],xi,curr_time);
                U_ghost = U_D + (U_D - U_inner);
                
                U_ghost = 0.5*(U_inner+U_ghost);
                
                const auto& J_G_U = Flux::computeJacobian(U_ghost,m_mu);
                
                // 第一级预计算
                const DenseMatrix<5,5> normal_normal_flux = Flux::computevKu(normal,normal,J_G_U);
                // 第二级预计算 (i循环)
                std::array<DenseMatrix<5,5>,Basis::NumBasis> grad_phi_flux;
                for(uInt i=0; i<Basis::NumBasis; ++i) {
                    grad_phi_flux[i] = Flux::computevKu(normal,J_grad_phi_k[i],J_G_U);
                }
                // 第三级预计算 (j循环)
                std::array<DenseMatrix<5,5>,Basis::NumBasis> phi_grad_flux;
                for(uInt j=0; j<Basis::NumBasis; j++) {
                    phi_grad_flux[j] = Flux::computevKu(J_grad_phi_k[j],normal,J_G_U);
                }
                for(uInt j=0; j<Basis::NumBasis; j++) {
                    const auto& sym_pen = 0.5*phi_grad_flux[j] - beta_p * basis[j]*normal_normal_flux;
                    for(uInt i=0; i<Basis::NumBasis; ++i) {
                        DenseMatrix<5,5> flux = 0.5*basis[j]*grad_phi_flux[i] + basis[i]*sym_pen;
                        MatrixView<5*N,5*N,5,5> block(face_matrix,5*j,5*i);
                        block += flux * jac_weight;
                    }
                    MatrixView<5*N,1,5,1> block(face_rhs,5*j,0);
                    block += sym_pen.multiply(U_ghost) * jac_weight;
                }
            }
            else if (bc==BoundaryType::WallTD) {
                const auto& U_D = U_Xi(mesh.m_cells[cells[0]],xi,curr_time);
                U_ghost = U_D + (U_D - U_inner);
                U_ghost[0] = U_inner[0];
                Scalar cvT_inner = U_inner[4]-0.5*(U_inner[1]*U_inner[1]+U_inner[2]*U_inner[2]+U_inner[3]*U_inner[3])/U_inner[0];
                Scalar cvT_ghost = 2*300 - cvT_inner;
                U_ghost[4] = cvT_ghost + 0.5*(U_ghost[1]*U_ghost[1]+U_ghost[2]*U_ghost[2]+U_ghost[3]*U_ghost[3])/U_ghost[0];

                U_ghost = 0.5*(U_inner+U_ghost);
                const auto& J_G_U = Flux::computeJacobian(U_ghost,m_mu);
                // 第一级预计算
                const DenseMatrix<5,5> normal_normal_flux = Flux::computevKu(normal,normal,J_G_U);
                // 第二级预计算 (i循环)
                std::array<DenseMatrix<5,5>,Basis::NumBasis> grad_phi_flux;
                for(uInt i=0; i<Basis::NumBasis; ++i) {
                    grad_phi_flux[i] = Flux::computevKu(normal,J_grad_phi_k[i],J_G_U);
                }
                // 第三级预计算 (j循环)
                std::array<DenseMatrix<5,5>,Basis::NumBasis> phi_grad_flux;
                for(uInt j=0; j<Basis::NumBasis; j++) {
                    phi_grad_flux[j] = Flux::computevKu(J_grad_phi_k[j],normal,J_G_U);
                }
                for(uInt j=0; j<Basis::NumBasis; j++) {
                    const auto& sym_pen = 0.5*phi_grad_flux[j] - beta_p * basis[j]*normal_normal_flux;
                    for(uInt i=0; i<Basis::NumBasis; ++i) {
                        DenseMatrix<5,5> flux = 0.5*basis[j]*grad_phi_flux[i] + basis[i]*sym_pen;
                        MatrixView<5*N,5*N,5,5> block(face_matrix,5*j,5*i);
                        block += flux * jac_weight;
                    }
                    MatrixView<5*N,1,5,1> block(face_rhs,5*j,0);
                    block += sym_pen.multiply(U_ghost) * jac_weight;
                }
            }
            else if (bc==BoundaryType::WallTN) {
                const auto& U_D = U_Xi(mesh.m_cells[cells[0]],xi,curr_time);
                U_ghost = U_D + (U_D - U_inner);
                U_ghost[0] = U_inner[0];
                Scalar cvT_inner = U_inner[4]-0.5*(U_inner[1]*U_inner[1]+U_inner[2]*U_inner[2]+U_inner[3]*U_inner[3])/U_inner[0];
                Scalar cvT_ghost = cvT_inner;
                U_ghost[4] = cvT_ghost + 0.5*(U_ghost[1]*U_ghost[1]+U_ghost[2]*U_ghost[2]+U_ghost[3]*U_ghost[3])/U_ghost[0];

                U_ghost = 0.5*(U_inner+U_ghost);
                const auto& J_G_U = Flux::computeJacobian(U_ghost,m_mu);
                // 第一级预计算
                const DenseMatrix<5,5> normal_normal_flux = Flux::computevKu(normal,normal,J_G_U);
                // 第二级预计算 (i循环)
                std::array<DenseMatrix<5,5>,Basis::NumBasis> grad_phi_flux;
                for(uInt i=0; i<Basis::NumBasis; ++i) {
                    grad_phi_flux[i] = Flux::computevKu(normal,J_grad_phi_k[i],J_G_U);
                }
                // 第三级预计算 (j循环)
                std::array<DenseMatrix<5,5>,Basis::NumBasis> phi_grad_flux;
                for(uInt j=0; j<Basis::NumBasis; j++) {
                    phi_grad_flux[j] = Flux::computevKu(J_grad_phi_k[j],normal,J_G_U);
                }
                for(uInt j=0; j<Basis::NumBasis; j++) {
                    const auto& sym_pen = 0.5*phi_grad_flux[j] - beta_p * basis[j]*normal_normal_flux;
                    for(uInt i=0; i<Basis::NumBasis; ++i) {
                        DenseMatrix<5,5> flux = 0.5*basis[j]*grad_phi_flux[i] + basis[i]*sym_pen;
                        MatrixView<5*N,5*N,5,5> block(face_matrix,5*j,5*i);
                        block += flux * jac_weight;
                    }
                    MatrixView<5*N,1,5,1> block(face_rhs,5*j,0);
                    block += sym_pen.multiply(U_ghost) * jac_weight;
                }
            }
            else if(bc==BoundaryType::Pseudo3DZ) {
                U_ghost = U_inner * DenseMatrix<5,1>({1,1,1,-1,1});
                U_ghost = 0.5*(U_inner+U_ghost);
                const auto& J_G_U = Flux::computeJacobian(U_ghost,m_mu);
                // 第一级预计算
                const DenseMatrix<5,5> normal_normal_flux = Flux::computevKu(normal,normal,J_G_U);
                // 第三级预计算 (j循环)
                std::array<DenseMatrix<5,5>,Basis::NumBasis> phi_grad_flux;
                for(uInt j=0; j<Basis::NumBasis; j++) {
                    phi_grad_flux[j] = Flux::computevKu(J_grad_phi_k[j],normal,J_G_U);
                }
                for(uInt j=0; j<Basis::NumBasis; j++) {
                    const auto& sym_pen = 0.5*phi_grad_flux[j] - beta_p * basis[j]*normal_normal_flux;
                    for(uInt i=0; i<Basis::NumBasis; ++i) {
                        auto id = DenseMatrix<5,5>::Zeros();
                        id(3,3) = 2;
                        DenseMatrix<5,5> flux = basis[i]*sym_pen.multiply(id);
                    }
                }
            }
        }
        #pragma omp critical
        {
        sparse_mat.add_block(cells[0], cells[0], (-1.)*face_matrix);
        sparse_rhs[cells[0]] += (-1.)*face_rhs;
        }
    }
}


template<uInt Order, typename Flux, typename GaussQuadCell, typename GaussQuadFace>
void ImplicitDiffusion<Order, Flux, GaussQuadCell, GaussQuadFace>::assemble(const ComputingMesh& mesh, 
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
vector3f ImplicitDiffusion<Order, Flux, GaussQuadCell, GaussQuadFace>::transform_to_cell(const CompTriangleFace& face, 
                            const vector2f& uv, uInt side) const {
    const auto& nc = face.m_natural_coords[side];
    return nc[0]*(1-uv[0]-uv[1]) + nc[1]*uv[0] + nc[2]*uv[1];
}

template<uInt Order, typename Flux, typename GaussQuadCell, typename GaussQuadFace>
void ImplicitDiffusion<Order, Flux, GaussQuadCell, GaussQuadFace>::assemble_cells(const ComputingMesh& mesh, 
                 const LongVector<5*N>& old_solution,
                 const Scalar curr_time,
                 BlockSparseMatrix<5*N,5*N>& sparse_mat,
                 LongVector<5*N>& sparse_rhs){
    // 预计算 phi_i(x_g),  grad_phi_i(x_g)
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
        // val+=vol_weights[g];
    }
    


    #pragma omp parallel for schedule(dynamic)
    for(uInt cid=0;cid<mesh.m_cells.size();cid++){

        const auto& cell = mesh.m_cells[cid];
        const auto& coef = old_solution[cid];
        DenseMatrix<5*N,5*N> cell_matrix;
        
        const auto& J = cell.get_invJacMat();
        // debug(J);
        // debug(coef);
        for(uInt g=0; g<num_vol_points; ++g) {
            // 预计算所有的grad_phi_i/j
            std::array<DenseMatrix<3,1>, Basis::NumBasis> J_grad_phi_k;
            for(uInt k=0; k<Basis::NumBasis; ++k) {
                J_grad_phi_k[k] = J.multiply(vol_grads[g][k]);
            }
            DenseMatrix<5,1> U = DenseMatrix<5,1>::Zeros();
            for(uInt bid=0; bid<Basis::NumBasis; ++bid) {
                const Scalar phi = vol_basis[g][bid];
                for(uInt k=0; k<5; ++k) {
                    U[k] += phi * coef[5*bid + k];
                }
            }
            


            const Scalar jac_weight = vol_weights[g] * cell.compute_jacobian_det();
            for(uInt i=0; i<Basis::NumBasis; ++i) {
                for(uInt j=0; j<Basis::NumBasis; j++) {
                    DenseMatrix<5,5> flux = Flux::computevKu(J_grad_phi_k[j],J_grad_phi_k[i],U,m_mu);
                    MatrixView<5*N,5*N,5,5> block(cell_matrix,5*j,5*i);
                    block -= flux * jac_weight;
                }
            }
        }
        #pragma omp critical
        sparse_mat.add_block(cid, cid, (-1.)*cell_matrix);
    }
}

template<uInt Order, typename Flux, typename GaussQuadCell, typename GaussQuadFace>
void ImplicitDiffusion<Order, Flux, GaussQuadCell, GaussQuadFace>::assemble_internals(const ComputingMesh& mesh, 
                 const LongVector<5*N>& old_solution,
                 const Scalar curr_time,
                 BlockSparseMatrix<5*N,5*N>& sparse_mat,
                 LongVector<5*N>& sparse_rhs){
    constexpr uInt num_face_points = QuadF::num_points;
    constexpr auto Qpoints = QuadF::get_points();
    constexpr auto Qweights = QuadF::get_weights();
    #pragma omp parallel for schedule(dynamic)
    for(uInt fid=0; fid<mesh.m_faces.size(); ++fid) {
        // std::array<Scalar,5> cumsum = {0,0,0,0,0};
        const auto& face = mesh.m_faces[fid];
        const auto& cells = face.m_neighbor_cells;
        if(cells[1] == uInt(-1)) continue;
        const auto& coef_L = old_solution[cells[0]];
        const auto& coef_R = old_solution[cells[1]];
        DenseMatrix<5*N,5*N> face_matrix_LL, face_matrix_LR;
        DenseMatrix<5*N,5*N> face_matrix_RL, face_matrix_RR;
        for(uInt g=0; g<num_face_points; ++g) {

            const auto& xi_L = transform_to_cell(face, Qpoints[g], 0);
            const auto& xi_R = transform_to_cell(face, Qpoints[g], 1);
            const auto& basis_L = Basis::eval_all(xi_L[0], xi_L[1], xi_L[2]);
            const auto& basis_R = Basis::eval_all(xi_R[0], xi_R[1], xi_R[2]);
            const auto& grad_basis_L = Basis::grad_all(xi_L[0], xi_L[1], xi_L[2]);
            const auto& grad_basis_R = Basis::grad_all(xi_R[0], xi_R[1], xi_R[2]);
            const Scalar jac_weight = Qweights[g] * face.compute_jacobian_det();

            // 预计算所有的grad_phi_i/j
            std::array<DenseMatrix<3,1>, Basis::NumBasis> J_grad_phi_k_L,J_grad_phi_k_R;
            const auto& J_L = mesh.m_cells[cells[0]].get_invJacMat();
            const auto& J_R = mesh.m_cells[cells[1]].get_invJacMat();
            for(uInt k=0; k<Basis::NumBasis; ++k) {
                J_grad_phi_k_L[k] = J_L.multiply(grad_basis_L[k]);
                J_grad_phi_k_R[k] = J_R.multiply(grad_basis_R[k]);
            }

            DenseMatrix<5,1> U_L, U_R;
            for(uInt bid=0; bid<Basis::NumBasis; ++bid) {
                for(uInt k=0; k<5; ++k) {
                    U_L[k] += basis_L[bid] * coef_L[5*bid + k];
                    U_R[k] += basis_R[bid] * coef_R[5*bid + k];
                }
            }
            const auto& U_avg = 0.5*(U_L+U_R);
            Scalar beta_p = 0.5*(Order+1)*(Order+1)/(0.5*(mesh.m_cells[cells[0]].m_h + mesh.m_cells[cells[1]].m_h));
            const auto& normal = DenseMatrix<3,1>(face.m_normal);
            const auto& J_G_U = Flux::computeJacobian(U_avg,m_mu);

            // 第一级预计算
            const DenseMatrix<5,5> normal_normal_flux = Flux::computevKu(normal, normal, J_G_U);

            // 第二级预计算 (i循环)
            std::array<DenseMatrix<5,5>,Basis::NumBasis> grad_phi_L_flux;
            std::array<DenseMatrix<5,5>,Basis::NumBasis> grad_phi_R_flux;
            for(uInt i=0; i<Basis::NumBasis; ++i) {
                grad_phi_L_flux[i] = Flux::computevKu(normal, J_grad_phi_k_L[i], J_G_U);
                grad_phi_R_flux[i] = Flux::computevKu(normal, J_grad_phi_k_R[i], J_G_U);
            }

            // 第三级预计算 (j循环)
            std::array<DenseMatrix<5,5>,Basis::NumBasis> phi_grad_L_flux;
            std::array<DenseMatrix<5,5>,Basis::NumBasis> phi_grad_R_flux;
            for(uInt j=0; j<Basis::NumBasis; j++) {
                phi_grad_L_flux[j] = Flux::computevKu(J_grad_phi_k_L[j], normal, J_G_U);
                phi_grad_R_flux[j] = Flux::computevKu(J_grad_phi_k_R[j], normal, J_G_U);
            }

            // 主计算循环
            for(uInt j=0; j<Basis::NumBasis; j++) {
                for(uInt i=0; i<Basis::NumBasis; ++i) {
                    // DenseMatrix<5,5> flux_LL = DenseMatrix<5,5>::Zeros(); 
                    // DenseMatrix<5,5> flux_LR = DenseMatrix<5,5>::Zeros(); 
                    // DenseMatrix<5,5> flux_RL = DenseMatrix<5,5>::Zeros(); 
                    // DenseMatrix<5,5> flux_RR = DenseMatrix<5,5>::Zeros(); 
                    // flux_LL += 0.5 * basis_L[j]*Flux::computevKu(normal,J_grad_phi_k_L[i],J_G_U);
                    // flux_LR += 0.5 * basis_L[j]*Flux::computevKu(normal,J_grad_phi_k_R[i],J_G_U);
                    // flux_RL -= 0.5 * basis_R[j]*Flux::computevKu(normal,J_grad_phi_k_L[i],J_G_U);
                    // flux_RR -= 0.5 * basis_R[j]*Flux::computevKu(normal,J_grad_phi_k_R[i],J_G_U);

                    // flux_LL += 0.5 * basis_L[i] * Flux::computevKu(J_grad_phi_k_L[j],normal,J_G_U);
                    // flux_LR -= 0.5 * basis_R[i] * Flux::computevKu(J_grad_phi_k_L[j],normal,J_G_U);
                    // flux_RL -= 0.5 * basis_L[i] * Flux::computevKu(J_grad_phi_k_R[j],normal,J_G_U);
                    // flux_RR += 0.5 * basis_R[i] * Flux::computevKu(J_grad_phi_k_R[j],normal,J_G_U);
                    
                    // flux_LL -= beta_p * basis_L[j]*basis_L[i]*Flux::computevKu(normal,normal,J_G_U);
                    // flux_LR += beta_p * basis_L[j]*basis_R[i]*Flux::computevKu(normal,normal,J_G_U);
                    // flux_RL += beta_p * basis_R[j]*basis_L[i]*Flux::computevKu(normal,normal,J_G_U);
                    // flux_RR -= beta_p * basis_R[j]*basis_R[i]*Flux::computevKu(normal,normal,J_G_U);
                    DenseMatrix<5,5> flux_LL = 0.5 * (basis_L[j] * grad_phi_L_flux[i] + basis_L[i] * phi_grad_L_flux[j])
                                            - beta_p * basis_L[j] * basis_L[i] * normal_normal_flux;
                    
                    DenseMatrix<5,5> flux_LR = 0.5 * (basis_L[j] * grad_phi_R_flux[i] - basis_R[i] * phi_grad_L_flux[j])
                                            + beta_p * basis_L[j] * basis_R[i] * normal_normal_flux;
                    
                    DenseMatrix<5,5> flux_RL = 0.5 * (-basis_R[j] * grad_phi_L_flux[i] - basis_L[i] * phi_grad_R_flux[j])
                                            + beta_p * basis_R[j] * basis_L[i] * normal_normal_flux;
                    
                    DenseMatrix<5,5> flux_RR = 0.5 * (-basis_R[j] * grad_phi_R_flux[i] + basis_R[i] * phi_grad_R_flux[j])
                                            - beta_p * basis_R[j] * basis_R[i] * normal_normal_flux;
                    
                    MatrixView<5*N,5*N,5,5> block_LL(face_matrix_LL, 5*j, 5*i);
                    MatrixView<5*N,5*N,5,5> block_LR(face_matrix_LR, 5*j, 5*i);
                    MatrixView<5*N,5*N,5,5> block_RL(face_matrix_RL, 5*j, 5*i);
                    MatrixView<5*N,5*N,5,5> block_RR(face_matrix_RR, 5*j, 5*i);
                    block_LL += flux_LL * jac_weight;
                    block_LR += flux_LR * jac_weight;
                    block_RL += flux_RL * jac_weight;
                    block_RR += flux_RR * jac_weight;
                }
            }
        }
        #pragma omp critical
        {   
            sparse_mat.add_block(cells[0], cells[0], (-1.)*face_matrix_LL);
            sparse_mat.add_block(cells[0], cells[1], (-1.)*face_matrix_LR);
            sparse_mat.add_block(cells[1], cells[0], (-1.)*face_matrix_RL); 
            sparse_mat.add_block(cells[1], cells[1], (-1.)*face_matrix_RR);
        }
    }
}

