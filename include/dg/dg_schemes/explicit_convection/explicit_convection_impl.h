#include "dg/dg_schemes/explicit_convection/explicit_convection.h"


template<uInt Order, typename Flux, typename GaussQuadCell, typename GaussQuadFace>
LongVector<5*ExplicitConvection<Order, Flux, GaussQuadCell, GaussQuadFace>::N> ExplicitConvection<Order, Flux, GaussQuadCell, GaussQuadFace>::eval_boundarys(const ComputingMesh& mesh, 
                 const LongVector<5*N>& old_solution,
                 const Scalar curr_time){
    constexpr uInt num_face_points = QuadF::num_points;
    constexpr auto Qpoints = QuadF::get_points();
    constexpr auto Qweights = QuadF::get_weights();
    LongVector<5*N> result(old_solution.size());
    #pragma omp parallel for schedule(dynamic)
    for(uInt fid=0; fid<mesh.m_faces.size(); ++fid) {
        const auto& face = mesh.m_faces[fid];
        const auto& cells = face.m_neighbor_cells;
        if(cells[1] != uInt(-1)) continue;
        const auto bc = mesh.m_boundaryTypes[fid];
        const auto& coef = old_solution[cells[0]];
        for(uInt g=0; g<num_face_points; ++g) {
            const auto& xi = transform_to_cell(face, Qpoints[g], 0);
            const auto& basis = Basis::eval_all(xi[0], xi[1], xi[2]);
            const Scalar jac_weight = Qweights[g] * face.m_area * 2;
            
            DenseMatrix<5,1> U_inner;
            for(uInt bid=0; bid<Basis::NumBasis; ++bid) {
                for(uInt k=0; k<5; ++k)
                    U_inner[k] += basis[bid] * coef[5*bid + k];
            }
            // auto U_ghost = (bc==BoundaryType::Dirichlet)?
            //                 2*DenseMatrix<5,1>({1,1,0.5,0,2.5+0.625})-U_inner:   //Dirichlet     下面Neumann
            //                 U_inner + 2*0.0*vec_dot(vector3f{0,0,0},face.m_normal);
            
            if (bc==BoundaryType::Dirichlet) {
                const auto& U_D = DenseMatrix<5,1>({rho_Xi(mesh.m_cells[cells[0]],xi,curr_time),
                                                    rhou_Xi(mesh.m_cells[cells[0]],xi,curr_time),
                                                    rhov_Xi(mesh.m_cells[cells[0]],xi,curr_time),
                                                    rhow_Xi(mesh.m_cells[cells[0]],xi,curr_time),
                                                    rhoe_Xi(mesh.m_cells[cells[0]],xi,curr_time)});
                const auto& U_ghost = U_D + (U_D - U_inner);
                const auto& FU_inner = Flux::computeFlux(U_inner);
                const auto& FU_ghost = Flux::computeFlux(U_ghost);
                const auto& FUn_inner = FU_inner.multiply(face.m_normal);
                const auto& FUn_ghost = FU_ghost.multiply(face.m_normal);
                const auto& Un = FUn_inner[1] * face.m_normal[0] + FUn_inner[2] * face.m_normal[1] + FUn_inner[3] * face.m_normal[2];

                
                const Scalar lambda = Flux::computeWaveSpeed(U_inner, U_ghost);
                // debug(lambda);

                const auto& LF_flux = 0.5*(FUn_inner + FUn_ghost + lambda*(U_inner-U_ghost));
                // const auto& LF_flux = Un>0? FUn_inner : FUn_ghost;

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
            else if(bc==BoundaryType::Pseudo3DZ) {
                // 把伪三维问题，视为是上下边界 (rho w)_{ghost} = -(rho w)_{inner}
                const auto& U_ghost = U_inner * DenseMatrix<5,1>({1,1,1,-1,1});
                const auto& FU_inner = Flux::computeFlux(U_inner);
                const auto& FU_ghost = Flux::computeFlux(U_ghost);
                const auto& FUn_inner = FU_inner.multiply(face.m_normal);
                const auto& FUn_ghost = FU_ghost.multiply(face.m_normal);
                const auto& Un = FUn_inner[1] * face.m_normal[0] + FUn_inner[2] * face.m_normal[1] + FUn_inner[3] * face.m_normal[2];

                
                const Scalar lambda = Flux::computeWaveSpeed(U_inner, U_ghost);
                // if(lambda>10)debug(lambda);
                const auto& LF_flux = 0.5*(FUn_inner + FUn_ghost + lambda*(U_inner-U_ghost));
                // const auto& LF_flux = Un>0? FUn_inner : FUn_ghost;
                // const auto& LF_flux = compute_roe_flux(U_L, U_R, face.m_normal);
                // if(cells[0] == target){
                //     debug(face.m_normal);
                //     // debug(std::array<Scalar,5>{FUn_inner[0],FUn_inner[1],FUn_inner[2],FUn_inner[3],FUn_inner[4]});
                //     // debug(std::array<Scalar,5>{FUn_ghost[0],FUn_ghost[1],FUn_ghost[2],FUn_ghost[3],FUn_ghost[4]});
                //     debug(std::array<Scalar,5>{LF_flux[3],FUn_inner[3],FUn_ghost[3],U_inner[3],U_ghost[3]});
                // }
                for(uInt j=0; j<Basis::NumBasis; ++j) {
                    const Scalar phi_j = basis[j];

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
            }
        }
    }
    return result;
}
template<uInt Order, typename Flux, typename GaussQuadCell, typename GaussQuadFace>
vector3f ExplicitConvection<Order, Flux, GaussQuadCell, GaussQuadFace>::transform_to_cell(const CompTriangleFace& face, 
                              const vector2f& uv, uInt side) const {
    const auto& nc = face.m_natural_coords[side];
    return nc[0]*(1-uv[0]-uv[1]) + nc[1]*uv[0] + nc[2]*uv[1];
}

template<uInt Order, typename Flux, typename GaussQuadCell, typename GaussQuadFace>
LongVector<5*ExplicitConvection<Order, Flux, GaussQuadCell, GaussQuadFace>::N> ExplicitConvection<Order, Flux, GaussQuadCell, GaussQuadFace>::eval(const ComputingMesh& mesh, 
                const LongVector<5*N>& old_solution,
                const Scalar curr_time){
    return eval_cells(mesh,old_solution,curr_time) + eval_internals(mesh,old_solution,curr_time) + eval_boundarys(mesh,old_solution,curr_time);
}


template<uInt Order, typename Flux, typename GaussQuadCell, typename GaussQuadFace>
LongVector<5*ExplicitConvection<Order, Flux, GaussQuadCell, GaussQuadFace>::N> 
ExplicitConvection<Order, Flux, GaussQuadCell, GaussQuadFace>::eval_cells(
                 const ComputingMesh& mesh, 
                 const LongVector<5*N>& old_solution,
                 const Scalar curr_time){
    LongVector<5*N> result(old_solution.size());

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

    #pragma omp parallel for schedule(dynamic)
    for(uInt cid=0;cid<mesh.m_cells.size();cid++){
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
            const auto& FU = Flux::computeFlux(U);
            // debug(FU);
            const Scalar jac_weight = vol_weights[g] * cell.m_volume * 6;
            for(uInt j=0; j<Basis::NumBasis; ++j) {
                // [p1-p0,p2-p0,p3-p0]已经自带转置，只需求逆
                const auto& J = cell.get_invJacMat();//DenseMatrix<3,3>(cell.compute_jacobian_mat()).inverse();
                // const auto& grad_phi_j = DenseMatrix<3,1>(vol_grads[g][j]);
                const auto& flux = FU.multiply(J.multiply(vol_grads[g][j]));

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
        }
    }
    return result;
}


template<uInt Order, typename Flux, typename GaussQuadCell, typename GaussQuadFace>
LongVector<5*ExplicitConvection<Order, Flux, GaussQuadCell, GaussQuadFace>::N> ExplicitConvection<Order, Flux, GaussQuadCell, GaussQuadFace>::eval_internals(const ComputingMesh& mesh, 
                 const LongVector<5*N>& old_solution,
                 const Scalar curr_time){
    constexpr uInt num_face_points = QuadF::num_points;
    constexpr auto Qpoints = QuadF::get_points();
    constexpr auto Qweights = QuadF::get_weights();
    LongVector<5*N> result(old_solution.size());
    #pragma omp parallel for schedule(dynamic)
    for(uInt fid=0; fid<mesh.m_faces.size(); ++fid) {
        const auto& face = mesh.m_faces[fid];
        const auto& cells = face.m_neighbor_cells;
        if(cells[1] == uInt(-1)) continue;


        const auto& coef_L = old_solution[cells[0]];
        const auto& coef_R = old_solution[cells[1]];
        for(uInt g=0; g<num_face_points; ++g) {
            const auto& uv = Qpoints[g];
            const Scalar jac_weight = Qweights[g] * face.m_area * 2;

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

            const auto& FU_L = Flux::computeFlux(U_L);
            const auto& FU_R = Flux::computeFlux(U_R);
            const auto& FUn_L = FU_L.multiply(face.m_normal);
            const auto& FUn_R = FU_R.multiply(face.m_normal);
            const auto& Un = FUn_L[1] * face.m_normal[0] + FUn_L[2] * face.m_normal[1] + FUn_L[3] * face.m_normal[2];

            // Lax-Friedrichs参数
            const Scalar lambda = Flux::computeWaveSpeed(U_L, U_R);
            const auto& LF_flux = 0.5*(FUn_L + FUn_R + lambda*(U_L-U_R));

            for(uInt j=0; j<Basis::NumBasis; ++j) {
                const Scalar phi_jL = basis_L[j];
                const Scalar phi_jR = basis_R[j];

                
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
        }
    }
    return result;
}


