#pragma once
#include "base/type.h"
#include "dg/dg_basis/dg_basis.h"
#include "mesh/mesh.h"
#include "matrix/matrix.h"
#include "base/exact.h"



template<uInt Order=3, typename GaussQuadCell = GaussLegendreTet::Auto, typename GaussQuadFace = GaussLegendreTri::Auto>
class ExplicitDiffusion {
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


    DenseMatrix<5,3> assemble_GU(const DenseMatrix<5,1>& U, const DenseMatrix<5,3>& grad_U){
        // U = {rho, rho u, rho v, rho w, rho e}
        Scalar rho = std::max(U[0],1e-8);
        Scalar u = U[1]/U[0], v = U[2]/U[0], w = U[3]/U[0], e = U[4]/U[0];
        Scalar gamma = 1.4; 
        Scalar mu = 1, Pr = 1;     // 以后改成接口传入
        Scalar Const = gamma/Pr;   // 以后改成接口传入
        // 对于标量场  s，考虑 rho*s 的梯度 grad_U 和 s 的梯度
        // grad(rho s) = grad(rho) s + rho grad(s)
        // grad(s) = 1/rho * (grad(rho s) - grad(rho) s)
        DenseMatrix<3,3> grad_uvw;
        #pragma GCC unroll 16
        for(uInt i=0;i<3;i++){
            for(uInt j=0;j<3;j++){
                grad_uvw(i,j) = 1/rho*(grad_U(i+1,j)-grad_U(0,j)*U(i+1,0)/rho);
            }
        }
        const DenseMatrix<3,3>& tau_ij = mu*(grad_uvw + grad_uvw.transpose() 
                            - 2.0/3.0*grad_uvw.trace()*DenseMatrix<3,3>::Identity());
        
        Scalar u2 = u*u + v*v + w*w;
        Scalar p = (gamma-1)*rho*(e-0.5*u2);
        // q = -mu/(Ma^2*Pr*(gamma-1))*grad(T),   p=rho T/(gamma*Ma^2) = rho(gamma-1)(e-0.5(u^2+v^2+w^2))
        // q = -mu/(Pr*(gamma-1))*grad(p*gamma/rho)
        // q = -mu*gamma/(Pr*(gamma-1))*grad((gamma-1)(e-0.5(u^2+v^2+w^2)))
        // q = -mu*gamma/(Pr)*grad(e-0.5(u^2+v^2+w^2))
        // Scalar T = (gamma-1)*(e - 0.5*(u*u + v*v + w*w));
        DenseMatrix<3,1> q = {
            -mu * Const * (1/rho*(grad_U(4,0)-grad_U(0,0)*U(4,0)/rho) - u*grad_uvw(0,0) - v*grad_uvw(1,0) - w*grad_uvw(2,0)),
            -mu * Const * (1/rho*(grad_U(4,1)-grad_U(0,1)*U(4,0)/rho) - u*grad_uvw(0,1) - v*grad_uvw(1,1) - w*grad_uvw(2,1)),
            -mu * Const * (1/rho*(grad_U(4,2)-grad_U(0,2)*U(4,0)/rho) - u*grad_uvw(0,2) - v*grad_uvw(1,2) - w*grad_uvw(2,2))
        };
        return {0,0,0,
                tau_ij(0,0),tau_ij(0,1),tau_ij(0,2),
                tau_ij(1,0),tau_ij(1,1),tau_ij(1,2),
                tau_ij(2,0),tau_ij(2,1),tau_ij(2,2),
                u*tau_ij(0,0)+v*tau_ij(1,0)+w*tau_ij(2,0)-q[0],
                u*tau_ij(0,1)+v*tau_ij(1,1)+w*tau_ij(2,1)-q[1],
                u*tau_ij(0,2)+v*tau_ij(1,2)+w*tau_ij(2,2)-q[2]};
    }

    inline std::array<std::array<DenseMatrix<5,5>,3>,3> assemble_jacobian(const DenseMatrix<5,1>& U) const{
        Scalar rho = std::max(U[0],1e-8);
        Scalar u = U[1]/U[0], v = U[2]/U[0], w = U[3]/U[0], e = U[4]/U[0];
        Scalar gamma = 1.4;
        Scalar u2 = u*u + v*v + w*w;
        Scalar mu = 1, Pr = 1;     // 以后改成接口传入
        Scalar Const = gamma/Pr;   // 以后改成接口传入
        const auto& G_fx_ux = mu/rho * DenseMatrix<5,5>({0, 0, 0, 0, 0, -4.0/3.0*u, 4.0/3.0, 0, 0, 0, -v, 0, 1, 0, 0, -w, 0, 0, 1, 0, -Const*e + std::pow(u, 2)*(Const - 4.0/3.0) + (Const - 1)*(std::pow(v, 2) + std::pow(w, 2)), u*(4.0/3.0 - Const), v*(1 - Const), w*(1 - Const), Const});
        const auto& G_fx_uy = mu/rho * DenseMatrix<5,5>({0, 0, 0, 0, 0, (2.0/3.0)*v, 0, -2.0/3.0, 0, 0, -u, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1.0/3.0*u*v, v, -2.0/3.0*u, 0, 0});
        const auto& G_fx_uz = mu/rho * DenseMatrix<5,5>({0, 0, 0, 0, 0, (2.0/3.0)*w, 0, 0, -2.0/3.0, 0, 0, 0, 0, 0, 0, -u, 1, 0, 0, 0, -1.0/3.0*u*w, w, 0, -2.0/3.0*u, 0});
        const auto& G_fy_ux = mu/rho * DenseMatrix<5,5>({0, 0, 0, 0, 0, -v, 0, 1, 0, 0, (2.0/3.0)*u, -2.0/3.0, 0, 0, 0, 0, 0, 0, 0, 0, -1.0/3.0*u*v, -2.0/3.0*v, u, 0, 0});
        const auto& G_fy_uy = mu/rho * DenseMatrix<5,5>({0, 0, 0, 0, 0, -u, 1, 0, 0, 0, -4.0/3.0*v, 0, 4.0/3.0, 0, 0, -w, 0, 0, 1, 0, -Const*e + Const*std::pow(v, 2) + Const*std::pow(w, 2) + std::pow(u, 2)*(Const - 1) - 4.0/3.0*std::pow(v, 2) - std::pow(w, 2), u*(1 - Const), v*(4.0/3.0 - Const), w*(1 - Const), Const});
        const auto& G_fy_uz = mu/rho * DenseMatrix<5,5>({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (2.0/3.0)*w, 0, 0, -2.0/3.0, 0, -v, 0, 1, 0, 0, -1.0/3.0*v*w, 0, w, -2.0/3.0*v, 0});
        const auto& G_fz_ux = mu/rho * DenseMatrix<5,5>({0, 0, 0, 0, 0, -w, 0, 0, 1, 0, 0, 0, 0, 0, 0, (2.0/3.0)*u, -2.0/3.0, 0, 0, 0, -1.0/3.0*u*w, -2.0/3.0*w, 0, u, 0});
        const auto& G_fz_uy = mu/rho * DenseMatrix<5,5>({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -w, 0, 0, 1, 0, (2.0/3.0)*v, 0, -2.0/3.0, 0, 0, -1.0/3.0*v*w, 0, -2.0/3.0*w, v, 0});
        const auto& G_fz_uz = mu/rho * DenseMatrix<5,5>({0, 0, 0, 0, 0, -u, 1, 0, 0, 0, -v, 0, 1, 0, 0, -4.0/3.0*w, 0, 0, 4.0/3.0, 0, -Const*e + Const*std::pow(v, 2) + Const*std::pow(w, 2) + std::pow(u, 2)*(Const - 1) - std::pow(v, 2) - 4.0/3.0*std::pow(w, 2), u*(1 - Const), v*(1 - Const), w*(4.0/3.0 - Const), Const});
        return std::array<std::array<DenseMatrix<5,5>,3>,3>{
            {{G_fx_ux, G_fx_uy, G_fx_uz},  {G_fy_ux, G_fy_uy, G_fy_uz},  {G_fz_ux, G_fz_uy, G_fz_uz}}};
    }

    DenseMatrix<5,5> assemble_Kij_g_v(const std::array<std::array<DenseMatrix<5,5>,3>,3>& Kij,const DenseMatrix<3,1>& grad_phi,const DenseMatrix<3,1>& vec) const{
        
        // 方案2：
        DenseMatrix<5,5> result;
        for(uInt i=0;i<3;i++)
        for(uInt j=0;j<3;j++)
        result += Kij[i][j]*grad_phi[i]*vec[j];
        return result;
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
        const Scalar rho_L = std::max(U_L[0],1e-1);
        const Scalar rho_R = std::max(U_R[0],1e-1);
        const vector3f vel_L{U_L[1]/rho_L, U_L[2]/rho_L, U_L[3]/rho_L};
        const vector3f vel_R{U_R[1]/rho_R, U_R[2]/rho_R, U_R[3]/rho_R};
        // debug(vector4f{vec_length(vel_L),vec_length(vel_R),a_L,a_R});
        return std::max(vec_length(vel_L) + a_L, vec_length(vel_R) + a_R)*1.0;
        // return 4;
    }

    Scalar compute_sound_speed(const DenseMatrix<5,1>& U) const {
        const Scalar gamma = 1.4;
        const Scalar rho = std::max(U[0],1e-1);
        
        const Scalar p = (gamma-1)*(U[4] - 0.5*(U[1]*U[1]+U[2]*U[2]+U[3]*U[3])/rho);
        // debug(vector2f{rho,p});
        return std::sqrt(std::max(gamma*p/rho, static_cast<Scalar>(1e-5)));
    }

public:
    LongVector<5*N> eval(const ComputingMesh& mesh, 
                 const LongVector<5*N>& old_solution,
                 const Scalar curr_time = 0.0){

        // 预计算 phi_i(x_g),  grad_phi_i(x_g)
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
            // val+=vol_weights[g];
        }


        #define is_debug
        #undef is_debug

        #ifdef is_debug
        uInt target = 2860;
        #endif
        
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
            
            const auto& J = DenseMatrix<3,3>(cell.compute_jacobian_mat()).inverse();
            // debug(J);
            // debug(coef);
            for(uInt g=0; g<num_vol_points; ++g) {
                // 预计算所有的grad_phi_i/j
                std::array<DenseMatrix<3,1>, Basis::NumBasis> J_grad_phi_k;
                for(uInt k=0; k<Basis::NumBasis; ++k) {
                    const auto& grad_phi_j = DenseMatrix<3,1>(vol_grads[g][k]);
                    J_grad_phi_k[k] = J.multiply(grad_phi_j);
                    // debug(J_grad_phi_k[k].transpose());
                }
                // debug(J_grad_phi_k);
                DenseMatrix<5,1> U = DenseMatrix<5,1>::Zeros();
                DenseMatrix<5,3> grad_U = DenseMatrix<5,3>::Zeros();
                for(uInt bid=0; bid<Basis::NumBasis; ++bid) {
                    const Scalar phi = vol_basis[g][bid];
                    for(uInt k=0; k<5; ++k) {
                        U[k] += phi * coef[5*bid + k];
                        grad_U(k,0) += J_grad_phi_k[bid][0] * coef[5*bid + k];
                        grad_U(k,1) += J_grad_phi_k[bid][1] * coef[5*bid + k];
                        grad_U(k,2) += J_grad_phi_k[bid][2] * coef[5*bid + k];
                        
                        
                    }
                    // debug(J_grad_phi_k[bid].transpose());
                    // debug(grad_U);
                }
                // debug(grad_U);
                


                const auto& J_G_U = assemble_jacobian(U);
                // debug(std::array<DenseMatrix<5,5>,9>{J_G_U[0][0],J_G_U[0][1],J_G_U[0][2],J_G_U[1][0],J_G_U[1][1],J_G_U[1][2],J_G_U[2][0],J_G_U[2][1],J_G_U[2][2]});
                const Scalar jac_weight = vol_weights[g] * cell.compute_jacobian_det();
                for(uInt j=0; j<Basis::NumBasis; j++) {
                    DenseMatrix<5,1> flux = DenseMatrix<5,1>::Zeros(); 
                    // 方案1：直接4重求和展开
                    for(uInt mm=0;mm<3;mm++)
                    for(uInt nn=0;nn<3;nn++)
                    for(uInt ii=0;ii<5;ii++)
                    for(uInt jj=0;jj<5;jj++)
                    flux[ii] += J_grad_phi_k[j](nn,0) *  J_G_U[mm][nn](ii,jj) * grad_U(jj,mm);
                    // 方案2：计算 Kij 单刚这种
                    // for(uInt i=0; i<Basis::NumBasis; i++) {
                    //     const auto& mat5x5 = assemble_Kij_g_v(J_G_U,J_grad_phi_k[i],J_grad_phi_k[j]);
                    //     for(uInt ii=0;ii<5;ii++)
                    //     for(uInt jj=0;jj<5;jj++)
                    //     flux(ii,0) += mat5x5(ii,jj) * coef[5*i + jj];
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
                    // debug(std::array<Scalar,5>{
                    //     result[cid](5*j+0,0),
                    //     result[cid](5*j+0,0),
                    //     result[cid](5*j+0,0),
                    //     result[cid](5*j+0,0),
                    //     result[cid](5*j+0,0)
                    // });
                }
                // print_cell_rhow(cid,1);
                // debug(result[cid]);
                
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
            // std::array<Scalar,5> cumsum = {0,0,0,0,0};
            const auto& face = mesh.m_faces[fid];
            const auto& cells = face.m_neighbor_cells;
            if(cells[1] == uInt(-1)) {
                // continue;
                const auto bc = mesh.m_boundaryTypes[fid];
                const auto& coef = old_solution[cells[0]];
                for(uInt g=0; g<QuadF::num_points; ++g) {
                    const auto& xi = transform_to_cell(face, QuadF::points[g], 0);
                    const auto& basis = Basis::eval_all(xi[0], xi[1], xi[2]);
                    const auto& grad_basis = Basis::grad_all(xi[0], xi[1], xi[2]);
                    const Scalar jac_weight = QuadF::weights[g] * face.compute_jacobian_det();
                    // 预计算所有的grad_phi_i/j
                    std::array<DenseMatrix<3,1>, Basis::NumBasis> J_grad_phi_k;
                    const auto& J = DenseMatrix<3,3>(mesh.m_cells[cells[0]].compute_jacobian_mat()).inverse();
                    for(uInt k=0; k<Basis::NumBasis; ++k) {
                        const auto& grad_phi_j = DenseMatrix<3,1>(grad_basis[k]);
                        J_grad_phi_k[k] = J.multiply(grad_phi_j);
                    }
                    DenseMatrix<5,1> U_inner;
                    DenseMatrix<5,3> grad_U_inner;
                    for(uInt bid=0; bid<Basis::NumBasis; ++bid) {
                        for(uInt k=0; k<5; ++k){
                            U_inner[k] += basis[bid] * coef[5*bid + k];
                            grad_U_inner(k,0) += J_grad_phi_k[bid][0] * coef[5*bid + k];
                            grad_U_inner(k,1) += J_grad_phi_k[bid][1] * coef[5*bid + k];
                            grad_U_inner(k,2) += J_grad_phi_k[bid][2] * coef[5*bid + k];
                        }
                    }
                    // auto U_ghost = (bc==BoundaryType::Dirichlet)?
                    //                 2*DenseMatrix<5,1>({1,1,0.5,0,2.5+0.625})-U_inner:   //Dirichlet     下面Neumann
                    //                 U_inner + 2*0.0*vec_dot(vector3f{0,0,0},face.m_normal);
                    
                // debug(FU);
                    if (bc==BoundaryType::Dirichlet) {
                        const auto& U_D = DenseMatrix<5,1>({rho_Xi(mesh.m_cells[cells[0]],xi,curr_time),
                                                            rhou_Xi(mesh.m_cells[cells[0]],xi,curr_time),
                                                            rhov_Xi(mesh.m_cells[cells[0]],xi,curr_time),
                                                            rhow_Xi(mesh.m_cells[cells[0]],xi,curr_time),
                                                            rhoe_Xi(mesh.m_cells[cells[0]],xi,curr_time)});
                        const auto& U_ghost = U_D + (U_D - U_inner);
                        
                        // K[{Q}]
                        const auto& J_G_U = assemble_jacobian(0.5*(U_inner+U_ghost));  
                        const auto& normal = DenseMatrix<3,1>(face.m_normal);

                        // phi_j (\mathbf{n} K[{Q}] grad_Q)
                        DenseMatrix<5,1> flux = DenseMatrix<5,1>::Zeros(); 
                        for(uInt i=0; i<Basis::NumBasis; i++) {
                            const auto& mat5x5 = assemble_Kij_g_v(J_G_U,J_grad_phi_k[i],normal);
                            for(uInt ii=0;ii<5;ii++)
                            for(uInt jj=0;jj<5;jj++)
                            flux(ii,0) += mat5x5(ii,jj) * coef[5*i + jj];
                        }

                        // \mathbf{n} K[{Q}] [[Q]]
                        // Scalar h = std::pow(mesh.m_cells[cells[0]].m_volume,1.0/3.0);
                        Scalar beta_p = (Order+1)*(Order+1)/mesh.m_cells[cells[0]].m_h;
                        for(uInt i=0; i<Basis::NumBasis; i++) {
                            const auto& mat5x5 = assemble_Kij_g_v(J_G_U,basis[i]*normal,normal);
                            for(uInt ii=0;ii<5;ii++)
                            for(uInt jj=0;jj<5;jj++)
                            flux(ii,0) += -mat5x5(ii,jj) * coef[5*i + jj]  * beta_p;
                        }

                        for(uInt j=0; j<Basis::NumBasis; ++j) {
                            #pragma omp atomic update
                            result[cells[0]](5*j+0,0) += basis[j]*flux(0,0) * jac_weight;
                            #pragma omp atomic update
                            result[cells[0]](5*j+1,0) += basis[j]*flux(1,0) * jac_weight;
                            #pragma omp atomic update
                            result[cells[0]](5*j+2,0) += basis[j]*flux(2,0) * jac_weight;
                            #pragma omp atomic update
                            result[cells[0]](5*j+3,0) += basis[j]*flux(3,0) * jac_weight;
                            #pragma omp atomic update
                            result[cells[0]](5*j+4,0) += basis[j]*flux(4,0) * jac_weight;
                        }

                        for(uInt j=0; j<Basis::NumBasis; ++j) {
                            // grad_Q K[{Q}] \mathbf{n}
                            DenseMatrix<5,1> flux = DenseMatrix<5,1>::Zeros(); 
                            const auto& mat = assemble_Kij_g_v(J_G_U,J_grad_phi_k[j],normal);
                            for(uInt ii=0;ii<5;ii++)
                            for(uInt jj=0;jj<5;jj++)
                            flux(ii,0) = mat(ii,jj)*(U_ghost(jj,0)*1.0- U_inner(jj,0));
                            
                            #pragma omp atomic update
                            result[cells[0]](5*j+0,0) += flux(0,0) * jac_weight;
                            #pragma omp atomic update
                            result[cells[0]](5*j+1,0) += flux(1,0) * jac_weight;
                            #pragma omp atomic update
                            result[cells[0]](5*j+2,0) += flux(2,0) * jac_weight;
                            #pragma omp atomic update
                            result[cells[0]](5*j+3,0) += flux(3,0) * jac_weight;
                            #pragma omp atomic update
                            result[cells[0]](5*j+4,0) += flux(4,0) * jac_weight;
                        }

                        
                        // print_cell_rhow(cells[0],2);
                        // debug(result[cells[0]]);
                    }
                    else if(bc==BoundaryType::Pseudo3DZ) {
                        if(0){
                            // 把伪三维问题，视为是上下边界 U cdot n = 0 去做
                            const auto& Un_N = vec_dot(vector3f{0,0,0},face.m_normal);

                        }
                        else{
                            // 把伪三维问题，视为是上下边界 (rho w)_{ghost} = -(rho w)_{inner}
                            const auto& U_ghost = U_inner * DenseMatrix<5,1>({1,1,1,-1,1});


                            // 复制的DIrichlet，不确定是否对
                            // K[{Q}]
                            const auto& J_G_U = assemble_jacobian(0.5*(U_inner+U_ghost));  
                            const auto& normal = DenseMatrix<3,1>(face.m_normal);


                            // phi_j (\mathbf{n} K[{Q}] grad_Q)

                            DenseMatrix<5,1> flux = DenseMatrix<5,1>::Zeros(); 
                            for(uInt i=0; i<Basis::NumBasis; i++) {
                                const auto& mat5x5 = assemble_Kij_g_v(J_G_U,J_grad_phi_k[i],normal);
                                for(uInt ii=0;ii<5;ii++)
                                for(uInt jj=0;jj<5;jj++)
                                flux(ii,0) += mat5x5(ii,jj) * coef[5*i + jj];
                            }

                            // phi_j \mathbf{n} K[{Q}] [[Q]]
                            // Scalar h = std::pow(mesh.m_cells[cells[0]].m_volume,1.0/3.0);
                            Scalar beta_p = (Order+1)*(Order+1)/mesh.m_cells[cells[0]].m_h;
                            for(uInt i=0; i<Basis::NumBasis; i++) {
                                const auto& mat5x5 = assemble_Kij_g_v(J_G_U,basis[i]*normal,normal);
                                for(uInt ii=0;ii<5;ii++)
                                for(uInt jj=0;jj<5;jj++)
                                flux(ii,0) += -mat5x5(ii,jj) * coef[5*i + jj]  * beta_p;
                            }
                            // debug(flux);
                            for(uInt j=0; j<Basis::NumBasis; ++j) {
                                
                                #ifdef is_debug
                                if(cells[0] == target){
                                    cumsum[0] = 2;
                                    cumsum[1+j] += LF_flux[3] * phi_j * jac_weight;
                                }
                                #endif
                                // #pragma omp critical
                                // MatrixView<5*N,1,5,1>(result[cells[0]],5*j,0) += LF_flux * phi_j * jac_weight;
                            
                                #pragma omp atomic update
                                result[cells[0]](5*j+0,0) += basis[j]*flux(0,0) * jac_weight;
                                #pragma omp atomic update
                                result[cells[0]](5*j+1,0) += basis[j]*flux(1,0) * jac_weight;
                                #pragma omp atomic update
                                result[cells[0]](5*j+2,0) += basis[j]*flux(2,0) * jac_weight;
                                #pragma omp atomic update
                                result[cells[0]](5*j+3,0) += basis[j]*flux(3,0) * jac_weight;
                                #pragma omp atomic update
                                result[cells[0]](5*j+4,0) += basis[j]*flux(4,0) * jac_weight;
                            }

                            
                            for(uInt j=0; j<Basis::NumBasis; ++j) {
                                // grad_Q K[{Q}] \mathbf{n}
                                DenseMatrix<5,1> flux = DenseMatrix<5,1>::Zeros(); 
                                const auto& mat = assemble_Kij_g_v(J_G_U,J_grad_phi_k[j],normal);
                                for(uInt ii=0;ii<5;ii++)
                                for(uInt jj=0;jj<5;jj++)
                                flux(ii,0) = mat(ii,jj)*(U_ghost(jj,0)*1.0-U_inner(jj,0));
                                
                                // debug(flux);
                                #pragma omp atomic update
                                result[cells[0]](5*j+0,0) += flux(0,0) * jac_weight;
                                #pragma omp atomic update
                                result[cells[0]](5*j+1,0) += flux(1,0) * jac_weight;
                                #pragma omp atomic update
                                result[cells[0]](5*j+2,0) += flux(2,0) * jac_weight;
                                #pragma omp atomic update
                                result[cells[0]](5*j+3,0) += flux(3,0) * jac_weight;
                                #pragma omp atomic update
                                result[cells[0]](5*j+4,0) += flux(4,0) * jac_weight;
                            }
                        }
                        // debug(result[cells[0]]);
                    }
                }
            }
            else{
                // continue;
                const auto& coef_L = old_solution[cells[0]];
                const auto& coef_R = old_solution[cells[1]];
                for(uInt g=0; g<QuadF::num_points; ++g) {

                    const auto& xi_L = transform_to_cell(face, QuadF::points[g], 0);
                    const auto& xi_R = transform_to_cell(face, QuadF::points[g], 1);
                    const auto& basis_L = Basis::eval_all(xi_L[0], xi_L[1], xi_L[2]);
                    const auto& basis_R = Basis::eval_all(xi_R[0], xi_R[1], xi_R[2]);
                    const auto& grad_basis_L = Basis::grad_all(xi_L[0], xi_L[1], xi_L[2]);
                    const auto& grad_basis_R = Basis::grad_all(xi_R[0], xi_R[1], xi_R[2]);
                    const Scalar jac_weight = QuadF::weights[g] * face.compute_jacobian_det();

                    // 预计算所有的grad_phi_i/j
                    std::array<DenseMatrix<3,1>, Basis::NumBasis> J_grad_phi_k_L,J_grad_phi_k_R;
                    const auto& J_L = DenseMatrix<3,3>(mesh.m_cells[cells[0]].compute_jacobian_mat()).inverse();
                    const auto& J_R = DenseMatrix<3,3>(mesh.m_cells[cells[1]].compute_jacobian_mat()).inverse();
                    for(uInt k=0; k<Basis::NumBasis; ++k) {
                        const auto& grad_phi_j_L = DenseMatrix<3,1>(grad_basis_L[k]);
                        const auto& grad_phi_j_R = DenseMatrix<3,1>(grad_basis_R[k]);
                        J_grad_phi_k_L[k] = J_L.multiply(grad_phi_j_L);
                        J_grad_phi_k_R[k] = J_R.multiply(grad_phi_j_R);
                    }

                    DenseMatrix<5,1> U_L, U_R;
                    DenseMatrix<5,3> grad_U_L;
                    DenseMatrix<5,3> grad_U_R;
                    for(uInt bid=0; bid<Basis::NumBasis; ++bid) {
                        for(uInt k=0; k<5; ++k) {
                            U_L[k] += basis_L[bid] * coef_L[5*bid + k];
                            U_R[k] += basis_R[bid] * coef_R[5*bid + k];
                            grad_U_L(k,0) += J_grad_phi_k_L[bid][0] * coef_L[5*bid + k];
                            grad_U_L(k,1) += J_grad_phi_k_L[bid][1] * coef_L[5*bid + k];
                            grad_U_L(k,2) += J_grad_phi_k_L[bid][2] * coef_L[5*bid + k];
                            grad_U_R(k,0) += J_grad_phi_k_R[bid][0] * coef_R[5*bid + k];
                            grad_U_R(k,1) += J_grad_phi_k_R[bid][1] * coef_R[5*bid + k];
                            grad_U_R(k,2) += J_grad_phi_k_R[bid][2] * coef_R[5*bid + k];
                        }
                    }

                    // K[{Q}]
                    const auto& J_G_U = assemble_jacobian(0.5*(U_L+U_R));  
                    const auto& normal = DenseMatrix<3,1>(face.m_normal);

                    // phi_j (\mathbf{n} K[{Q}] grad_Q)
                    // DenseMatrix<5,1> flux_L = DenseMatrix<5,1>::Zeros(); 
                    // DenseMatrix<5,1> flux_R = DenseMatrix<5,1>::Zeros(); 
                    DenseMatrix<5,1> flux = DenseMatrix<5,1>::Zeros(); 
                    for(uInt i=0; i<Basis::NumBasis; i++) {
                        const auto& mat5x5_L = assemble_Kij_g_v(J_G_U,J_grad_phi_k_L[i],normal);
                        const auto& mat5x5_R = assemble_Kij_g_v(J_G_U,J_grad_phi_k_R[i],normal);
                        
                        for(uInt ii=0;ii<5;ii++)
                        for(uInt jj=0;jj<5;jj++){
                            const auto& mat5x5 = 0.5*(mat5x5_L(ii,jj)*coef_L[5*i + jj] + mat5x5_R(ii,jj)*coef_R[5*i + jj]);
                            // flux_L(ii,0) += mat5x5;
                            // flux_R(ii,0) += mat5x5;
                            flux(ii,0) += mat5x5;
                        }
                    }

                    // phi_j \mathbf{n} K[{Q}] [[Q]]
                    // Scalar h = std::pow(mesh.m_cells[cells[0]].m_volume,1.0/3.0);
                    
                    Scalar beta_p = (Order+1)*(Order+1)/mesh.m_cells[cells[0]].m_h;
                    for(uInt i=0; i<Basis::NumBasis; i++) {
                        const auto& mat5x5_L = assemble_Kij_g_v(J_G_U,basis_L[i]*normal,normal);
                        const auto& mat5x5_R = assemble_Kij_g_v(J_G_U,basis_R[i]*normal,normal);
                        
                        for(uInt ii=0;ii<5;ii++)
                        for(uInt jj=0;jj<5;jj++){
                            const auto& mat5x5 = -1.0*(mat5x5_L(ii,jj)*coef_L[5*i + jj] - mat5x5_R(ii,jj)*coef_R[5*i + jj]);
                            // flux_L(ii,0) += mat5x5 * beta_p;
                            // flux_R(ii,0) += mat5x5 * beta_p;
                            flux(ii,0) += mat5x5 * beta_p;
                        }
                    }

                    for(uInt j=0; j<Basis::NumBasis; ++j) {
                        // debug(basis_L[j] * flux_L * jac_weight);
                        // debug(basis_R[j] * flux_R * jac_weight);
                        #pragma omp atomic update
                        result[cells[0]](5*j+0,0) += basis_L[j] * flux(0,0) * jac_weight;
                        #pragma omp atomic update
                        result[cells[0]](5*j+1,0) += basis_L[j] * flux(1,0) * jac_weight;
                        #pragma omp atomic update
                        result[cells[0]](5*j+2,0) += basis_L[j] * flux(2,0) * jac_weight;
                        #pragma omp atomic update
                        result[cells[0]](5*j+3,0) += basis_L[j] * flux(3,0) * jac_weight;
                        #pragma omp atomic update
                        result[cells[0]](5*j+4,0) += basis_L[j] * flux(4,0) * jac_weight;
                        #pragma omp atomic update
                        result[cells[1]](5*j+0,0) -= basis_R[j] * flux(0,0) * jac_weight;
                        #pragma omp atomic update
                        result[cells[1]](5*j+1,0) -= basis_R[j] * flux(1,0) * jac_weight;
                        #pragma omp atomic update
                        result[cells[1]](5*j+2,0) -= basis_R[j] * flux(2,0) * jac_weight;
                        #pragma omp atomic update
                        result[cells[1]](5*j+3,0) -= basis_R[j] * flux(3,0) * jac_weight;
                        #pragma omp atomic update
                        result[cells[1]](5*j+4,0) -= basis_R[j] * flux(4,0) * jac_weight;
                    }


                    for(uInt j=0; j<Basis::NumBasis; ++j) {
                        // grad_Q K[{Q}] \mathbf{n}
                        // DenseMatrix<5,1> flux_L = DenseMatrix<5,1>::Zeros(); 
                        // DenseMatrix<5,1> flux_R = DenseMatrix<5,1>::Zeros(); 
                        DenseMatrix<5,1> flux = DenseMatrix<5,1>::Zeros(); 
                        const auto& mat = 0.5*(
                            assemble_Kij_g_v(J_G_U,J_grad_phi_k_L[j],normal)+
                            assemble_Kij_g_v(J_G_U,J_grad_phi_k_R[j],normal)
                        );
                        for(uInt ii=0;ii<5;ii++)
                        for(uInt jj=0;jj<5;jj++){
                            // flux_L(ii,0) = mat(ii,jj)*U_L(jj,0);
                            // flux_R(ii,0) = mat(ii,jj)*U_R(jj,0);
                            flux(ii,0) = mat(ii,jj)*(U_R(jj,0)-U_L(jj,0));
                        }
                        

                        #ifdef is_debug
                        if(cells[0] == target){
                            cumsum[0] = 2;
                            cumsum[1+j] += LF_flux[3] * phi_j * jac_weight;
                        }
                        #endif
                        // #pragma omp critical
                        // MatrixView<5*N,1,5,1>(result[cells[0]],5*j,0) += LF_flux * phi_j * jac_weight;
                        
                        // debug(flux_L * jac_weight);
                        // debug(flux_R * jac_weight);
                        #pragma omp atomic update
                        result[cells[0]](5*j+0,0) += flux(0,0) * jac_weight;
                        #pragma omp atomic update
                        result[cells[0]](5*j+1,0) += flux(1,0) * jac_weight;
                        #pragma omp atomic update
                        result[cells[0]](5*j+2,0) += flux(2,0) * jac_weight;
                        #pragma omp atomic update
                        result[cells[0]](5*j+3,0) += flux(3,0) * jac_weight;
                        #pragma omp atomic update
                        result[cells[0]](5*j+4,0) += flux(4,0) * jac_weight;
                        #pragma omp atomic update
                        result[cells[1]](5*j+0,0) -= flux(0,0) * jac_weight;
                        #pragma omp atomic update
                        result[cells[1]](5*j+1,0) -= flux(1,0) * jac_weight;
                        #pragma omp atomic update
                        result[cells[1]](5*j+2,0) -= flux(2,0) * jac_weight;
                        #pragma omp atomic update
                        result[cells[1]](5*j+3,0) -= flux(3,0) * jac_weight;
                        #pragma omp atomic update
                        result[cells[1]](5*j+4,0) -= flux(4,0) * jac_weight;
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
        return result;
    }

};