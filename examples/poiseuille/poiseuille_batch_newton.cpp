#include "base/type.h"
#include "mesh/mesh.h"
#include "matrix/matrix.h"
#include "dg/dg_basis/dg_basis.h"
#include "dg/dg_schemes/explicit_convection.h"
#include "dg/dg_schemes/explicit_diffusion.h"
#include "dg/dg_schemes/implicit_convection.h"
#include "dg/dg_schemes/implicit_diffusion.h"
#include "Eigen/Sparse"
#include "unsupported/Eigen/IterativeSolvers"

typedef Eigen::SparseMatrix<double,Eigen::RowMajor> EigenSpMat;
typedef Eigen::Triplet<double> Triplet;

#include "base/exact.h"



constexpr Scalar param_L = 1;
constexpr Scalar param_theta = 0.01;
constexpr Scalar param_Ma = 0.1;
constexpr Scalar param_gamma = 1.4;
constexpr Scalar param_mu = 1.0e-2;
constexpr Scalar param_dpdx = -8*param_mu/(param_L*param_L);


template<typename Type>
inline Type rho_xyz(Type x, Type y, Type z, Type t){
    return 1.0;
}
template<typename Type>
inline Type u_xyz(Type x, Type y, Type z, Type t){
    Type yLy = y * (param_L - y);
    Type u = -0.5 / param_mu * param_dpdx * (yLy + param_theta * yLy * yLy);
    return t>0.0? u: 0;
    // return u;
}
template<typename Type>
inline Type v_xyz(Type x, Type y, Type z, Type t){
    return 0.0;
}
template<typename Type>
inline Type w_xyz(Type x, Type y, Type z, Type t){
    return 0.0;
}
template<typename Type>
inline Type p_xyz(Type x, Type y, Type z, Type t){
    Type p = 1/(param_gamma*param_Ma*param_Ma)+param_dpdx*x;
    return t>0.0?p:1/(param_gamma*param_Ma*param_Ma);
    // return p;
}
template<typename Type>
inline Type e_xyz(Type x, Type y, Type z, Type t){
    Type p = p_xyz(x,y,z,t);
    Type u = u_xyz(x,y,z,t);
    Type v = v_xyz(x,y,z,t);
    return p/rho_xyz(x,y,z,t)/(param_gamma-1) + 0.5*(u*u+v*v);
}

template<typename Type>
inline Type fu_xyz(Type x, Type y, Type z, Type t){
    return param_dpdx*param_theta*(param_L*param_L-6*y*param_L+6*y*y);
}

template<typename Type>
inline Type fe_xyz(Type x, Type y, Type z, Type t){
    Scalar L11 = 2 * 0.25 / param_mu * param_dpdx * param_dpdx * y*(param_L - y);
    Scalar L12 = 1-param_theta*(param_L*param_L-6*y*param_L+6*y*y)-param_gamma/(param_gamma-1);
    Scalar L13 = 1+param_theta*y*(param_L - y);  //把解代进去多了一个这个，L11*L12*L13
    Scalar L21 = -0.25 / param_mu * param_dpdx * param_dpdx * (param_L - 2*y) * (param_L - 2*y);
    Scalar L22 = 2*param_theta*param_L*y-2*param_theta*y*y+1;
    return L11*L12*L13  +  L21*L22*L22;
}


#define Filed_Func(filedname) \
inline Scalar filedname##_xyz(const vector3f& xyz, Scalar t){\
    Scalar x = xyz[0], y = xyz[1], z = xyz[2];\
    return filedname##_xyz(x,y,z,t);\
}\
inline Scalar filedname##_Xi(const CompTetrahedron& cell, const vector3f& Xi, Scalar t){\
    const vector3f& xyz = cell.transform_to_physical(Xi);\
    Scalar x = xyz[0], y = xyz[1], z = xyz[2];\
    return filedname##_xyz(x,y,z,t);\
}\
inline Scalar rho##filedname##_Xi(const CompTetrahedron& cell, const vector3f& Xi, Scalar t){\
    const vector3f& xyz = cell.transform_to_physical(Xi);\
    Scalar x = xyz[0], y = xyz[1], z = xyz[2];\
    return rho_xyz(x,y,z,t)*filedname##_xyz(x,y,z,t);\
}
Filed_Func(rho);
Filed_Func(u);
Filed_Func(v);
Filed_Func(w);
Filed_Func(p);
Filed_Func(e);
Filed_Func(fu);
Filed_Func(fe);


#undef Filed_Func



template<uInt Order>
void Run(uInt N){
    auto chrono_start = std::chrono::steady_clock::now();
    auto chrone_clock = [&](){return std::chrono::duration<double>(std::chrono::steady_clock::now()-chrono_start).count();};
    auto logging = [&](std::string ss){debug("Time  " + std::to_string(chrone_clock()) + "  \tsec      " + ss);};
    debug("Start   " + std::to_string(chrone_clock()));
    GeneralMesh mesh = OrthHexMesh({0.0, 0.0, 0.0},{param_L, param_L, param_L/N},{N,N,1});
    mesh.split_hex5_scan();                                   
    mesh.rebuild_cell_topology();                             
    mesh.validate_mesh();                                     
    ComputingMesh cmesh(mesh);                                
    cmesh.m_boundary.resize(cmesh.m_faces.size());            
    for(uInt fid=0;fid<cmesh.m_faces.size();fid++){           
        if(cmesh.m_faces[fid].m_neighbor_cells[1]==uInt(-1)){ 
            const auto& face = cmesh.m_faces[fid];            
            if(std::abs(face.m_normal[2])>0.9)                
                cmesh.m_boundary[fid] = 2;
            else
                cmesh.m_boundary[fid] = 1;
        }
    }
    logging("Split Hex Mesh to Tet");

    using Basis = DGBasisEvaluator<Order>;
    using QuadC = typename AutoQuadSelector<Basis::OrderBasis, GaussLegendreTet::Auto>::type;


    LongVector<5*Basis::NumBasis> x(mesh.m_cells.size());

    #pragma omp parallel for schedule(dynamic)
    for(uInt cid=0;cid<cmesh.m_cells.size();cid++){
        const auto& cell = cmesh.m_cells[cid];
        const auto& rho_coef  = Basis::func2coef([&](vector3f Xi){return rho_Xi(cell,Xi); });
        const auto& rhou_coef = Basis::func2coef([&](vector3f Xi){return rhou_Xi(cell,Xi);});
        const auto& rhov_coef = Basis::func2coef([&](vector3f Xi){return rhov_Xi(cell,Xi);});
        const auto& rhow_coef = Basis::func2coef([&](vector3f Xi){return rhow_Xi(cell,Xi);});
        const auto& rhoe_coef = Basis::func2coef([&](vector3f Xi){return rhoe_Xi(cell,Xi);});
        #pragma omp simd
        for(uInt k=0;k<Basis::NumBasis;k++){
            x[cid][5*k+0] = rho_coef[k];
            x[cid][5*k+1] = rhou_coef[k];
            x[cid][5*k+2] = rhov_coef[k];
            x[cid][5*k+3] = rhow_coef[k];
            x[cid][5*k+4] = rhoe_coef[k];
        }
    }
    
    logging("Set Init Value");
    std::ofstream fp;
    fp.open("./Poiseuille/Order_" + std::to_string(Basis::OrderBasis) + "/rho_T_" + std::to_string(0) 
            + "_N_" + std::to_string(N) + ".txt");
    fp <<std::setprecision(16)<< ("#       x") << "  " 
        <<std::setprecision(16)<< (" y") << "  " 
        <<std::setprecision(16)<< (" z")
                << "  " <<std::setprecision(16)<<  (" rho")
                << "  " <<std::setprecision(16)<<  (" u")
                << "  " <<std::setprecision(16)<<  (" v")
                << "  " <<std::setprecision(16)<<  (" w")
                << "  " <<std::setprecision(16)<<  (" e") << std::endl;
    for(uInt cid=0;cid<cmesh.m_cells.size();cid++){
        const auto& cell = cmesh.m_cells[cid];
        for(uInt g=0; g<QuadC::num_points; ++g) {
            const auto& p = QuadC::points[g];
            const auto& pos = cell.transform_to_physical(p);
            fp <<std::setprecision(16)<< pos[0] << "  " <<std::setprecision(16)<< pos[1] << "  " <<std::setprecision(16)<< pos[2]
                << "  " <<std::setprecision(16)<<  rho_xyz(pos)
                << "  " <<std::setprecision(16)<<  u_xyz(pos)
                << "  " <<std::setprecision(16)<<  v_xyz(pos)
                << "  " <<std::setprecision(16)<<  w_xyz(pos)
                << "  " <<std::setprecision(16)<<  e_xyz(pos) << std::endl;
        }
    }
    fp.close();


    ImplicitConvection<Basis::OrderBasis> convection;
    ImplicitDiffusion<Basis::OrderBasis> diffusion;
    ImplicitDiffusion_NewtonConvection<Basis::OrderBasis> diffusion_newton;
    ExplicitDiffusion<Basis::OrderBasis> explicitdiffusion;
    
    LongVector<5*Basis::NumBasis> mass(x.size());
    for(uInt cid=0;cid<cmesh.m_cells.size();cid++){
        DenseMatrix<5*Basis::NumBasis,5*Basis::NumBasis> mass_matrix;
        for(uInt g=0; g<QuadC::num_points; ++g) {
            const auto& p = QuadC::points[g];
            auto phi = Basis::eval_all(p[0], p[1], p[2]);
            for(uInt i=0; i<Basis::NumBasis; ++i) {
                for(uInt k=0; k<5; ++k) {
                    mass[cid](5*i + k, 0) += phi[i] * phi[i] * 
                        QuadC::weights[g] * cmesh.m_cells[cid].compute_jacobian_det();
                }
            }
        }
    }


    print(std::array<std::string,8>{"#       time", "rel.err  rho",
                    "rel.err  u", "rel.err  v", "rel.err  w", 
                    "rel.err  e", "rel.err coef", "cpu time"});
    for(uInt kkkk=0;kkkk<1;kkkk++){    
        // Scalar dt = 0.001;
        
        // logging("Iter    " + std::to_string(kkkk+1));
        
        Scalar Dt = 0.5;


        Scalar max_dt = Dt;// * std::pow((5.0/N),(Order+1));
        Scalar dt = max_dt;
        // Scalar sub_dt = 0.1;
        // // debug(dt);
        Scalar sub_t = 0;
        Scalar init_delta = 0.0, init_residula = 0.0;
        do{
            dt = std::min(max_dt,Dt-sub_t);
            sub_t += dt;
            LongVector<5*Basis::NumBasis> dx = x;
            LongVector<5*Basis::NumBasis> r1 = x * 0.0;
            LongVector<5*Basis::NumBasis> r2 = x * 0.0;
            LongVector<5*Basis::NumBasis> r3 = x * 0.0;
            LongVector<5*Basis::NumBasis> r4 = x * 0.0;
            LongVector<5*Basis::NumBasis> x1 = x;
            LongVector<5*Basis::NumBasis> x2 = x;
            LongVector<5*Basis::NumBasis> x3 = x;
            LongVector<5*Basis::NumBasis> x4 = x;

            auto compute_with_u = [&](LongVector<5 * Basis::NumBasis> &dx, BlockSparseMatrix<5*Basis::NumBasis,5*Basis::NumBasis>& sparse_mat, LongVector<5*Basis::NumBasis>& rhs){
                
                BlockSparseMatrix<5*Basis::NumBasis,5*Basis::NumBasis> sparse_mat0;
                LongVector<5*Basis::NumBasis> rhs0(x.size());
                diffusion_newton.assemble(cmesh, dx, Dt * (kkkk) + sub_t + 0.5 * dt,
                                    sparse_mat0, rhs0);
                sparse_mat0.finalize();
                //print(sparse_mat0.multiply(dx).dot(sparse_mat0.multiply(dx)));
                // print(rhs0.dot(rhs0));
                // 用 保存的 dx 代入、离散
                convection.assemble(cmesh, dx, Dt * (kkkk) + sub_t + 0.5 * dt,
                                sparse_mat, rhs);

                // 在算子里面反转了一下，所以直接加法
                diffusion.assemble(cmesh, dx, Dt * (kkkk) + sub_t + 0.5 * dt,
                                    sparse_mat, rhs);
                diffusion_newton.assemble(cmesh, dx, Dt * (kkkk) + sub_t + 0.5 * dt,
                                    sparse_mat, rhs);

                rhs  = rhs + sparse_mat0.multiply(dx)-rhs0;
                // 解的是全量型，所以 delta(N1+N2+N3) delta x = -(N1+N2)
                //  delta(N1+N2+N3) delta x = -(N1+N2)
                //  delta(N1+N2+N3) x = delta(N1+N2+N3) x0 -(N1+N2)

                // LongVector<5*Basis::NumBasis> f(x.size());
                for(uInt cid=0;cid<cmesh.m_cells.size();cid++){
                    // f[cid] = 0.0;
                    for(uInt g=0; g<QuadC::num_points; ++g) {
                        const auto& p = QuadC::points[g];
                        auto phi = Basis::eval_all(p[0], p[1], p[2]);
                        auto fu = fu_Xi(cmesh.m_cells[cid],p,0.1 * (kkkk) + sub_t);
                        auto fe = fe_Xi(cmesh.m_cells[cid],p,0.1 * (kkkk) + sub_t);
                        for(uInt i=0; i<Basis::NumBasis; ++i) {
                            // for(uInt k=0; k<5; ++k) {
                            rhs[cid](5*i + 1, 0) += fu * phi[i] * QuadC::weights[g] * cmesh.m_cells[cid].compute_jacobian_det();
                            rhs[cid](5*i + 4, 0) += fe * phi[i] * QuadC::weights[g] * cmesh.m_cells[cid].compute_jacobian_det();
                            // }
                        }
                    }
                }
                // rhs += 
                sparse_mat.finalize();
            };

            for(uInt picard_iter = 0; picard_iter < 10000; picard_iter++){

                LongVector<5*Basis::NumBasis> rhs(x.size());
                BlockSparseMatrix<5*Basis::NumBasis,5*Basis::NumBasis> sparse_mat;
                
                Scalar curr_time = Dt * (kkkk) + sub_t + 0.5 * dt;
                compute_with_u(dx,sparse_mat,rhs);
                
                std::vector<Triplet> tripletList;
                uInt ELL_row_length = sparse_mat.storage.ell_blocks.size();
                uInt ELL_max_row = sparse_mat.storage.ell_max_per_row;
                uInt CSR_nnz = sparse_mat.storage.csr_blocks.size();
                uInt NNZ = (ELL_row_length * ELL_max_row + CSR_nnz) * 5*Basis::NumBasis * 5*Basis::NumBasis;
                uInt ROWs = sparse_mat.num_block_rows * 5*Basis::NumBasis;
                tripletList.reserve(NNZ);

                for(uInt brow=0; brow<sparse_mat.num_block_rows; ++brow){
                    for(uInt i=0; i<sparse_mat.storage.ell_max_per_row; ++i){
                        const uInt bcol = sparse_mat.storage.ell_cols[brow][i];
                        if(bcol == sparse_mat.invalid_index) continue;
                        const auto& block = sparse_mat.storage.ell_blocks[brow][i];
                        for(uInt row=0;row<block.rows();row++){
                            for(uInt col=0;col<block.cols();col++){
                                // if(std::abs(block(row,col))>1e-15)
                                tripletList.push_back(Triplet(brow*block.rows()+row,bcol*block.cols()+col,block(row,col)));
                            }
                        }
                    }

                    const uInt start = sparse_mat.storage.csr_row_ptr[brow];
                    const uInt end = sparse_mat.storage.csr_row_ptr[brow+1];
                    for(uInt idx = start; idx < end; ++idx) {
                        const uInt bcol = sparse_mat.storage.csr_cols[idx];
                        const auto& block = sparse_mat.storage.csr_blocks[idx];

                        for(uInt row=0;row<block.rows();row++){
                            for(uInt col=0;col<block.cols();col++){
                                // if(std::abs(block(row,col))>1e-15)
                                tripletList.push_back(Triplet(brow*block.rows()+row,bcol*block.cols()+col,block(row,col)));
                            }
                        }
                    }
                }
                EigenSpMat Eigen_A(ROWs,ROWs);
                Eigen::VectorXd Eigen_b(ROWs), Eigen_x(ROWs);




                Eigen_A.setFromTriplets(tripletList.begin(), tripletList.end());
                for(uInt r=0;r<rhs.size();r++){
                    auto& block = rhs[r];
                    for(uInt rr=0;rr<block.size();rr++){
                        Eigen_b[r * block.size() + rr] = block[rr];
                    }
                }

                // Eigen::DGMRES<EigenSpMat,Eigen::IncompleteLUT<Scalar>> Eigen_solver;
                // Eigen_solver.set_restart(5*Basis::NumBasis);
                // Eigen_solver.set_restart((5*Basis::NumBasis*3)/2);
                // Eigen_solver.setL(4); 
                // Eigen_solver.setS(4);
                Eigen::SparseLU<EigenSpMat> Eigen_solver;
                // Eigen_solver.setTolerance(1e-20);
                Eigen_solver.compute(Eigen_A);
                if(Eigen_solver.info()!=Eigen::Success) {
                    throw std::runtime_error("Matrix decomposition failed");
                }
                Eigen_x = Eigen_solver.solve(Eigen_b);
                if(Eigen_solver.info()!=Eigen::Success) {
                    throw std::runtime_error("Matrix solving failed");
                }

                // 解出来一个新的 dx
                Scalar delta = 0.0;
                for(uInt r=0;r<dx.size();r++){
                    auto& block = dx[r];
                    // auto dd = dx[r] - ddx[r];
                    // ddx[r] = dx[r];
                    for(uInt rr=0;rr<block.size();rr++){
                        Scalar xxx = block[rr] - Eigen_x[r * block.size() + rr];
                        delta += xxx*xxx;
                        block[rr] = Eigen_x[r * block.size() + rr];// + 0.5*dd[rr];
                    }
                }
                if(picard_iter==0){
                    x1 = dx;
                    LongVector<5*Basis::NumBasis> rhs(x.size());
                    BlockSparseMatrix<5*Basis::NumBasis,5*Basis::NumBasis> sparse_mat;
                    compute_with_u(x1,sparse_mat,rhs);
                    r1 = (sparse_mat.multiply(x1)) - (rhs);
                }
                if(picard_iter==1){
                    x2 = dx;
                    LongVector<5*Basis::NumBasis> rhs(x.size());
                    BlockSparseMatrix<5*Basis::NumBasis,5*Basis::NumBasis> sparse_mat;
                    compute_with_u(x2,sparse_mat,rhs);
                    r2 = (sparse_mat.multiply(x2)) - (rhs);
                }
                if(picard_iter==2){
                    x3 = dx;
                    LongVector<5*Basis::NumBasis> rhs(x.size());
                    BlockSparseMatrix<5*Basis::NumBasis,5*Basis::NumBasis> sparse_mat;
                    compute_with_u(x3,sparse_mat,rhs);
                    r3 = (sparse_mat.multiply(x3)) - (rhs);
                }
                if(picard_iter==3){
                    x4 = dx;
                    LongVector<5*Basis::NumBasis> rhs(x.size());
                    BlockSparseMatrix<5*Basis::NumBasis,5*Basis::NumBasis> sparse_mat;
                    compute_with_u(x4,sparse_mat,rhs);
                    r4 = (sparse_mat.multiply(x4)) - (rhs);
                }
                // if(picard_iter>3){
                //     LongVector<5*Basis::NumBasis> rhs(x.size());
                //     BlockSparseMatrix<5*Basis::NumBasis,5*Basis::NumBasis> sparse_mat;
                //     compute_with_u(dx,sparse_mat,rhs);
                //     const auto& x5 = dx;
                //     const auto& r5 = (sparse_mat.multiply(dx)) - (rhs);

                //     if(picard_iter > 30){

                    
                //     Eigen::MatrixXd R(4, 4);
                //     Eigen::VectorXd beta(4);
                //     R(0,0) = (r1-r2).dot(r1-r2) + 1e-16;
                //     R(0,1) = (r1-r2).dot(r2-r3);
                //     R(0,2) = (r1-r2).dot(r3-r4);
                //     R(0,3) = (r1-r2).dot(r4-r5);
                //     beta(0) = (r1 - r2).dot(r5);
                //     R(1,0) = (r2-r3).dot(r1-r2);
                //     R(1,1) = (r2-r3).dot(r2-r3) + 1e-16;
                //     R(1,2) = (r2-r3).dot(r3-r4);
                //     R(1,3) = (r2-r3).dot(r4-r5);
                //     beta(1) = (r2 - r3).dot(r5);
                //     R(2,0) = (r3-r4).dot(r1-r2);
                //     R(2,1) = (r3-r4).dot(r2-r3);
                //     R(2,2) = (r3-r4).dot(r3-r4) + 1e-16;
                //     R(2,3) = (r3-r4).dot(r4-r5);
                //     beta(2) = (r3 - r4).dot(r5);
                //     R(3,0) = (r4-r5).dot(r1-r2);
                //     R(3,1) = (r4-r5).dot(r2-r3);
                //     R(3,2) = (r4-r5).dot(r3-r4);
                //     R(3,3) = (r4-r5).dot(r4-r5) + 1e-16;
                //     beta(3) = (r4 - r5).dot(r5);

                //     Eigen::VectorXd gamma = R.fullPivLu().solve(beta);
                //     Eigen::VectorXd alpha(5);
                //     alpha.setZero();
                //     alpha.tail(4) = gamma;
                //     alpha(0) = 1.0 - gamma.sum();
                //     // if(alpha(4)<0){
                //     //     alpha(4) = 1.0;
                //     //     alpha(0) = 0.0;
                //     //     alpha(1) = 0.0;
                //     //     alpha(2) = 0.0;
                //     //     alpha(3) = 0.0;
                //     // }else
                //     for (int i = 0; i < 5; ++i) {
                //         // if(alpha(i)<0){
                //         //     alpha(4) += -alpha(i);
                //         //     alpha(i) = 0.0;
                //         // }
                //         alpha(i) = std::max(0.0, std::min(1.0, alpha(i)));
                //     }   
                //     alpha /= alpha.sum(); // 归一化
                //     dx = alpha(0)*x1 + alpha(1)*x2 + alpha(2)*x3 + alpha(3)*x4 + alpha(4)*x5;


                //     }
                //     x1 = x2;
                //     x2 = x3;
                //     x3 = x4;
                //     x4 = dx;
                //     r1 = r2;
                //     r2 = r3;
                //     r3 = r4;
                //     r4 = r5;
                // }
                // if(picard_iter>3){
                //     LongVector<5*Basis::NumBasis> rhs(x.size());
                //     BlockSparseMatrix<5*Basis::NumBasis,5*Basis::NumBasis> sparse_mat;
                //     compute_with_u(dx,sparse_mat,rhs);
                //     const auto& x5 = dx;
                //     const auto& r5 = (sparse_mat.multiply(dx)) - (rhs);

                    
                //     Eigen::MatrixXd R(2, 2);
                //     Eigen::VectorXd beta(2);
                //     R(0,0) = (r3-r4).dot(r3-r4) + 1e-16;
                //     R(0,1) = (r3-r4).dot(r4-r5);
                //     beta(0) = (r3 - r4).dot(r5);
                //     R(1,0) = (r4-r5).dot(r3-r4);
                //     R(1,1) = (r4-r5).dot(r4-r5) + 1e-16;
                //     beta(1) = -(r4 - r5).dot(r5);

                //     Eigen::VectorXd gamma = R.fullPivLu().solve(beta);
                //     Eigen::VectorXd alpha(3);
                //     alpha.setZero();
                //     alpha.tail(2) = gamma;
                //     alpha(0) = 1.0 - gamma.sum();
                //     if(alpha(2)<0){
                //         alpha(2) = 1.0;
                //         alpha(0) = 0.0;
                //         alpha(1) = 0.0;
                //     }else
                //     for (int i = 0; i < 3; ++i) {
                //         if(alpha(i)<0){
                //             alpha(2) += -alpha(i);
                //             alpha(i) = 0.0;
                //         }
                //         alpha(i) = std::max(0.0, std::min(1.0, alpha(i)));
                //     }   
                //     alpha /= alpha.sum(); // 归一化
                //     dx = alpha(0)*x3 + alpha(1)*x4 + alpha(2)*x5;

                //     x1 = x2;
                //     x2 = x3;
                //     x3 = x4;
                //     x4 = dx;
                //     r1 = r2;
                //     r2 = r3;
                //     r3 = r4;
                //     r4 = r5;
                // }

                if(picard_iter>3){
                    LongVector<5*Basis::NumBasis> rhs(x.size());
                    BlockSparseMatrix<5*Basis::NumBasis,5*Basis::NumBasis> sparse_mat;
                    compute_with_u(dx,sparse_mat,rhs);
                    const auto& x5 = dx;
                    const auto& r5 = (sparse_mat.multiply(dx)) - (rhs);
                    Scalar a = 0;
                    a = (r5-r4).dot(r5)/(r5-r4).dot(r5-r4);
                    a = std::max(a,-5.0);
                    a = std::min(a,0.1);

                    // a = -0.5;
                    
                    dx = a*x4 + (1-a)*x5;
                    // debug(a);

                    x1 = x2;
                    x2 = x3;
                    x3 = x4;
                    x4 = dx;
                    r1 = r2;
                    r2 = r3;
                    r3 = r4;
                    r4 = r5;
                }







                // const auto& xxxx = sparse_mat.multiply(dx);
                // Scalar delta = xxxx.dot(xxxx);
                delta = std::sqrt(delta);
                if(picard_iter == 0)  init_delta = delta;
                std::ostringstream oss;
                oss << delta;
                // std::ostringstream oss1;
                // oss1 << Eigen_solver.error();
                // logging("BiCGSTAB it " + std::to_string(Eigen_solver.iterations()) + "  " + oss1.str());
                Scalar residula_norm = 0.0;
                {
                    LongVector<5*Basis::NumBasis> rhs(x.size());
                    BlockSparseMatrix<5*Basis::NumBasis,5*Basis::NumBasis> sparse_mat;
                    
                    compute_with_u(dx,sparse_mat,rhs);
                    
                    
                    const auto& residual = (sparse_mat.multiply(dx)) - (rhs);
                    r4 = residual;
                    residula_norm = std::sqrt(residual.dot(residual));
                    std::ostringstream oss;
                    oss << residula_norm;
                    logging("Picard iter " + std::to_string(picard_iter) + "  " + oss.str());

                }
                
                if(picard_iter == 0)  init_residula = residula_norm;
                
                logging("Picard iter " + std::to_string(picard_iter) + "  " + oss.str());
                if(residula_norm < 1e-8 * init_residula) break;
            }
            // Picard 结束，赋值，x 为 u^{n+1}
            x = dx;
            logging("Iter  " + std::to_string(kkkk+1) + " \tSub TimeStep \t" + std::to_string(sub_t));


        }while(sub_t < Dt);






        Scalar curr_time = Dt * (kkkk+1);

        std::ofstream fp;
        fp.open("./Order_" + std::to_string(Basis::OrderBasis) + "/rho_T_" + std::to_string(kkkk+1) 
                + "_N_" + std::to_string(N) + ".txt");
        fp <<std::setprecision(16)<< ("#       x") << "  " 
            <<std::setprecision(16)<< (" y") << "  " 
            <<std::setprecision(16)<< (" z")
                 << "  " <<std::setprecision(16)<<  (" rho")
                 << "  " <<std::setprecision(16)<<  (" u")
                 << "  " <<std::setprecision(16)<<  (" v")
                 << "  " <<std::setprecision(16)<<  (" w")
                 << "  " <<std::setprecision(16)<<  (" e") << std::endl;
        auto err_integral = [&](LongVector<QuadC::num_points> uh,LongVector<QuadC::num_points> us){
            Scalar err_per_cells = 0.0, sol_per_cells = 1e-47;
            for(uInt cid=0;cid<cmesh.m_cells.size();cid++){
                const auto& cell = cmesh.m_cells[cid];
                for(uInt g=0; g<QuadC::num_points; ++g) {
                    const auto& weight = QuadC::weights[g] * cell.compute_jacobian_det();
                    Scalar error_cell = uh[cid][g] - us[cid][g];
                    err_per_cells += std::pow(error_cell,2) * weight;
                    sol_per_cells += std::pow(us[cid][g],2) * weight;
                }
            }
            return std::pow(err_per_cells/sol_per_cells,0.5);
        };


        LongVector<QuadC::num_points> rho_h(cmesh.m_cells.size());
        LongVector<QuadC::num_points> rho_s(cmesh.m_cells.size());
        LongVector<QuadC::num_points> u_h(cmesh.m_cells.size());
        LongVector<QuadC::num_points> u_s(cmesh.m_cells.size());
        LongVector<QuadC::num_points> v_h(cmesh.m_cells.size());
        LongVector<QuadC::num_points> v_s(cmesh.m_cells.size());
        LongVector<QuadC::num_points> w_h(cmesh.m_cells.size());
        LongVector<QuadC::num_points> w_s(cmesh.m_cells.size());
        LongVector<QuadC::num_points> e_h(cmesh.m_cells.size());
        LongVector<QuadC::num_points> e_s(cmesh.m_cells.size());
        LongVector<5 * Basis::NumBasis> error(cmesh.m_cells.size());

        #pragma omp parallel for schedule(dynamic)
        for(uInt cid=0;cid<cmesh.m_cells.size();cid++){
            const auto& cell = cmesh.m_cells[cid];
            // rho[cid] = 0.0;
            for(uInt g=0; g<QuadC::num_points; ++g) {
                const auto& p = QuadC::points[g];
                const auto& pos = cell.transform_to_physical(p);
                const auto& value = Basis::eval_all(p[0],p[1],p[2]);
                for(uInt k=0;k<Basis::NumBasis;k++){
                    rho_h[cid][g] += value[k] * x[cid][5*k+0];
                    u_h[cid][g] += value[k] * x[cid][5*k+1];
                    v_h[cid][g] += value[k] * x[cid][5*k+2];
                    w_h[cid][g] += value[k] * x[cid][5*k+3];
                    e_h[cid][g] += value[k] * x[cid][5*k+4];
                }
                u_h[cid][g] /= rho_h[cid][g];
                v_h[cid][g] /= rho_h[cid][g];
                w_h[cid][g] /= rho_h[cid][g];
                e_h[cid][g] /= rho_h[cid][g];
                rho_s[cid][g] = rho_xyz(pos,curr_time);
                u_s[cid][g] = u_xyz(pos,curr_time);
                v_s[cid][g] = v_xyz(pos,curr_time);
                w_s[cid][g] = w_xyz(pos,curr_time);
                e_s[cid][g] = e_xyz(pos,curr_time);
            }
            const auto& rho_coef  = Basis::func2coef([&](vector3f Xi){return rho_Xi( cell,Xi,curr_time);});
            const auto& rhou_coef = Basis::func2coef([&](vector3f Xi){return rhou_Xi(cell,Xi,curr_time);});
            const auto& rhov_coef = Basis::func2coef([&](vector3f Xi){return rhov_Xi(cell,Xi,curr_time);});
            const auto& rhow_coef = Basis::func2coef([&](vector3f Xi){return rhow_Xi(cell,Xi,curr_time);});
            const auto& rhoe_coef = Basis::func2coef([&](vector3f Xi){return rhoe_Xi(cell,Xi,curr_time);});
            #pragma omp simd
            for(uInt k=0;k<Basis::NumBasis;k++){
                error[cid][5*k+0] = x[cid][5*k+0] - rho_coef[k];
                error[cid][5*k+1] = x[cid][5*k+1] - rhou_coef[k];
                error[cid][5*k+2] = x[cid][5*k+2] - rhov_coef[k];
                error[cid][5*k+3] = x[cid][5*k+3] - rhow_coef[k];
                error[cid][5*k+4] = x[cid][5*k+4] - rhoe_coef[k];
            }
        }

        for(uInt cid=0;cid<cmesh.m_cells.size();cid++){
            const auto& cell = cmesh.m_cells[cid];
            for(uInt g=0; g<QuadC::num_points; ++g) {
                const auto& p = QuadC::points[g];
                const auto& pos = cell.transform_to_physical(p);
                fp <<std::setprecision(16)<< pos[0] << "  " <<std::setprecision(16)<< pos[1] << "  " <<std::setprecision(16)<< pos[2]
                 << "  " <<std::setprecision(16)<<  rho_h[cid][g] << "  " <<std::setprecision(16)<<  rho_s[cid][g] 
                 << "  " <<std::setprecision(16)<<  u_h[cid][g] << "  " <<std::setprecision(16)<<  u_s[cid][g] 
                 << "  " <<std::setprecision(16)<<  v_h[cid][g] << "  " <<std::setprecision(16)<<  v_s[cid][g] 
                 << "  " <<std::setprecision(16)<<  w_h[cid][g] << "  " <<std::setprecision(16)<<  w_s[cid][g] 
                 << "  " <<std::setprecision(16)<<  e_h[cid][g] << "  " <<std::setprecision(16)<<  e_s[cid][g] << std::endl;
            }
        }
        fp.close();

        print(std::array<Scalar,8>{curr_time, err_integral(rho_h,rho_s), 
                        err_integral(u_h,u_s), err_integral(v_h,v_s), err_integral(w_h,w_s), 
                        err_integral(e_h,e_s), std::sqrt(error.dot(error)/x.dot(x)), chrone_clock()});
    }
    
}

// template void Run<0>(uInt);
// template void Run<1>(uInt);
// template void Run<2>(uInt);
// template void Run<3>(uInt);
// template void Run<4>(uInt);
// template void Run<5>(uInt);
// template void Run<6>(uInt);
// template void Run<7>(uInt);
// template void Run<8>(uInt);
// template void Run<9>(uInt);



int main(int argc, char** argv){
    

    omp_set_num_threads(get_phy_cpu());
    Eigen::setNbThreads(get_phy_cpu());


    int order = std::stoi(argv[1]);
    int meshN = std::stoi(argv[2]);
                             
    // if(order == 0) Run<0>(meshN);
    if(order == 1) Run<1>(meshN);
    if(order == 2) Run<2>(meshN);
    if(order == 3) Run<3>(meshN);
    if(order == 4) Run<4>(meshN);
    if(order == 5) Run<5>(meshN);
    // if(order == 6) Run<6>(meshN);
    // if(order == 7) Run<7>(meshN);
    // if(order == 8) Run<8>(meshN);
    // if(order == 9) Run<9>(meshN);
}