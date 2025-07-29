#include "base/type.h"
#include "mesh/mesh.h"
#include "matrix/matrix.h"
#include "dg/dg_basis/dg_basis.h"
// #include "dg/dg_schemes/explicit_convection.h"
// #include "dg/dg_schemes/explicit_diffusion.h"
#include "dg/dg_schemes/implicit_convection.h"
#include "dg/dg_schemes/implicit_diffusion.h"
// #include "EigenSolver/EigenSparseSolver.h"
#include "Eigen/Sparse"
#include "unsupported/Eigen/IterativeSolvers"

typedef Eigen::SparseMatrix<double,Eigen::RowMajor> EigenSpMat;
typedef Eigen::Triplet<double> Triplet;



string sol_filename(uInt Order, uInt Time, uInt Nmesh){
    return "./Order_" + std::to_string(Order) 
            + "/solution/T_" + std::to_string(Time) 
            + "_N_" + std::to_string(Nmesh) + ".txt";
}
                

#include "problem.h"
ComputingMesh create_mesh(uInt N){
    GeneralMesh mesh = OrthHexMesh({0.0, 0.0, 0.0},{param_L, param_L, param_L/N},{N,N,1});
    mesh.split_hex5_scan();                                   
    mesh.rebuild_cell_topology();                             
    mesh.validate_mesh();                                     
    ComputingMesh cmesh(mesh);                                
    cmesh.m_boundaryTypes.resize(cmesh.m_faces.size());            
    for(uInt faceId=0;faceId<cmesh.m_faces.size();faceId++){           
        if(cmesh.m_faces[faceId].m_neighbor_cells[1]==uInt(-1)){ 
            const auto& face = cmesh.m_faces[faceId];            
            if(std::abs(face.m_normal[2])>0.9)                
                cmesh.m_boundaryTypes[faceId] = BoundaryType::Pseudo3DZ;
            else
                cmesh.m_boundaryTypes[faceId] = BoundaryType::Dirichlet;
        }
    }
    return cmesh;
}



template<uInt Order>
void Run(uInt N){
    auto chrono_start = std::chrono::steady_clock::now();
    auto chrone_clock = [&](){return std::chrono::duration<double>(std::chrono::steady_clock::now()-chrono_start).count();};
    auto logging = [&](std::string ss){debug("Time  " + std::to_string(chrone_clock()) + "  \tsec      " + ss);};
    debug("Start   " + std::to_string(chrone_clock()));

    const auto& cmesh = create_mesh(N);
    logging("Split Hex Mesh to Tet");

    using Basis = DGBasisEvaluator<Order>;
    using QuadC = typename AutoQuadSelector<Basis::OrderBasis, GaussLegendreTet::Auto>::type;
    constexpr uInt DoFs = 5*Basis::NumBasis;


    LongVector<DoFs> x(cmesh.m_cells.size());

    #pragma omp parallel for schedule(dynamic)
    for(uInt cellId=0;cellId<cmesh.m_cells.size();cellId++){
        const auto& cell = cmesh.m_cells[cellId];
        const auto& rhoU_coef = Basis::func2coef([&](vector3f Xi)->DenseMatrix<5,1>{
            return {rho_Xi(cell,Xi),rhou_Xi(cell,Xi),rhov_Xi(cell,Xi),rhow_Xi(cell,Xi),rhoe_Xi(cell,Xi)};
        });
        for(uInt k=0;k<Basis::NumBasis;k++){
            MatrixView<DoFs,1,5,1>(x[cellId],5*k,0) = rhoU_coef[k];
        }
    }
    
    logging("Set Init Value");
    std::ofstream fp;
    fp.open(sol_filename(Basis::OrderBasis, 0, N));
    fp <<std::setprecision(16)<< ("#       x") << "  " 
        <<std::setprecision(16)<< (" y") << "  " 
        <<std::setprecision(16)<< (" z")
                << "  " <<std::setprecision(16)<<  (" rho")
                << "  " <<std::setprecision(16)<<  (" u")
                << "  " <<std::setprecision(16)<<  (" v")
                << "  " <<std::setprecision(16)<<  (" w")
                << "  " <<std::setprecision(16)<<  (" e") << std::endl;
    for(uInt cellId=0;cellId<cmesh.m_cells.size();cellId++){
        const auto& cell = cmesh.m_cells[cellId];
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
    // ImplicitDiffusion_NewtonConvection<Basis::OrderBasis> diffusion_newton;
    // ExplicitDiffusion<Basis::OrderBasis> explicitdiffusion;
    
    LongVector<DoFs> mass(x.size());
    for(uInt cellId=0;cellId<cmesh.m_cells.size();cellId++){
        DenseMatrix<DoFs,DoFs> mass_matrix;
        for(uInt g=0; g<QuadC::num_points; ++g) {
            const auto& p = QuadC::points[g];
            auto phi = Basis::eval_all(p[0], p[1], p[2]);
            for(uInt i=0; i<Basis::NumBasis; ++i) {
                for(uInt k=0; k<5; ++k) {
                    mass[cellId](5*i + k, 0) += phi[i] * phi[i] * 
                        QuadC::weights[g] * cmesh.m_cells[cellId].compute_jacobian_det();
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
            LongVector<DoFs> dx = x;
            // LongVector<DoFs> ddx = x;
            LongVector<DoFs> r4 = x * 0.0;
            LongVector<DoFs> x4 = x;

            auto compute_with_u = [&](LongVector<5 * Basis::NumBasis> &dx, BlockSparseMatrix<DoFs,DoFs>& sparse_mat, LongVector<DoFs>& rhs){
                
                // BlockSparseMatrix<DoFs,DoFs> sparse_mat0;
                // LongVector<DoFs> rhs0(x.size());
                // diffusion_newton.assemble(cmesh, dx, Dt * (kkkk) + sub_t + 0.5 * dt,
                //                     sparse_mat0, rhs0);
                // sparse_mat0.finalize();
                //print(sparse_mat0.multiply(dx).dot(sparse_mat0.multiply(dx)));
                // print(rhs0.dot(rhs0));
                // 用 保存的 dx 代入、离散
                convection.assemble(cmesh, dx, Dt * (kkkk) + sub_t + 0.5 * dt,
                                sparse_mat, rhs);

                // 在算子里面反转了一下，所以直接加法
                diffusion.assemble(cmesh, dx, Dt * (kkkk) + sub_t + 0.5 * dt,
                                    sparse_mat, rhs);
                // diffusion_newton.assemble(cmesh, dx, Dt * (kkkk) + sub_t + 0.5 * dt,
                //                     sparse_mat, rhs);

                // rhs  = rhs + sparse_mat0.multiply(dx)-rhs0;
                // 解的是全量型，所以 delta(N1+N2+N3) delta x = -(N1+N2)
                //  delta(N1+N2+N3) delta x = -(N1+N2)
                //  delta(N1+N2+N3) x = delta(N1+N2+N3) x0 -(N1+N2)

                // LongVector<DoFs> f(x.size());
                for(uInt cellId=0;cellId<cmesh.m_cells.size();cellId++){
                    // f[cellId] = 0.0;
                    for(uInt g=0; g<QuadC::num_points; ++g) {
                        const auto& p = QuadC::points[g];
                        auto phi = Basis::eval_all(p[0], p[1], p[2]);
                        auto fu = fu_Xi(cmesh.m_cells[cellId],p,0.1 * (kkkk) + sub_t);
                        auto fe = fe_Xi(cmesh.m_cells[cellId],p,0.1 * (kkkk) + sub_t);
                        for(uInt i=0; i<Basis::NumBasis; ++i) {
                            // for(uInt k=0; k<5; ++k) {
                            rhs[cellId](5*i + 1, 0) += fu * phi[i] * QuadC::weights[g] * cmesh.m_cells[cellId].compute_jacobian_det();
                            rhs[cellId](5*i + 4, 0) += fe * phi[i] * QuadC::weights[g] * cmesh.m_cells[cellId].compute_jacobian_det();
                            // }
                        }
                    }
                }
                // rhs += 
                sparse_mat.finalize();
            };

            for(uInt picard_iter = 0; picard_iter < 10000; picard_iter++){
                // debug("第零步或者新一轮");
                LongVector<DoFs> rhs(x.size());
                BlockSparseMatrix<DoFs,DoFs> sparse_mat;
                
                Scalar curr_time = Dt * (kkkk) + sub_t + 0.5 * dt;
                compute_with_u(dx,sparse_mat,rhs);

                // print(sparse_mat.multiply(x).dot(sparse_mat.multiply(x)));
                // print(rhs.dot(rhs));

                // EigenSparseSolver<DoFs,DoFs> solver(sparse_mat,rhs);
                // debug("第一步成功");
                // LongVector<DoFs> ddx = solver.SparseLU(dx);
                // debug("第二步成功");


                // 添加限制器
                // dx 是上一步，或者上一时刻
                // ddx 用来修正
                // 最后 dx 被 ddx 覆盖 
                
                
                std::vector<Triplet> tripletList;
                uInt ELL_row_length = sparse_mat.storage.ell_blocks.size();
                uInt ELL_max_row = sparse_mat.storage.ell_max_per_row;
                uInt CSR_nnz = sparse_mat.storage.csr_blocks.size();
                uInt NNZ = (ELL_row_length * ELL_max_row + CSR_nnz) * DoFs * DoFs;
                uInt ROWs = sparse_mat.num_block_rows * DoFs;
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
                // Eigen_solver.set_restart(DoFs);
                // Eigen_solver.set_restart((DoFs*3)/2);
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
                // Scalar delta = 0.0;
                LongVector<DoFs> ddx(dx.size());
                for(uInt r=0;r<dx.size();r++){
                    auto& block = ddx[r];
                    // auto dd = dx[r] - ddx[r];
                    // ddx[r] = dx[r];
                    for(uInt rr=0;rr<block.size();rr++){
                        // Scalar xxx = block[rr] - Eigen_x[r * block.size() + rr];
                        // delta += xxx*xxx;
                        block[rr] = Eigen_x[r * block.size() + rr];// + 0.5*dd[rr];
                    }
                }

                if(picard_iter==3){
                    x4 = ddx;
                    LongVector<DoFs> rhs(x.size());
                    BlockSparseMatrix<DoFs,DoFs> sparse_mat;
                    compute_with_u(x4,sparse_mat,rhs);
                    r4 = (sparse_mat.multiply(x4)) - (rhs);
                }


                if(picard_iter>3){
                    LongVector<DoFs> rhs(x.size());
                    BlockSparseMatrix<DoFs,DoFs> sparse_mat;
                    compute_with_u(ddx,sparse_mat,rhs);
                    const auto& x5 = ddx;
                    const auto& r5 = (sparse_mat.multiply(ddx)) - (rhs);
                    Scalar a = 0;
                    a = (r5-r4).dot(r5)/(r5-r4).dot(r5-r4);
                    a = std::max(a,-5.0);
                    a = std::min(a,0.1);

                    // a = -0.5;
                    
                    ddx = a*x4 + (1-a)*x5;
                    // debug(a);

                    x4 = ddx;
                    r4 = r5;
                }



                // debug("这是第三步，做完了Anderson加速");



                // const auto& xxxx = sparse_mat.multiply(dx);
                // Scalar delta = xxxx.dot(xxxx);

                const auto& delta_x = ddx - dx;
                dx = ddx;
                Scalar delta = delta_x.dot(delta_x);
                delta = std::sqrt(delta);
                if(picard_iter == 0)  init_delta = delta;
                std::ostringstream oss;
                oss << delta;
                // std::ostringstream oss1;
                // oss1 << Eigen_solver.error();
                // logging("BiCGSTAB it " + std::to_string(Eigen_solver.iterations()) + "  " + oss1.str());
                Scalar residula_norm = 0.0;
                {
                    LongVector<DoFs> rhs(x.size());
                    BlockSparseMatrix<DoFs,DoFs> sparse_mat;
                    
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
                // debug("这是第四步，准备判断是否退出");
                if(residula_norm < 1e-8 * init_residula) break;
                // debug("这是第五步，还没有退出");
            }
            // Picard 结束，赋值，x 为 u^{n+1}
            x = dx;
            logging("Iter  " + std::to_string(kkkk+1) + " \tSub TimeStep \t" + std::to_string(sub_t));


        }while(sub_t < Dt);






        Scalar curr_time = Dt * (kkkk+1);

        std::ofstream fp;
        fp.open(sol_filename(Basis::OrderBasis, kkkk+1, N));
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
            for(uInt cellId=0;cellId<cmesh.m_cells.size();cellId++){
                const auto& cell = cmesh.m_cells[cellId];
                for(uInt g=0; g<QuadC::num_points; ++g) {
                    const auto& weight = QuadC::weights[g] * cell.compute_jacobian_det();
                    Scalar error_cell = uh[cellId][g] - us[cellId][g];
                    err_per_cells += std::pow(error_cell,2) * weight;
                    sol_per_cells += std::pow(us[cellId][g],2) * weight;
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
        for(uInt cellId=0;cellId<cmesh.m_cells.size();cellId++){
            const auto& cell = cmesh.m_cells[cellId];
            // rho[cellId] = 0.0;
            for(uInt g=0; g<QuadC::num_points; ++g) {
                const auto& p = QuadC::points[g];
                const auto& pos = cell.transform_to_physical(p);
                const auto& value = Basis::eval_all(p[0],p[1],p[2]);
                for(uInt k=0;k<Basis::NumBasis;k++){
                    rho_h[cellId][g] += value[k] * x[cellId][5*k+0];
                    u_h[cellId][g] += value[k] * x[cellId][5*k+1];
                    v_h[cellId][g] += value[k] * x[cellId][5*k+2];
                    w_h[cellId][g] += value[k] * x[cellId][5*k+3];
                    e_h[cellId][g] += value[k] * x[cellId][5*k+4];
                }
                u_h[cellId][g] /= rho_h[cellId][g];
                v_h[cellId][g] /= rho_h[cellId][g];
                w_h[cellId][g] /= rho_h[cellId][g];
                e_h[cellId][g] /= rho_h[cellId][g];
                rho_s[cellId][g] = rho_xyz(pos,curr_time);
                u_s[cellId][g] = u_xyz(pos,curr_time);
                v_s[cellId][g] = v_xyz(pos,curr_time);
                w_s[cellId][g] = w_xyz(pos,curr_time);
                e_s[cellId][g] = e_xyz(pos,curr_time);
            }
            const auto& rho_coef  = Basis::func2coef([&](vector3f Xi){return rho_Xi( cell,Xi,curr_time);});
            const auto& rhou_coef = Basis::func2coef([&](vector3f Xi){return rhou_Xi(cell,Xi,curr_time);});
            const auto& rhov_coef = Basis::func2coef([&](vector3f Xi){return rhov_Xi(cell,Xi,curr_time);});
            const auto& rhow_coef = Basis::func2coef([&](vector3f Xi){return rhow_Xi(cell,Xi,curr_time);});
            const auto& rhoe_coef = Basis::func2coef([&](vector3f Xi){return rhoe_Xi(cell,Xi,curr_time);});
            #pragma omp simd
            for(uInt k=0;k<Basis::NumBasis;k++){
                error[cellId][5*k+0] = x[cellId][5*k+0] - rho_coef[k];
                error[cellId][5*k+1] = x[cellId][5*k+1] - rhou_coef[k];
                error[cellId][5*k+2] = x[cellId][5*k+2] - rhov_coef[k];
                error[cellId][5*k+3] = x[cellId][5*k+3] - rhow_coef[k];
                error[cellId][5*k+4] = x[cellId][5*k+4] - rhoe_coef[k];
            }
        }

        for(uInt cellId=0;cellId<cmesh.m_cells.size();cellId++){
            const auto& cell = cmesh.m_cells[cellId];
            for(uInt g=0; g<QuadC::num_points; ++g) {
                const auto& p = QuadC::points[g];
                const auto& pos = cell.transform_to_physical(p);
                fp <<std::setprecision(16)<< pos[0] << "  " <<std::setprecision(16)<< pos[1] << "  " <<std::setprecision(16)<< pos[2]
                 << "  " <<std::setprecision(16)<<  rho_h[cellId][g] << "  " <<std::setprecision(16)<<  rho_s[cellId][g] 
                 << "  " <<std::setprecision(16)<<  u_h[cellId][g] << "  " <<std::setprecision(16)<<  u_s[cellId][g] 
                 << "  " <<std::setprecision(16)<<  v_h[cellId][g] << "  " <<std::setprecision(16)<<  v_s[cellId][g] 
                 << "  " <<std::setprecision(16)<<  w_h[cellId][g] << "  " <<std::setprecision(16)<<  w_s[cellId][g] 
                 << "  " <<std::setprecision(16)<<  e_h[cellId][g] << "  " <<std::setprecision(16)<<  e_s[cellId][g] << std::endl;
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