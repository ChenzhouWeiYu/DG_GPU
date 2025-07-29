#include "base/type.h"
#include "base/filesystem_manager.h"
#include "mesh/mesh.h"
#include "matrix/matrix.h"
#include "dg/dg_basis/dg_basis.h"
#include "dg/dg_schemes/explicit_convection.h"
#include "dg/dg_schemes/explicit_diffusion.h"
#include "dg/dg_schemes/implicit_convection.h"
#include "dg/dg_schemes/implicit_diffusion.h"
#include "DG/DG_Schemes/PositiveLimiter.h"
#include "DG/DG_Schemes/PWENOLimiter.h"
#include "EigenSolver/EigenSparseSolver.h"

#include "problem.h"
#include "tools.h"

template<uInt Order>
void Run(uInt N, FilesystemManager fsm);

int main(int argc, char** argv){
    omp_set_num_threads(get_phy_cpu());
    Eigen::setNbThreads(get_phy_cpu());

    int order = std::stoi(argv[1]);
    int meshN = std::stoi(argv[2]);

    // 文件管理系统
    FilesystemManager fsm("./Order_" + std::to_string(order) + "_Mesh_" + std::to_string(meshN));

    // 创建目录结构
    fsm.prepare_output_directory();

    // 获取各种路径
    std::cout << "Solution file: " << fsm.get_solution_file(0, 100) << "\n";
    std::cout << "Error log:     " << fsm.get_error_log_file() << "\n";
    std::cout << "Config file:   " << fsm.get_config_file() << "\n";
    std::cout << "Run info:      " << fsm.get_run_info_file() << "\n";
                             
    // if(order == 0) Run<0>(meshN);
    if(order == 1) Run<1>(meshN, fsm);
    if(order == 2) Run<2>(meshN, fsm);
    if(order == 3) Run<3>(meshN, fsm);
    // if(order == 4) Run<4>(meshN);
    // if(order == 5) Run<5>(meshN);
    // if(order == 6) Run<6>(meshN);
    // if(order == 7) Run<7>(meshN);
    // if(order == 8) Run<8>(meshN);
    // if(order == 9) Run<9>(meshN);
}












template<uInt Order>
void Run(uInt N, FilesystemManager fsm){
    auto chrono_start = std::chrono::steady_clock::now();
    auto chrone_clock = [&](){return std::chrono::duration<double>(std::chrono::steady_clock::now()-chrono_start).count();};
    auto logging = [&](std::string ss){debug("Time  " + std::to_string(chrone_clock()) + "  \tsec      " + ss);};
    debug("Start   " + std::to_string(chrone_clock()));
    
    const auto& cmesh = create_mesh(N);
    logging("Split Hex Mesh to Tet");

    using Basis = DGBasisEvaluator<Order>;
    using QuadC = typename AutoQuadSelector<Basis::OrderBasis, GaussLegendreTet::Auto>::type;
    constexpr uInt DoFs = 5*Basis::NumBasis;

    LongVector<5*Basis::NumBasis> x(cmesh.m_cells.size());
    

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


    ExplicitConvection<Basis::OrderBasis> explicitconvection;
    ImplicitConvection<Basis::OrderBasis> convection;

    OrthoPWENOLimiter<Basis::OrderBasis, QuadC> pwenolimiter(cmesh);
    PositiveLimiter<Basis::OrderBasis, QuadC, true> positivelimiter(cmesh, param_gamma);
    

    
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
    for(uInt kkkk=0;kkkk<10;kkkk++){    
        Scalar Dt = 0.025;
        Scalar max_dt = Dt * std::pow((1.0/N),(Order+1));
        max_dt = (1.0/N)*0.25;

        Scalar dt = max_dt;
        uInt kk = 0;
        
        Scalar sub_t = 0;
        do{ 

            // if(sub_t == 0){
            //     dt = 1e-3 * max_dt;
            // }
            // else{
            //     dt = 1e-2 * max_dt;
            // }
            // dt = 1e-2 * max_dt;
            // if(kkkk>1 || kk>3){
            //     dt = 1.0 * max_dt;
            // }else if(kk==0){
            //     dt = 0.1 * max_dt;
            // }else if(kk==1){
            //     dt = 0.2 * max_dt;
            // }else if(kk==2){
            //     dt = 0.3 * max_dt;
            // }else if(kk==3){
            //     dt = 0.4 * max_dt;
            // }
            // kk++;

            dt = std::min(dt,Dt-sub_t);
            sub_t += dt;
            

            
            LongVector<5*Basis::NumBasis> dx = x;
            Scalar init_delta = 0.0;
            Scalar prev_delta = 0.0;
            Scalar init_residual = 0.0;
            for(uInt picard_iter = 0; picard_iter < 100; picard_iter++){

                LongVector<5*Basis::NumBasis> rhs(x.size());
                BlockSparseMatrix<5*Basis::NumBasis,5*Basis::NumBasis> sparse_mat;
                
                // 用 保存的 dx 代入、离散
                convection.assemble(cmesh, dx, Dt * (kkkk) + sub_t + 0.5 * dt,
                                    sparse_mat, rhs);

                for(uInt cellId = 0;cellId<cmesh.m_cells.size();cellId++){
                    sparse_mat.add_block(cellId, cellId, DenseMatrix<DoFs,DoFs>::Diag(mass[cellId]/dt));
                    rhs[cellId] += mass[cellId]/dt * x[cellId];
                }

                sparse_mat.finalize();


                EigenSparseSolver<DoFs,DoFs> solver(sparse_mat,rhs);
                LongVector<DoFs> ddx = solver.DGMRES(dx);




     

                // 应用限制器
                // pwenolimiter.apply(ddx, dx); 
                positivelimiter.constructMinMax(ddx, dx); 
                positivelimiter.apply(ddx, dx); 

                for(uInt cid = 0; cid < cmesh.m_cells.size(); cid++){
                    ddx[cid][2 + 5*0] = 0;
                    ddx[cid][3 + 5*0] = 0;
                    for (uInt k = 1; k < Basis::NumBasis; k++) {
                        ddx[cid][2 + 5*k] = 0;
                        ddx[cid][3 + 5*k] = 0;
                    }
                }
                

                Scalar delta = (ddx - dx).dot(ddx - dx);
                dx = ddx;

                delta = std::sqrt(delta);
                if(picard_iter==0) {
                    init_delta = delta;
                    prev_delta = delta;
                }
                Scalar rate_delta = delta/prev_delta;
                Scalar rel_delta = delta/init_delta;
                prev_delta = delta;
                std::ostringstream oss;
                oss << delta;
                {
                    LongVector<5*Basis::NumBasis> rhs(x.size());
                    BlockSparseMatrix<5*Basis::NumBasis,5*Basis::NumBasis> sparse_mat;
                    

                    // 用 保存的 dx 代入、离散
                    convection.assemble(cmesh, dx, Dt * (kkkk) + sub_t + 0.5 * dt,
                                        sparse_mat, rhs);
                    sparse_mat.finalize();
                    const auto& residual = (mass/dt*dx + sparse_mat.multiply(dx)) - (rhs + mass/dt * x);
                    std::ostringstream oss;
                    oss << std::sqrt(residual.dot(residual));
                    logging("Picard iter " + std::to_string(picard_iter) + "  " + oss.str());

                }
                logging("Picard iter " + std::to_string(picard_iter) + "  " + oss.str());
                if(delta < 1e-8 || rel_delta < 1e-8 || (picard_iter>10 && rate_delta > 0.95)) break;
            }
            // Picard 结束，赋值，x 为 u^{n+1}
            x = dx;// - dt / mass * dx;

            logging("Iter  " + std::to_string(kkkk+1) + " \tSub TimeStep \t" + std::to_string(sub_t));


        }while(sub_t < Dt);


        Scalar curr_time = Dt * (kkkk+1);

        std::ofstream fp(fsm.get_solution_file(kkkk+1, N));
        auto err_integral = [&](LongVector<5*QuadC::num_points> U_h,LongVector<5*QuadC::num_points> U_s){
            DenseMatrix<5,1> err_per_cells = DenseMatrix<5,1>::Zeros();
            DenseMatrix<5,1> sol_per_cells = 1e-47 * DenseMatrix<5,1>::Ones();
            for(uInt cellId=0;cellId<cmesh.m_cells.size();cellId++){
                const auto& cell = cmesh.m_cells[cellId];
                for(uInt xgId=0; xgId<QuadC::num_points; ++xgId) {
                    const auto& bUh = U_h[cellId].template SubMat<5,1>(5*xgId,0);
                    const auto& bUs = U_s[cellId].template SubMat<5,1>(5*xgId,0);
                    const auto& bUe = bUh - bUs;
                    const auto& weight = QuadC::weights[xgId] * cell.compute_jacobian_det();
                    // Scalar error_cell = uh[cellId][xgId] - us[cellId][xgId];

                    err_per_cells += pow(bUe,2) * weight;
                    sol_per_cells += pow(bUs,2) * weight;
                }
            }
            return pow(err_per_cells/sol_per_cells,0.5);
        };

        auto [U_h, U_s, error] = reconstruct_solution<QuadC,Basis>(cmesh, x, curr_time);

        for(uInt cellId=0;cellId<cmesh.m_cells.size();cellId++){
            const auto& cell = cmesh.m_cells[cellId];
            for(uInt xgId=0; xgId<QuadC::num_points; ++xgId) {
                const auto& p = QuadC::points[xgId];
                const auto& pos = cell.transform_to_physical(p);
                const auto& bUh = U_h[cellId].template SubMat<5,1>(5*xgId,0);
                const auto& bUs = U_s[cellId].template SubMat<5,1>(5*xgId,0);
                fp <<std::setprecision(16)<< pos[0] << "  " <<std::setprecision(16)<< pos[1] << "  " <<std::setprecision(16)<< pos[2]
                 << "  " <<std::setprecision(16)<<  bUh[0] << "  " <<std::setprecision(16)<<  bUs[0] 
                 << "  " <<std::setprecision(16)<<  bUh[1] << "  " <<std::setprecision(16)<<  bUs[1] 
                 << "  " <<std::setprecision(16)<<  bUh[2] << "  " <<std::setprecision(16)<<  bUs[2]
                 << "  " <<std::setprecision(16)<<  bUh[3] << "  " <<std::setprecision(16)<<  bUs[3] 
                 << "  " <<std::setprecision(16)<<  bUh[4] << "  " <<std::setprecision(16)<<  bUs[4] << std::endl;
            }
        }
        fp.close();

        const auto& U_err = err_integral(U_h,U_s);
        print(std::array<Scalar,8>{curr_time, U_err[0], U_err[1],U_err[2],U_err[3],U_err[4], 
                std::sqrt(error.dot(error)/x.dot(x)), chrone_clock()});
        // print(vector3f{curr_time, std::sqrt(err.dot(err)/rho.dot(rho)),chrone_clock()});
    }
}