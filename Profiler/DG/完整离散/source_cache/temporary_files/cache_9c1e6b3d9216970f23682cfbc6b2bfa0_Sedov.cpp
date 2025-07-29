#include "base/type.h"
#include "base/filesystem_manager.h"
#include "base/LoggerSystem.hpp"
#include "base/io.h"
#include "mesh/mesh.h"
#include "matrix/matrix.h"
#include "dg/dg_basis/dg_basis.h"
#include "dg/dg_schemes/implicit_convection.h"
// #include "dg/dg_schemes/implicit_diffusion.h"
#include "DG/DG_Schemes/PositiveLimiter.h"
// #include "DG/DG_Schemes/PWENOLimiter.h"
#include "solver/eigen_sparse_solver.h"

#include "problem.h"
#include "tools.h"

template<uInt Order>
void Run(uInt N, FilesystemManager& fsm, LoggerSystem& logger);

int main(int argc, char** argv){
    int cpus = get_phy_cpu();
    int order = std::stoi(argv[1]);
    int meshN = std::stoi(argv[2]);
    if(argc > 3){
        cpus = std::stoi(argv[3]);
    }
    omp_set_num_threads(cpus);
    Eigen::setNbThreads(cpus);

    // 文件管理系统
    FilesystemManager fsm("./Order_" + std::to_string(order) + "_Mesh_" + std::to_string(meshN));

    // 创建目录结构
    fsm.prepare_output_directory();

    // 日志系统
    LoggerSystem logger(fsm);
    logger.log_boxed_title("Discontinuous Galerkin Simulation");

    logger.set_indent(0);

    // 获取各种路径
    // std::cout << "Solution file: " << fsm.get_solution_file(0, 100) << "\n";
    // std::cout << "Error log:     " << fsm.get_error_log_file() << "\n";
    // std::cout << "Config file:   " << fsm.get_config_file() << "\n";
    // std::cout << "Run info:      " << fsm.get_run_info_file() << "\n";
    // std::cout << "CPU used:      " << cpus << "\n";
    logger.print_header("Discontinuous Galerkin Simulation");
    logger.print_config(order, meshN, cpus);
                             
    // if(order == 0) Run<0>(meshN);
    if(order == 1) Run<1>(meshN, fsm, logger);
    if(order == 2) Run<2>(meshN, fsm, logger);
    if(order == 3) Run<3>(meshN, fsm, logger);
    // if(order == 4) Run<4>(meshN);
    // if(order == 5) Run<5>(meshN);
    // if(order == 6) Run<6>(meshN);
    // if(order == 7) Run<7>(meshN);
    // if(order == 8) Run<8>(meshN);
    // if(order == 9) Run<9>(meshN);
}












template<uInt Order>
void Run(uInt N, FilesystemManager& fsm, LoggerSystem& logger){
    // auto chrono_start = std::chrono::steady_clock::now();
    // auto chrone_clock = [&](){return std::chrono::duration<double>(std::chrono::steady_clock::now()-chrono_start).count();};
    // auto logging = [&](std::string ss){debug("CPU Time: " + std::to_string(chrone_clock()) + "  \tsec      " + ss);};
    // debug("Start   " + std::to_string(chrone_clock()));

    logger.log_section_title("Setup Stage");

    logger.start_stage("Split Hex Mesh to Tet");

    const auto& cmesh = create_mesh(N);
    check_mesh(cmesh);


    logger.end_stage();




    using Basis = DGBasisEvaluator<Order>;
    using QuadC = typename AutoQuadSelector<Basis::OrderBasis, GaussLegendreTet::Auto>::type;
    constexpr uInt DoFs = 5*Basis::NumBasis;
    /* ======================================================= *\
    **   算子 和 限制器 的实例化
    \* ======================================================= */
    ImplicitConvection<Basis::OrderBasis> convection;
    // ImplicitDiffusion<Basis::OrderBasis> diffusion(param_mu);

    /* 这个WENO是错的 */
    // OrthoPWENOLimiter<Basis::OrderBasis, QuadC> pwenolimiter(cmesh);
    /*  这个是保极值、保正，第三个参数是 Min 和 Max 的策略     *\
          true 采用相邻的均值作为 Min Max，更宽松，开销低
    \*    false 为所有积分点的 Min Max，更紧致，开销大        */
    PositiveLimiter<Basis::OrderBasis, QuadC, false> positivelimiter(cmesh, param_gamma);
    
    




    
    logger.start_stage("Set Initial Condition");
    /* ======================================================= *\
    **   设置初值
    \* ======================================================= */
    LongVector<DoFs> U_n(cmesh.m_cells.size());
    #pragma omp parallel for schedule(dynamic)
    for(uInt cellId=0;cellId<cmesh.m_cells.size();cellId++){
        /* 获取单元 cell 的信息 */
        const auto& cell = cmesh.m_cells[cellId];
        /* 单元 cell 上，计算初值的多项式插值系数 */
        const auto& rhoU_coef = Basis::func2coef([&](vector3f Xi)->DenseMatrix<5,1>{
            return {rho_Xi(cell,Xi),rhou_Xi(cell,Xi),rhov_Xi(cell,Xi),rhow_Xi(cell,Xi),rhoe_Xi(cell,Xi)};
        });
        /* 写入到向量 U_n 的单元 cell 那一段*/
        for(uInt k=0;k<Basis::NumBasis;k++){
            MatrixView<DoFs,1,5,1>(U_n[cellId],5*k,0) = rhoU_coef[k];
        }
    }
    positivelimiter.constructMinMax(U_n, U_n); 
    positivelimiter.apply(U_n, U_n); 

    // logging("Set Init Value");
    
    save_Uh_Us<QuadC,Basis>(cmesh, U_n, 0.0, fsm.get_solution_file(0, N));

    logger.end_stage();

    /* ======================================================= *\
    **   计算 (\phi_i, \phi_i) 作为质量矩阵
    **   正交基，只需要计算、保存对角元
    \* ======================================================= */
    
    constexpr uInt num_vol_points = QuadC::num_points;
    constexpr auto Qpoints = QuadC::get_points();
    constexpr auto Qweights = QuadC::get_weights();
    LongVector<DoFs> mass(U_n.size());
    for(uInt cid=0;cid<cmesh.m_cells.size();cid++){
        const auto& detJac = cmesh.m_cells[cid].compute_jacobian_det();
        for(uInt i=0; i<Basis::NumBasis; ++i) {
            Scalar val = 0.0;
            for(uInt g=0; g<num_vol_points; ++g) {
                const auto& weight = Qweights[g] * detJac;
                const auto& p = Qpoints[g];
                auto phi = Basis::eval_all(p[0], p[1], p[2]);
                val += phi[i] * phi[i] * weight;
            }
            for(uInt k=0; k<5; ++k) {
                mass[cid](5*i + k, 0) = val;
            }
        }
    }

    
    

    /* ======================================================= *\
    **   开始迭代
    **   第一层迭代，关于数值解的保存的间隔
    **   间隔 Dt 时间保存一次
    \* ======================================================= */
    // print(std::array<std::string,8>{"#       time", "rel.err  rho",
    //                 "rel.err  u", "rel.err  v", "rel.err  w", 
    //                 "rel.err  e", "rel.err coef", "cpu time"});
    logger.log_section_title("Time Marching");
    Scalar total_time = 0.0;
    for(uInt save_step = 0; save_step < 1; save_step++){    
        Scalar Dt = 0.001;
        Scalar max_dt = Dt * std::pow((1.1*5/N),(Order+1));
        max_dt = (1.1/N) * std::pow(1.0/(2.0*Order+1.0),1);
        Scalar dt = max_dt;
        uInt kk = 0;
        

        /* ======================================================= *\
        **   第二层迭代，关于 保存间隔 Dt 内的时间推进
        **   需要判断是否超过 Dt，超过了就截断
        **   sub_t 是当前的 Dt 间隔内的时间推进
        \* ======================================================= */
        Scalar sub_t = 0;
        do{ 
            
            /* 时间步长 dt 截断到 Dt 长度 */
            dt = std::min(max_dt, Dt-sub_t); 
            if(total_time < max_dt * 1.00e-6) dt = max_dt * 1.00e-6; else
            if(total_time < max_dt * 1.00e-5) dt = max_dt * 1.00e-5; else
            if(total_time < max_dt * 1.00e-4) dt = max_dt * 1.00e-4; else
            if(total_time < max_dt * 3.16e-3) dt = max_dt * 3.16e-3; else
            if(total_time < max_dt * 1.00e-2) dt = max_dt * 1.00e-2; else
            if(total_time < max_dt * 3.00e-2) dt = max_dt * 2.00e-2; else
            if(total_time < max_dt * 6.00e-2) dt = max_dt * 3.00e-2; else
            if(total_time < max_dt * 1.00e-1) dt = max_dt * 4.00e-2; else
            if(total_time < max_dt * 1.50e-1) dt = max_dt * 5.00e-2; else
            if(total_time < max_dt * 2.10e-1) dt = max_dt * 6.00e-2; else
            if(total_time < max_dt * 2.80e-1) dt = max_dt * 7.00e-2; else
            if(total_time < max_dt * 3.60e-1) dt = max_dt * 8.00e-2; else
            if(total_time < max_dt * 4.50e-1) dt = max_dt * 9.00e-2; else
            if(total_time < max_dt * 5.50e-1) dt = max_dt * 1.00e-1; else
            if(total_time < max_dt * 7.00e-1) dt = max_dt * 1.50e-1; else
            if(total_time < max_dt * 1.00e-0) dt = max_dt * 3.00e-1; else
            if(total_time < max_dt * 1.50e-0) dt = max_dt * 5.00e-1; else
            if(total_time < max_dt * 2.10e-0) dt = max_dt * 7.00e-1;
            // logging("Delta t = " + std::to_string(dt));
            sub_t += dt;
            total_time += dt; // 实际物理时间

            logger.log_time_step(save_step, total_time, dt);

            
            /* 记录下旧时刻的解 */
            LongVector<DoFs> U_k = U_n;

            /* 记录初始迭代步的 残差、初始的增量，用于非线性迭代停机条件 */
            Scalar init_delta = 0.0;
            Scalar prev_delta = 0.0;
            Scalar init_res_norm = 0.0;
            Scalar prev_res_norm = 0.0;

            LongVector<5*Basis::NumBasis> res_old(U_n.size());
            /* ======================================================= *\
            **   lambda 表达式，用于第三层迭代的 离散算子
            **   U_k : 上一步迭代的解，用于离散时的线性化，边界条件的 ghost 
            **   U_n : 当前时刻的解，用于时间离散的源项
            \* ======================================================= */
            const auto& get_matrix = [&](
                BlockSparseMatrix<5*Basis::NumBasis,5*Basis::NumBasis>& sparse_mat,
                LongVector<5*Basis::NumBasis>& rhs, 
                const LongVector<5*Basis::NumBasis>& U_k)
            {
                /* 用 保存的 dx 代入、离散 */ 
                convection.assemble(cmesh, U_k, total_time, sparse_mat, rhs);
                // diffusion.assemble(cmesh, U_k, total_time, sparse_mat, rhs);
                
                /* 补上质量矩阵，作为时间项的离散 */
                for(uInt cellId = 0;cellId<cmesh.m_cells.size();cellId++){
                    sparse_mat.add_block(cellId, cellId, DenseMatrix<DoFs,DoFs>::Diag(mass[cellId]/dt));
                    rhs[cellId] += mass[cellId]/dt * U_n[cellId];
                }
                /* 收集了所有 Block 后，组装为稀疏矩阵 */
                sparse_mat.finalize();
            };
            /* ======================================================= *\
            **   第三层迭代，关于 时间步 dt 内的 非线性迭代
            **   需要判断是否超过 Dt，超过了就截断
            \* ======================================================= */
            for(uInt picard_iter = 0; picard_iter < 2000; picard_iter++){
                logger.log_picard_iteration(picard_iter);
                
                // logging("Start discretization");
                logger.log_discretization_start();

                LongVector<5*Basis::NumBasis> rhs(U_n.size());
                BlockSparseMatrix<5*Basis::NumBasis,5*Basis::NumBasis> sparse_mat;
                /* 用 第 k 步迭代的解 U_k 进行离散 */ 
                get_matrix(sparse_mat, rhs, U_k);

                logger.log_discretization_end();

                // sparse_mat.output_as_scalar("12345.txt");
                
                // logging("Start linear solver");

                logger.log_krylov_start();
                /* 调用 Eigen 求解线性方程组 */
                EigenSparseSolver<DoFs,DoFs> solver(sparse_mat,rhs);
                LongVector<DoFs> U_k_tmp(U_k.size());
                U_k_tmp = U_k;
                // solver.set_iterations(1000);
                // solver.set_tol(1e-14);
                // auto [krylov_iters, krylov_residual] = solver.DGMRES(U_k, U_k_tmp);
                // logger.log_krylov_end(krylov_iters, krylov_residual);

                // if (krylov_residual > 1e-12){
                //     logger.log_krylov_fallback_start();
                //     U_k_tmp = solver.SparseLU(U_k);
                //     // logger.output("        Krylov solver failed, using LU instead.");
                //     logger.log_krylov_fallback_end();
                // }

                // logging("Linear solver finished");

                // if(picard_iter>3){
                //     LongVector<DoFs> rhs(U_n.size());
                //     BlockSparseMatrix<DoFs,DoFs> sparse_mat;
                //     get_matrix(sparse_mat, rhs, U_k_tmp);
                //     const auto& res_new = (sparse_mat.multiply(U_k_tmp)) - (rhs);
                //     Scalar a = 0;
                //     a = (res_new-res_old).dot(res_new)/(res_new-res_old).dot(res_new-res_old);
                //     a = std::max(a,-5.0);
                //     a = std::min(a,0.1);

                //     // a = -0.5;
                    
                //     U_k_tmp = a*U_k + (1-a)*U_k_tmp;
                //     // debug(a);
                //     // res_old = res_new;
                // }
                
                // /* 降维模拟的后处理 */
                for(uInt cid = 0; cid < cmesh.m_cells.size(); cid++){
                    U_k_tmp[cid][3 + 5*0] = 0;
                    for (uInt k = 1; k < Basis::NumBasis; k++) {
                        U_k_tmp[cid][3 + 5*k] = 0;
                    }
                }

                // /* 施加限制器，这里是只有保极值保正，需要先计算允许的 Min Max */
                // pwenolimiter.apply(U_k_tmp, U_k); 
                positivelimiter.constructMinMax(U_k_tmp, U_k); 
                positivelimiter.apply(U_k_tmp, U_k); 

                /* 计算 新的解 U_k_tmp 与 上一非线性迭代步 U_k 的增量 */
                Scalar delta = (U_k_tmp - U_k).dot(U_k_tmp - U_k);
                delta = std::sqrt(delta);

                /* 替换 U_k 为新解 U_k_tmp */
                U_k = U_k_tmp;

                                /* 关于停机条件的处理，包括第一次迭代时的设置 */
                if(picard_iter==0) {
                    init_delta = delta;
                    prev_delta = delta;
                }
                Scalar rate_delta = delta/prev_delta;  // 下降率充分大
                Scalar rel_delta = delta/init_delta;   // 相对变化率足够小
                Scalar rate_res_norm, rel_res_norm;
                prev_delta = delta;

                /* 计算非线性残差，但没有用于停机条件 */
                std::ostringstream oss;
                oss << delta;
                Scalar res_norm;
                // logging("Re-discretize, calculate nonlinear residuals");
                {
                    LongVector<5*Basis::NumBasis> rhs(U_n.size());
                    BlockSparseMatrix<5*Basis::NumBasis,5*Basis::NumBasis> sparse_mat;
                    get_matrix(sparse_mat, rhs, U_k);
                    res_old = sparse_mat.multiply(U_k) - rhs;
                    res_norm = res_old.dot(res_old);
                    res_norm = std::sqrt(res_norm);
                    if(picard_iter==0) {
                        init_res_norm = res_norm;
                        prev_res_norm = res_norm;
                    }
                    rate_res_norm = res_norm/prev_res_norm;
                    rel_res_norm = res_norm/init_res_norm;
                    prev_res_norm = res_norm;
                    // std::ostringstream oss;
                    // oss << std::sqrt(res_norm);
                    // logging("Picard iter " + std::to_string(picard_iter) + "\t ||N(U_k)|| = " + oss.str());

                }
                // logging("Picard iter " + std::to_string(picard_iter) + "\t ||delta U|| = " + oss.str());
                logger.log_nonlinear_residual(res_norm, delta);
                logger.log_convergence_check(delta, rel_delta, res_norm, rel_res_norm, rate_res_norm);
                logger.end_picard_iteration();
                if(delta < 1e-4 || rel_delta < 1e-4 || 
                    res_norm < 1e-4 || rel_res_norm < 1e-4 || 
                    (picard_iter>10 && rate_res_norm > 1-1e-4 && rel_res_norm < 1e-3) || 
                    (picard_iter>5 && rate_res_norm > 1-1e-6)) {
                    // logging("delta = " + std::to_string(delta) + "    rel_delta = " + std::to_string(rel_delta) + "    res_norm = " + std::to_string(res_norm) + "    rel_res_norm = " + std::to_string(rel_res_norm) + "    rate_res_norm = " + std::to_string(rate_res_norm));
                    break;
                }
                if(res_norm>1e8){
                    sub_t = Dt; // 可能是数值不稳定，跳出迭代
                    break; // 可能是数值不稳定，跳出迭代
                }
            }
            /* 非线性迭代结束，U_n 赋值为最后的 U_k，作为新的时间步 U_{n+1} */ 
            U_n = U_k;
            logger.end_stage();
            if(total_time < max_dt) {
                save_Uh_Us<QuadC,Basis>(cmesh, U_n, 0.0, fsm.get_solution_file(1000000 + uInt(100000000*total_time), N));
            }
            // logging("Iter  " + std::to_string(save_step+1) + " \t Total time: " + std::to_string(total_time));
        }while(sub_t < Dt);

        save_Uh_Us<QuadC,Basis>(cmesh, U_n, total_time, fsm.get_solution_file(save_step+1, N));
    }
}
