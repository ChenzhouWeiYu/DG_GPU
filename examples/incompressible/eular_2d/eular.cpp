#include "base/type.h"
#include "base/filesystem_manager.h"
#include "mesh/mesh.h"
#include "matrix/matrix.h"
#include "dg/dg_basis/dg_basis.h"
#include "dg/dg_schemes/incompressible_implicit_convection.h"
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
                             
    if(order == 1) Run<1>(meshN, fsm);
    if(order == 2) Run<2>(meshN, fsm);
    if(order == 3) Run<3>(meshN, fsm);
    // if(order == 4) Run<4>(meshN);
    // if(order == 5) Run<5>(meshN);
}



template<uInt Order>
void Run(uInt N, FilesystemManager fsm){
    auto chrono_start = std::chrono::steady_clock::now();
    auto chrone_clock = [&](){return std::chrono::duration<double>(std::chrono::steady_clock::now()-chrono_start).count();};
    auto logging = [&](std::string ss){debug("Time  " + std::to_string(chrone_clock()) + "  \tsec      " + ss);};
    debug("Start   " + std::to_string(chrone_clock()));

    const auto& cmesh = create_mesh(N);
    logging("Split Hex Mesh to Tet");

    using BasisU = DGBasisEvaluator<Order>;
    using BasisP = DGBasisEvaluator<Order-1>;
    using QuadC = typename AutoQuadSelector<BasisU::OrderBasis, GaussLegendreTet::Auto>::type;
    constexpr uInt DoFs = 3*BasisU::NumBasis + BasisP::NumBasis;

    /* ======================================================= *\
    **   设置初值
    \* ======================================================= */
    LongVector<DoFs> x(cmesh.m_cells.size());
    #pragma omp parallel for schedule(dynamic)
    for(uInt cellId=0;cellId<cmesh.m_cells.size();cellId++){
        /* 获取单元 cell 的信息 */
        const auto& cell = cmesh.m_cells[cellId];
        /* 单元 cell 上，计算初值的多项式插值系数 */
        const auto& uvw_coef = BasisU::func2coef([&](vector3f Xi){
            return uvw_Xi(cell,Xi,0.0);
        });
        const auto& p_coef = BasisP::func2coef([&](vector3f Xi){
            return p_Xi(cell,Xi,0.0);
        });
        /* 写入到向量 x 的单元 cell 那一段*/
        for(uInt k=0;k<BasisU::NumBasis;k++){
            MatrixView<DoFs,1,3,1>(x[cellId],3*k,0) = uvw_coef[k];
        }
        for(uInt k=0;k<BasisP::NumBasis;k++){
            x[cellId](3*BasisU::NumBasis+k,0) = p_coef[k];
        }
    }
    logging("Set Init Value");


    /* ======================================================= *\
    **   算子 和 限制器 的实例化
    \* ======================================================= */
    InCompressibleImplicitConvection<BasisU::OrderBasis,BasisP::OrderBasis> convection;

    // /* 这个WENO是错的 */
    // OrthoPWENOLimiter<Basis::OrderBasis, QuadC> pwenolimiter(cmesh);    
    // /*  这个是保极值、保正，第三个参数是 Min 和 Max 的策略     *\
    //       true 采用相邻的均值作为 Min Max，更宽松，开销低
    // \*    false 为所有积分点的 Min Max，更紧致，开销大        */
    // PositiveLimiter<Basis::OrderBasis, QuadC, true> positivelimiter(cmesh, param_gamma);
    
    LongVector<DoFs> rhs(x.size());
    BlockSparseMatrix<DoFs,DoFs> sparse_mat;
    
    /* 用 保存的 dx 代入、离散 */ 
    convection.assemble(cmesh, x, 0.0,
                        sparse_mat, rhs);

    /* 补上质量矩阵，作为时间项的离散 */
    // for(uInt cellId = 0;cellId<cmesh.m_cells.size();cellId++){
    //     sparse_mat.add_block(cellId, cellId, DenseMatrix<DoFs,DoFs>::Diag(mass[cellId]/dt));
    //     rhs[cellId] += mass[cellId]/dt * x[cellId];
    // }

    /* 收集了所有 Block 后，组装为稀疏矩阵 */
    sparse_mat.finalize();   

    auto Ax = sparse_mat.multiply(x);
    auto residual = rhs-Ax;
    print("||Ax|| = " + std::to_string(Ax.norm()));
    print("||rhs|| = " + std::to_string(rhs.norm()));
    print("||rhs-Ax|| = " + std::to_string(residual.norm()/std::sqrt(cmesh.m_cells.size()*5*DoFs)));

    auto [U_h, U_s, error] = reconstruct_solution<QuadC,BasisU,BasisP>(cmesh, x, 0.0);


    std::ofstream fp(fsm.get_solution_file(0, N));
    for(uInt cellId=0;cellId<cmesh.m_cells.size();cellId++){
        const auto& cell = cmesh.m_cells[cellId];
        for(uInt xgId=0; xgId<QuadC::num_points; ++xgId) {
            const auto& p = QuadC::points[xgId];
            const auto& pos = cell.transform_to_physical(p);
            const auto& bUh = U_h[cellId].template SubMat<4,1>(4*xgId,0);
            const auto& bUs = U_s[cellId].template SubMat<4,1>(4*xgId,0);
            fp <<std::setprecision(16)<< pos[0] << "  " <<std::setprecision(16)<< pos[1] << "  " <<std::setprecision(16)<< pos[2]
                << "  " <<std::setprecision(16)<<  bUh[0] << "  " <<std::setprecision(16)<<  bUs[0] 
                << "  " <<std::setprecision(16)<<  bUh[1] << "  " <<std::setprecision(16)<<  bUs[1] 
                << "  " <<std::setprecision(16)<<  bUh[2] << "  " <<std::setprecision(16)<<  bUs[2]
                << "  " <<std::setprecision(16)<<  bUh[3] << "  " <<std::setprecision(16)<<  bUs[3] << std::endl;
        }
    }
    fp.close(); 

    // return;
    sparse_mat.output_as_scalar("A.txt");
    // return;

    /* ======================================================= *\
    **   计算 (\phi_i, \phi_i) 作为质量矩阵
    \* ======================================================= */
    LongVector<DoFs> mass(x.size());
    for(uInt cid=0;cid<cmesh.m_cells.size();cid++){
        const auto& detJac = cmesh.m_cells[cid].compute_jacobian_det();
        for(uInt i=0; i<BasisU::NumBasis; ++i) {
            Scalar val = 0.0;
            for(uInt g=0; g<QuadC::num_points; ++g) {
                const auto& weight = QuadC::weights[g] * detJac;
                const auto& p = QuadC::points[g];
                auto phi = BasisU::eval_all(p[0], p[1], p[2]);
                val += phi[i] * phi[i] * weight;
            }
            for(uInt k=0; k<3; ++k) {
                mass[cid](3*i + k, 0) = val;
            }
        }
        for(uInt i=0; i<BasisP::NumBasis; ++i) {
            Scalar val = 0.0;
            for(uInt g=0; g<QuadC::num_points; ++g) {
                const auto& weight = QuadC::weights[g] * detJac;
                const auto& p = QuadC::points[g];
                auto phi = BasisP::eval_all(p[0], p[1], p[2]);
                val += phi[i] * phi[i] * weight;
            }
            
            mass[cid](3*BasisU::NumBasis + i, 0) = val;
            
        }
    }
    
   
   
    /* ======================================================= *\
    **   开始迭代
    **   第一层迭代，关于数值解的保存的间隔
    **   间隔 Dt 时间保存一次
    \* ======================================================= */
    print(std::array<std::string,7>{"#       time", 
                    "rel.err  u", "rel.err  v", "rel.err  w", 
                    "rel.err  p", "rel.err coef", "cpu time"});
    for(uInt kkkk=0;kkkk<10;kkkk++){    
        Scalar Dt = 0.02;
        Scalar max_dt = Dt * std::pow((1.0/N),(Order+1));
        // max_dt = Dt*0.1; // 可能会给的太大，上面那个是肯定稳定的
        max_dt *= 10;
        Scalar dt = max_dt;
        uInt kk = 0;


        /* ======================================================= *\
        **   第二层迭代，关于 保存间隔 Dt 内的时间推进
        **   需要判断是否超过 Dt，超过了就截断
        \* ======================================================= */
        Scalar sub_t = 0;
        do{ 
            dt = std::min(dt,Dt-sub_t);
            sub_t += dt;
            
            /* 记录下旧时刻的解，为 dx，本来这个变量想用来解增量型的 */
            LongVector<DoFs> dx = x;

            /* 记录初始迭代步的 残差、初始的增量，用于非线性迭代停机条件 */
            Scalar init_delta = 0.0;
            Scalar prev_delta = 0.0;
            Scalar init_residual = 0.0;

            /* ======================================================= *\
            **   第三层迭代，关于 时间步 dt 内的 非线性迭代
            **   需要判断是否超过 Dt，超过了就截断
            \* ======================================================= */
            for(uInt picard_iter = 0; picard_iter < 100; picard_iter++){

                LongVector<DoFs> rhs(x.size());
                BlockSparseMatrix<DoFs,DoFs> sparse_mat;
                
                /* 用 保存的 dx 代入、离散 */ 
                convection.assemble(cmesh, dx, Dt * (kkkk) + sub_t + 0.5 * dt,
                                    sparse_mat, rhs);

                /* 补上质量矩阵，作为时间项的离散 */
                for(uInt cellId = 0;cellId<cmesh.m_cells.size();cellId++){
                    sparse_mat.add_block(cellId, cellId, DenseMatrix<DoFs,DoFs>::Diag(mass[cellId]/dt));
                    rhs[cellId] += mass[cellId]/dt * x[cellId];
                }

                /* 收集了所有 Block 后，组装为稀疏矩阵 */
                sparse_mat.finalize();   

                /* 将之前冗长的 矩阵转换、求解，打包为一个接口 */
                EigenSparseSolver<DoFs,DoFs> solver(sparse_mat,rhs);
                LongVector<DoFs> ddx = solver.SparseLU(dx);

                /* 降维模拟的后处理 */
                for(uInt cid = 0; cid < cmesh.m_cells.size(); cid++){
                    ddx[cid][2 + 3*0] = 0;
                    for (uInt k = 1; k < BasisU::NumBasis; k++) {
                        ddx[cid][2 + 3*k] = 0;
                    }
                }
                
                /* 施加限制器，这里是只有保极值保正，需要先计算允许的 Min Max */
                // positivelimiter.constructMinMax(ddx, dx); 
                // positivelimiter.apply(ddx, dx); 

                /* 计算限制后的解 ddx 与上一非线性迭代步 dx 的增量 */
                Scalar delta = (ddx - dx).dot(ddx - dx);
                delta = std::sqrt(delta);

                /* 替换 dx 为限制后的新解 */
                dx = ddx;

                /* 关于停机条件的处理，包括第一次迭代时的设置 */
                if(picard_iter==0) {
                    init_delta = delta;
                    prev_delta = delta;
                }
                Scalar rate_delta = delta/init_delta;  // 下降率充分大
                Scalar rel_delta = delta/prev_delta;   // 相对变化率足够小
                prev_delta = delta;

                /* 计算非线性残差，但没有用于停机条件 */
                std::ostringstream oss;
                oss << delta;
                {
                    LongVector<DoFs> rhs(x.size());
                    BlockSparseMatrix<DoFs,DoFs> sparse_mat;
                    
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
            /* 非线性迭代结束，x 赋值为最后的 dx */ 
            x = dx;

            logging("Iter  " + std::to_string(kkkk+1) + " \tSub TimeStep \t" + std::to_string(sub_t));


        }while(sub_t < Dt);


        Scalar curr_time = Dt * (kkkk+1);

        
        auto err_integral = [&](LongVector<4*QuadC::num_points> U_h,LongVector<4*QuadC::num_points> U_s){
            DenseMatrix<4,1> err_per_cells = DenseMatrix<4,1>::Zeros();
            DenseMatrix<4,1> sol_per_cells = 1e-47 * DenseMatrix<4,1>::Ones();
            for(uInt cellId=0;cellId<cmesh.m_cells.size();cellId++){
                const auto& cell = cmesh.m_cells[cellId];
                for(uInt xgId=0; xgId<QuadC::num_points; ++xgId) {
                    const auto& bUh = U_h[cellId].template SubMat<4,1>(4*xgId,0);
                    const auto& bUs = U_s[cellId].template SubMat<4,1>(4*xgId,0);
                    const auto& bUe = bUh - bUs;
                    const auto& weight = QuadC::weights[xgId] * cell.compute_jacobian_det();
                    // Scalar error_cell = uh[cellId][xgId] - us[cellId][xgId];

                    err_per_cells += pow(bUe,2) * weight;
                    sol_per_cells += pow(bUs,2) * weight;
                }
            }
            return pow(err_per_cells/sol_per_cells,0.5);
        };

        auto [U_h, U_s, error] = reconstruct_solution<QuadC,BasisU,BasisP>(cmesh, x, curr_time);

        std::ofstream fp(fsm.get_solution_file(kkkk+1, N));
        for(uInt cellId=0;cellId<cmesh.m_cells.size();cellId++){
            const auto& cell = cmesh.m_cells[cellId];
            for(uInt xgId=0; xgId<QuadC::num_points; ++xgId) {
                const auto& p = QuadC::points[xgId];
                const auto& pos = cell.transform_to_physical(p);
                const auto& bUh = U_h[cellId].template SubMat<4,1>(4*xgId,0);
                const auto& bUs = U_s[cellId].template SubMat<4,1>(4*xgId,0);
                fp <<std::setprecision(16)<< pos[0] << "  " <<std::setprecision(16)<< pos[1] << "  " <<std::setprecision(16)<< pos[2]
                 << "  " <<std::setprecision(16)<<  bUh[0] << "  " <<std::setprecision(16)<<  bUs[0] 
                 << "  " <<std::setprecision(16)<<  bUh[1] << "  " <<std::setprecision(16)<<  bUs[1] 
                 << "  " <<std::setprecision(16)<<  bUh[2] << "  " <<std::setprecision(16)<<  bUs[2]
                 << "  " <<std::setprecision(16)<<  bUh[3] << "  " <<std::setprecision(16)<<  bUs[3] << std::endl;
            }
        }
        fp.close();

        const auto& U_err = err_integral(U_h,U_s);
        print(std::array<Scalar,7>{curr_time, U_err[0], U_err[1],U_err[2],U_err[3], 
                std::sqrt(error.dot(error)/x.dot(x)), chrone_clock()});
    }
}