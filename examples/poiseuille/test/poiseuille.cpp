#include "base/type.h"
#include "mesh/mesh.h"
#include "matrix/matrix.h"
#include "dg/dg_basis/dg_basis.h"
// #include "dg/dg_schemes/explicit_convection.h"
// #include "dg/dg_schemes/explicit_diffusion.h"
#include "dg/dg_schemes/implicit_convection/implicit_convection.h"
#include "dg/dg_schemes/implicit_diffusion/implicit_diffusion.h"
#include "unsupported/Eigen/IterativeSolvers"
#include "solver/eigen_sparse_solver.h"
#include "base/filesystem_manager.h"
#include "base/io.h"

typedef Eigen::SparseMatrix<double,Eigen::RowMajor> EigenSpMat;
typedef Eigen::Triplet<double> Triplet;



template<uInt Order>
void Run(uInt N, FilesystemManager fsm);

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

    // 获取各种路径
    std::cout << "Solution file: " << fsm.get_solution_file(0, 100) << "\n";
    std::cout << "Error log:     " << fsm.get_error_log_file() << "\n";
    std::cout << "Config file:   " << fsm.get_config_file() << "\n";
    std::cout << "Run info:      " << fsm.get_run_info_file() << "\n";
    std::cout << "CPU used:      " << cpus << "\n";
                             
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


    LongVector<DoFs> U_n(cmesh.m_cells.size());

    #pragma omp parallel for schedule(dynamic)
    for(uInt cellId=0;cellId<cmesh.m_cells.size();cellId++){
        const auto& cell = cmesh.m_cells[cellId];
        const auto& rhoU_coef = Basis::func2coef([&](vector3f Xi)->DenseMatrix<5,1>{
            return {rho_Xi(cell,Xi),rhou_Xi(cell,Xi),rhov_Xi(cell,Xi),rhow_Xi(cell,Xi),rhoe_Xi(cell,Xi)};
        });
        for(uInt k=0;k<Basis::NumBasis;k++){
            MatrixView<DoFs,1,5,1>(U_n[cellId],5*k,0) = rhoU_coef[k];
        }
    }
    
    logging("Set Init Value");
    std::ofstream fp;
    fp.open(fsm.get_solution_file(0, N));
    fp <<std::setprecision(16)<< ("#       U_n") << "  " 
        <<std::setprecision(16)<< (" y") << "  " 
        <<std::setprecision(16)<< (" z")
                << "  " <<std::setprecision(16)<<  (" rho")
                << "  " <<std::setprecision(16)<<  (" u")
                << "  " <<std::setprecision(16)<<  (" v")
                << "  " <<std::setprecision(16)<<  (" w")
                << "  " <<std::setprecision(16)<<  (" e") << std::endl;
    constexpr auto Qpoints = QuadC::get_points();
    constexpr auto Qweights = QuadC::get_weights();
    for(uInt cellId=0;cellId<cmesh.m_cells.size();cellId++){
        const auto& cell = cmesh.m_cells[cellId];
        for(uInt g=0; g<QuadC::num_points; ++g) {
            const auto& p = Qpoints[g];
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
    
    LongVector<DoFs> mass(U_n.size());
    for(uInt cellId=0;cellId<cmesh.m_cells.size();cellId++){
        DenseMatrix<DoFs,DoFs> mass_matrix;
        for(uInt g=0; g<QuadC::num_points; ++g) {
            const auto& p = Qpoints[g];
            auto phi = Basis::eval_all(p[0], p[1], p[2]);
            for(uInt i=0; i<Basis::NumBasis; ++i) {
                for(uInt k=0; k<5; ++k) {
                    mass[cellId](5*i + k, 0) += phi[i] * phi[i] * 
                        Qweights[g] * cmesh.m_cells[cellId].compute_jacobian_det();
                }
            }
        }
    }


    print(std::array<std::string,8>{"#       time", "rel.err  rho",
                    "rel.err  u", "rel.err  v", "rel.err  w", 
                    "rel.err  e", "rel.err coef", "cpu time"});
    Scalar total_time = 0.0;
    for(uInt save_step = 0; save_step < 400; save_step++){    
        Scalar Dt = 0.5;
        Scalar max_dt = Dt;// * std::pow((5.0/N),(Order+1));
        Scalar dt = max_dt;
        Scalar sub_t = 0;
        Scalar init_delta = 0.0, init_residula = 0.0;
        uInt kk = 0;
        do{
            /* 时间步长 dt 截断到 Dt 长度 */
            dt = std::min(max_dt, Dt-sub_t); 
            sub_t += dt;
            total_time += dt; // 实际物理时间

            /* 记录下旧时刻的解 */
            LongVector<DoFs> U_k = U_n;

            /* 记录初始迭代步的 残差、初始的增量，用于非线性迭代停机条件 */
            Scalar init_delta = 0.0;
            Scalar prev_delta = 0.0;
            Scalar init_residual = 0.0;
            LongVector<5*Basis::NumBasis> res_old(U_n.size());

            auto get_matrix = [&](BlockSparseMatrix<DoFs,DoFs>& sparse_mat, LongVector<DoFs>& rhs, LongVector<5 * Basis::NumBasis> &U_k){
                
                // BlockSparseMatrix<DoFs,DoFs> sparse_mat0;
                // LongVector<DoFs> rhs0(U_n.size());
                // diffusion_newton.assemble(cmesh, U_k, Dt * (kkkk) + sub_t + 0.5 * dt,
                //                     sparse_mat0, rhs0);
                // sparse_mat0.finalize();
                //print(sparse_mat0.multiply(U_k).dot(sparse_mat0.multiply(U_k)));
                // print(rhs0.dot(rhs0));
                // 用 保存的 U_k 代入、离散
                convection.assemble(cmesh, U_k, total_time, sparse_mat, rhs);

                // 在算子里面反转了一下，所以直接加法
                diffusion.assemble(cmesh, U_k, total_time, sparse_mat, rhs);
                // diffusion_newton.assemble(cmesh, U_k, Dt * (kkkk) + sub_t + 0.5 * dt,
                //                     sparse_mat, rhs);

                // rhs  = rhs + sparse_mat0.multiply(U_k)-rhs0;
                // 解的是全量型，所以 delta(N1+N2+N3) delta U_n = -(N1+N2)
                //  delta(N1+N2+N3) delta U_n = -(N1+N2)
                //  delta(N1+N2+N3) U_n = delta(N1+N2+N3) x0 -(N1+N2)

                // LongVector<DoFs> f(U_n.size());
                for(uInt cellId=0;cellId<cmesh.m_cells.size();cellId++){
                    // f[cellId] = 0.0;
                    for(uInt g=0; g<QuadC::num_points; ++g) {
                        const auto& p = Qpoints[g];
                        auto phi = Basis::eval_all(p[0], p[1], p[2]);
                        auto fu = fu_Xi(cmesh.m_cells[cellId],p,total_time);
                        auto fe = fe_Xi(cmesh.m_cells[cellId],p,total_time);
                        for(uInt i=0; i<Basis::NumBasis; ++i) {
                            // for(uInt k=0; k<5; ++k) {
                            rhs[cellId](5*i + 1, 0) += fu * phi[i] * Qweights[g] * cmesh.m_cells[cellId].compute_jacobian_det();
                            rhs[cellId](5*i + 4, 0) += fe * phi[i] * Qweights[g] * cmesh.m_cells[cellId].compute_jacobian_det();
                            // }
                        }
                    }
                }
                // rhs += 
                sparse_mat.finalize();
            };

            /* ======================================================= *\
            **   第三层迭代，关于 时间步 dt 内的 非线性迭代
            **   需要判断是否超过 Dt，超过了就截断
            \* ======================================================= */
            for(uInt picard_iter = 0; picard_iter < 2000; picard_iter++){
                
                logging("Start discretization");

                LongVector<5*Basis::NumBasis> rhs(U_n.size());
                BlockSparseMatrix<5*Basis::NumBasis,5*Basis::NumBasis> sparse_mat;
                /* 用 第 k 步迭代的解 U_k 进行离散 */ 
                get_matrix(sparse_mat, rhs, U_k);

                // sparse_mat.output_as_scalar("12345.txt");
                
                logging("Start linear solver");

                /* 调用 Eigen 求解线性方程组 */
                EigenSparseSolver<DoFs,DoFs> solver(sparse_mat,rhs);
                LongVector<DoFs> U_k_tmp = solver.SparseLU(U_k);
                
                logging("Linear solver finished");

                if(picard_iter>3){
                    LongVector<DoFs> rhs(U_n.size());
                    BlockSparseMatrix<DoFs,DoFs> sparse_mat;
                    get_matrix(sparse_mat, rhs, U_k_tmp);
                    const auto& res_new = (sparse_mat.multiply(U_k_tmp)) - (rhs);
                    Scalar a = 0;
                    a = (res_new-res_old).dot(res_new)/(res_new-res_old).dot(res_new-res_old);
                    a = std::max(a,-5.0);
                    a = std::min(a,0.1);

                    // a = -0.5;
                    
                    U_k_tmp = a*U_k + (1-a)*U_k_tmp;
                    // debug(a);
                    // res_old = res_new;
                }
                
                // /* 降维模拟的后处理 */
                // for(uInt cid = 0; cid < cmesh.m_cells.size(); cid++){
                //     ddx[cid][3 + 5*0] = 0;
                //     for (uInt k = 1; k < Basis::NumBasis; k++) {
                //         ddx[cid][3 + 5*k] = 0;
                //     }
                // }

                // /* 施加限制器，这里是只有保极值保正，需要先计算允许的 Min Max */
                // pwenolimiter.apply(ddx, dx); 
                // positivelimiter.constructMinMax(ddx, dx); 
                // positivelimiter.apply(ddx, dx); 

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
                Scalar rate_delta = delta/init_delta;  // 下降率充分大
                Scalar rel_delta = delta/prev_delta;   // 相对变化率足够小
                prev_delta = delta;

                /* 计算非线性残差，但没有用于停机条件 */
                std::ostringstream oss;
                oss << delta;
                logging("Re-discretize, calculate nonlinear residuals");
                {
                    LongVector<5*Basis::NumBasis> rhs(U_n.size());
                    BlockSparseMatrix<5*Basis::NumBasis,5*Basis::NumBasis> sparse_mat;
                    get_matrix(sparse_mat, rhs, U_k);
                    res_old = sparse_mat.multiply(U_k) - rhs;
                    std::ostringstream oss;
                    oss << std::sqrt(res_old.dot(res_old));
                    logging("Picard iter " + std::to_string(picard_iter) + "\t ||N(U_k)|| = " + oss.str());

                }
                logging("Picard iter " + std::to_string(picard_iter) + "\t ||delta U|| = " + oss.str());
                if(delta < 1e-8 || rel_delta < 1e-8 || (picard_iter>10 && rate_delta > 0.99)) break;
            }
            /* 非线性迭代结束，U_n 赋值为最后的 U_k，作为新的时间步 U_{n+1} */ 
            U_n = U_k;

            logging("Iter  " + std::to_string(save_step+1) + " \t Total time: " + std::to_string(total_time));
        }while(sub_t < Dt);


        const std::string& filename = fsm.get_solution_file_h5(save_step+1, N);
        save_DG_solution_to_hdf5<QuadC,Basis>(cmesh, U_n, filename,total_time,0);

    }
    
}
