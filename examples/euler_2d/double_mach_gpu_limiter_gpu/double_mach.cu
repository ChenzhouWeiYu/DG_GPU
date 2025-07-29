#include "base/type.h"
#include "base/filesystem_manager.h"
#include "base/logger_system.h"
#include "base/io.h"
#include "mesh/mesh.h"
#include "matrix/matrix.h"
#include "dg/dg_basis/dg_basis.h"
#include "dg/dg_schemes/explicit_convection.h"
#include "dg/dg_schemes/explicit_convection_gpu.h"
// #include "dg/dg_schemes/implicit_diffusion.h"
#include "DG/DG_Schemes/PositiveLimiter.h"
#include "DG/DG_Schemes/PositiveLimiterGPU.h"
// #include "DG/DG_Schemes/PWENOLimiter.h"
#include "solver/eigen_sparse_solver.h"
#include "mesh/device_mesh.h"

#include "time_integrator.h"

#include "problem.h"
#include "tools.h"
// #include "save_to_hdf5.h"

template<typename T>
T copy_device_to_host(const T* device_ptr) {
    T host_data;
    cudaMemcpy(&host_data, device_ptr, sizeof(T), cudaMemcpyDeviceToHost);
    return host_data;
}

__global__ void mat_kernel(const DenseMatrix<3,3>* inputs, DenseMatrix<3,3>* output) {
    
    const auto A = *inputs;
    const auto At = A.transpose();
    *output = At.multiply(A);
}

template<uInt DoFs>
__global__ void update_solution(DenseMatrix<DoFs,1>* U_n,
                                 const DenseMatrix<DoFs,1>* U_1,
                                 const DenseMatrix<DoFs,1>* r_mass,
                                 Scalar dt, uInt size)
{
    uInt cellId = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellId >= size) return;

    for (int i = 0; i < DoFs; ++i) {
        U_n[cellId](i,0) -= dt * r_mass[cellId](i,0) * U_1[cellId](i,0);
        if(i%5==3) U_n[cellId](i,0) = 0.0;
    }
}


template<uInt DoFs, uInt Order>
void Eular_Step(ExplicitConvectionGPU<Order>& convection, Scalar curr_time,
                const DeviceMesh& gpu_mesh, Scalar dt,
                LongVectorDevice<DoFs>& gpu_U_n,
                const LongVectorDevice<DoFs>& gpu_r_mass) 
{
    // LongVector<DoFs> U_1(gpu_U_n.size());
    // LongVectorDevice<DoFs> gpu_U_1 = U_1.to_device();
    LongVectorDevice<DoFs> gpu_U_1(gpu_U_n.size());

    // print(gpu_U_1.download().norm());
    // print(gpu_U_n.download().norm());
    convection.eval(gpu_mesh, gpu_U_n, gpu_U_1, curr_time);

    // print(gpu_U_1.download().norm());
    // print(gpu_U_n.download().norm());
    const int num_blocks = (gpu_U_n.size() + 31) / 32;
    update_solution<DoFs><<<num_blocks, 32>>>(gpu_U_n.d_blocks, gpu_U_1.d_blocks, gpu_r_mass.d_blocks, dt, gpu_U_n.size());
    cudaDeviceSynchronize();
    // print(gpu_U_1.download().norm());
    // print(gpu_U_n.download().norm());
}


template<uInt Order, typename QuadC, typename Basis>
__global__ void reconstruct_and_speed_kernel(
    const GPUTetrahedron* d_cells, uInt num_cells,
    const DenseMatrix<5 * Basis::NumBasis, 1>* coef,
    DenseMatrix<5 * QuadC::num_points, 1>* U_h,
    Scalar* wave_speeds,
    Scalar gamma)
{
    const uInt cellId = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellId >= num_cells) return;

    constexpr auto Qpoints = QuadC::get_points();
    constexpr auto Qweights = QuadC::get_weights();

    const DenseMatrix<5 * Basis::NumBasis, 1>& coef_cell = coef[cellId];
    DenseMatrix<5 * QuadC::num_points, 1> U_reconstructed;
    Scalar lambda = 0.0;

    for (uInt q = 0; q < QuadC::num_points; ++q) {
        auto Uq = DGBasisEvaluator<Order>::template coef2filed<5, Scalar>(coef_cell, Qpoints[q]);

        for (int i = 0; i < 5; ++i)
            U_reconstructed(5 * q + i, 0) = Uq(i, 0);

        Scalar rho = Uq(0, 0), rhou = Uq(1, 0), rhov = Uq(2, 0), rhow = Uq(3, 0), rhoE = Uq(4, 0);
        Scalar ke = (rhou*rhou + rhov*rhov + rhow*rhow) / (2.0 * rho + 1e-12);
        Scalar p = (gamma - 1.0) * (rhoE - ke);
        Scalar a = sqrt(gamma * p / rho);
        Scalar u = rhou / rho, v = rhov / rho, w = rhow / rho;
        Scalar vel = sqrt(u*u + v*v + w*w);
        lambda += (a + vel) * Qweights[q] * 6.0;
    }

    U_h[cellId] = U_reconstructed;
    wave_speeds[cellId] = lambda;
}


template<uInt Order, typename QuadC, typename Basis>
Scalar compute_CFL_time_step(
    const ComputingMesh& cpu_mesh,
    const DeviceMesh& gpu_mesh,
    const LongVectorDevice<5 * Basis::NumBasis>& coef_device,
    Scalar CFL,
    Scalar gamma)
{
    const uInt num_cells = gpu_mesh.num_cells();
    const uInt Q = QuadC::num_points;

    // 分配 GPU 缓存
    LongVectorDevice<5 * Q> U_h(num_cells);
    Scalar* d_wave_speeds;
    cudaMalloc(&d_wave_speeds, num_cells * sizeof(Scalar));

    // 启动 kernel：重构 & 波速估计
    dim3 block(128);
    dim3 grid((num_cells + block.x - 1) / block.x);
    reconstruct_and_speed_kernel<Order, QuadC, Basis>
        <<<grid, block>>>(gpu_mesh.device_cells(), num_cells,
                          coef_device.d_blocks,
                          U_h.d_blocks,
                          d_wave_speeds,
                          gamma);

    // 下载结果
    std::vector<Scalar> h_wave_speeds(num_cells);
    cudaMemcpy(h_wave_speeds.data(), d_wave_speeds,
               num_cells * sizeof(Scalar), cudaMemcpyDeviceToHost);

    cudaFree(d_wave_speeds);

    // h / lambda 最小值
    Scalar min_dt = std::numeric_limits<Scalar>::max();
    for (uInt i = 0; i < num_cells; ++i) {
        Scalar h_i = cpu_mesh.m_cells[i].m_h;
        Scalar lam = h_wave_speeds[i];
        if (lam > 1e-12)
            min_dt = std::min(min_dt, h_i / lam);
    }

    Scalar dt = CFL * min_dt / std::pow(2 * Order + 1, 1); // k=1
    return dt;
}






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
    logger.print_header("Discontinuous Galerkin Simulation");
    logger.print_config(order, meshN, cpus);
                             
    if(order == 1) Run<1>(meshN, fsm, logger);
    if(order == 2) Run<2>(meshN, fsm, logger);
    if(order == 3) Run<3>(meshN, fsm, logger);
    // if(order == 4) Run<4>(meshN);
    // if(order == 5) Run<5>(meshN);
}












template<uInt Order>
void Run(uInt N, FilesystemManager& fsm, LoggerSystem& logger){

    logger.log_section_title("Setup Stage");

    logger.start_stage("Split Hex Mesh to Tet");

    const auto& cmesh = create_mesh(N);
    check_mesh(cmesh);
    DeviceMesh gpu_mesh;
    // print("000000");
    gpu_mesh.initialize_from(cmesh);  // 这部分完全 CPU 逻辑
    // print("1111");
    gpu_mesh.upload_to_gpu();            // 上传
    // print("22222");

    logger.end_stage();
    // return ; // 只测试网格划分

    logger.print_mesh_info(gpu_mesh);



    using Basis = DGBasisEvaluator<Order>;
    using QuadC = typename AutoQuadSelector<Basis::OrderBasis, GaussLegendreTet::Auto>::type;
    constexpr uInt DoFs = 5*Basis::NumBasis;
    /* ======================================================= *\
    **   算子 和 限制器 的实例化
    \* ======================================================= */
    ExplicitConvection<Basis::OrderBasis> CPUconvection;
    ExplicitConvectionGPU<Basis::OrderBasis> convection;
    // ImplicitDiffusion<Basis::OrderBasis> diffusion(param_mu);

    /* 这个WENO是错的 */
    // OrthoPWENOLimiter<Basis::OrderBasis, QuadC> pwenolimiter(cmesh);
    /*  这个是保极值、保正，第三个参数是 Min 和 Max 的策略     *\
          true 采用相邻的均值作为 Min Max，更宽松，开销低
    \*    false 为所有积分点的 Min Max，更紧致，开销大        */
    PositiveLimiter<Basis::OrderBasis, QuadC, false> CPUpositivelimiter(cmesh, param_gamma);
    PositiveLimiterGPU<Basis::OrderBasis, QuadC, false> positivelimiter(gpu_mesh, param_gamma);
    
    




    
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
    CPUpositivelimiter.constructMinMax(U_n, U_n); 
    DenseMatrix<5,1> U_max{8.0,8*8.25*std::cos(-M_PI/6.0),0.0,0.0,rhoe_xyz({0.1,0.1,0.0},0.0)};
    DenseMatrix<5,1> U_min{1.4,0.0,8*8.25*std::sin(-M_PI/6.0),0.0,rhoe_xyz({2.1,0.8,0.0},0.0)};
    #pragma omp parallel for schedule(dynamic)
    for(uInt cellId = 0; cellId < U_n.size(); cellId++){
        for(uInt k = 0; k < 5; k++){
            U_n[cellId][k] = std::min(std::max(U_n[cellId][k],U_min[k]),U_max[k]);
            CPUpositivelimiter.per_cell_max[cellId][k] = std::min(CPUpositivelimiter.per_cell_max[cellId][k], U_max[k]);
            CPUpositivelimiter.per_cell_min[cellId][k] = std::max(CPUpositivelimiter.per_cell_min[cellId][k], U_min[k]);
        }
        // DenseMatrix<5,1> U_n_avg{U_n[cellId][0],U_n[cellId][1],U_n[cellId][2],U_n[cellId][3],U_n[cellId][4]};
        // U_n_avg = std::min(std::max(U_n_avg,U_min),U_max);
        // U_n[cellId][0] = U_n_avg[0];
        // U_n[cellId][1] = U_n_avg[1];
        // U_n[cellId][2] = U_n_avg[2];
        // U_n[cellId][3] = U_n_avg[3];
        // U_n[cellId][4] = U_n_avg[4];
        // CPUpositivelimiter.per_cell_max[cellId] = std::min(CPUpositivelimiter.per_cell_max[cellId],U_max);
        // CPUpositivelimiter.per_cell_min[cellId] = std::max(CPUpositivelimiter.per_cell_min[cellId],U_min);
        // CPUpositivelimiter.per_cell_max[cellId][0] = std::min(CPUpositivelimiter.per_cell_max[cellId][0], 8.0);
        // CPUpositivelimiter.per_cell_max[cellId][1] = std::min(CPUpositivelimiter.per_cell_max[cellId][1], 8*8.25*std::cos(-M_PI/6.0));
        // CPUpositivelimiter.per_cell_max[cellId][2] = std::min(CPUpositivelimiter.per_cell_max[cellId][2], 0.0);
        // CPUpositivelimiter.per_cell_max[cellId][3] = std::min(CPUpositivelimiter.per_cell_max[cellId][3], 0.0);
        // CPUpositivelimiter.per_cell_max[cellId][4] = std::min(CPUpositivelimiter.per_cell_max[cellId][4], rhoe_xyz({0.1,0.1,0.0},0.0));
        // CPUpositivelimiter.per_cell_min[cellId][0] = std::max(CPUpositivelimiter.per_cell_min[cellId][0], 1.4);
        // CPUpositivelimiter.per_cell_min[cellId][1] = std::max(CPUpositivelimiter.per_cell_min[cellId][1], 0.0);
        // CPUpositivelimiter.per_cell_min[cellId][2] = std::max(CPUpositivelimiter.per_cell_min[cellId][2], 8*8.25*std::sin(-M_PI/6.0));
        // CPUpositivelimiter.per_cell_min[cellId][3] = std::max(CPUpositivelimiter.per_cell_min[cellId][3], 0.0);
        // CPUpositivelimiter.per_cell_min[cellId][4] = std::max(CPUpositivelimiter.per_cell_min[cellId][4], rhoe_xyz({2.1,0.8,0.0},0.0));
    }
    CPUpositivelimiter.apply(U_n, U_n); 

    LongVectorDevice<DoFs> gpu_U_n = U_n.to_device();
    // positivelimiter.constructMinMax(gpu_U_n); 
    // positivelimiter.apply(gpu_U_n); 

    logger.end_stage();


    /* ======================================================= *\
    **   计算 (\phi_i, \phi_i) 作为质量矩阵
    **   正交基，只需要计算、保存对角元。  r_mass 表示是 倒数
    \* ======================================================= */
    LongVector<DoFs> r_mass(U_n.size());
    for(uInt cid=0;cid<cmesh.m_cells.size();cid++){
        const auto& detJac = cmesh.m_cells[cid].compute_jacobian_det();
        for(uInt i=0; i<Basis::NumBasis; ++i) {
            Scalar val = 0.0;
            const auto& Qpoints = QuadC::get_points();
            const auto& Qweights = QuadC::get_weights();
            for(uInt g=0; g<QuadC::num_points; ++g) {
                const auto& weight = Qweights[g] * detJac;
                const auto& p = Qpoints[g];
                auto phi = Basis::eval_all(p[0], p[1], p[2]);
                val += phi[i] * phi[i] * weight;
            }
            for(uInt k=0; k<5; ++k) {
                r_mass[cid](5*i + k, 0) = 1.0/val;
            }
        }
    }
    
    LongVectorDevice<DoFs> gpu_r_mass = r_mass.to_device();
    
    U_n = gpu_U_n.download();
    save_DG_solution_to_hdf5<QuadC,Basis>(cmesh, U_n, fsm.get_solution_file_h5(0, N));
    // return ;

    // const LongVector<DoFs>& cpu_next = CPUconvection.eval(cmesh,U_n,0.0);
    // LongVector<DoFs> U_1(gpu_U_n.size());
    // LongVectorDevice<DoFs> gpu_U_1 = U_1.to_device();
    // convection.eval(gpu_mesh, gpu_U_n, gpu_U_1);
    // const LongVector<DoFs>& gpu_next = gpu_U_1.download();

    // print((cpu_next-gpu_next).norm());
    // print(cpu_next-gpu_next);
    // // print(gpu_next);

    // return;

    logger.log_section_title("Time Marching");
    Scalar total_time = 0.0, final_time = 0.2;
    std::vector<Scalar> save_time;
    for(uInt i=0; i<20; ++i) {
        save_time.push_back((i+1) * 0.01);
    }
    // for(uInt i=0; i<10; ++i) {
    //     save_time[i] = (i+1) * (0.1/11.0);
    // }
    // for(uInt i=10; i<20; ++i) {
    //     save_time[i] = (i-9) * 0.1;
    // }
    for(const auto& p : save_time) std::cout<<std::setw(6)<<p<<"  "; std::cout<<std::endl;

    uInt save_index = 0;
    uInt iter = 0;
    TimeIntegrator<DoFs,Order> time_integrator(gpu_mesh,gpu_U_n,gpu_r_mass,0.5);
    // time_integrator.set_scheme(TimeIntegrationScheme::SSP_RK3);
    logger.log_explicit_step(uInt(-1), 0.0, 0.0, 0.0);
    while (total_time < final_time) {
        Scalar dt = compute_CFL_time_step<Order, QuadC, Basis>(
            cmesh, gpu_mesh, gpu_U_n, 0.5, param_gamma);

        // 截断到下一个 save_time 保证不会错过保存时间点
        if (save_index < save_time.size() && total_time + dt > save_time[save_index])
            dt = save_time[save_index] - total_time;

        if (total_time + dt > final_time)
            dt = final_time - total_time;
        positivelimiter.constructMinMax(gpu_U_n);
        // {
        //     const auto& cpu_max = positivelimiter.d_per_cell_max.download();
        //     const auto& cpu_min = positivelimiter.d_per_cell_min.download();
        //     const auto& cpu_sol = reconstruct_solution<QuadC,Basis>(cmesh,gpu_U_n.download());
        //     for(uInt cellId=0;cellId<cmesh.m_cells.size();cellId++){
        //         Scalar x = cmesh.m_cells[cellId].m_centroid[0], y = cmesh.m_cells[cellId].m_centroid[1];
        //         if(cpu_max[cellId][0]>8.01 < cpu_max[cellId][1]>57.17 || cpu_min[cellId][2] < -33.01)
        //         printf("(x,y)=(%5.2lf,%5.2lf), rho(%5.2lf,%5.2lf,%5.2lf), u(%5.2lf,%5.2lf,%5.2lf), v(%5.2lf,%5.2lf,%5.2lf), rhou(%6.2lf,%6.2lf,%6.2lf), rhov(%6.2lf,%6.2lf,%6.2lf)\n", x,y,
        //                 cpu_max[cellId][0],cpu_sol[cellId][0],cpu_min[cellId][0],
        //                 cpu_max[cellId][1]/cpu_max[cellId][0],cpu_sol[cellId][1]/cpu_sol[cellId][0],cpu_min[cellId][1]/cpu_min[cellId][0],
        //                 cpu_max[cellId][2]/cpu_min[cellId][0],cpu_sol[cellId][2]/cpu_sol[cellId][0],cpu_min[cellId][2]/cpu_max[cellId][0],
        //                 cpu_max[cellId][1],cpu_sol[cellId][1],cpu_min[cellId][1],
        //                 cpu_max[cellId][2],cpu_sol[cellId][2],cpu_min[cellId][2]);
        //     }
        // }
        // Eular_Step(convection, total_time, gpu_mesh,    dt, gpu_U_n, gpu_r_mass);
        time_integrator.advance(convection,total_time,dt);
        positivelimiter.apply(gpu_U_n);

        total_time += dt;
        iter++;
        if(logger.log_explicit_step(iter, total_time, dt, save_time[save_index])){
            const std::string& filename = fsm.get_solution_file_h5(save_index+1, N);
            logger.log_save_solution(iter, total_time, filename);
            save_DG_solution_to_hdf5<QuadC,Basis>(cmesh, gpu_U_n.download(), filename,total_time,iter);
            save_index++;
        }

        // // 按迭代步保存
        // if (save_time.size() == 0 &&  output_interval > 0 && iter % output_interval == 0) {
        //     U_n = gpu_U_n.download();
        //     print("按步数保存成功: iter=" + std::to_string(iter) + " time=" + std::to_string(total_time));
        //     save_Uh_Us<QuadC,Basis>(cmesh,U_n,total_time,fsm.get_solution_file(0, N));
        //     // break;
        // }

        // // 按时间保存
        // if (save_index < save_time.size() && total_time >= save_time[save_index] - 1e-12) {
        //     U_n = gpu_U_n.download();
        //     print("按时间保存成功: time=" + std::to_string(save_time[save_index]));
        //     // save_Uh_Us<QuadC,Basis>(cmesh,U_n,total_time,fsm.get_solution_file(save_index+1, N));
        //     save_DG_solution_to_hdf5<QuadC,Basis>(cmesh, U_n, fsm.get_solution_file_h5(save_index+1, N),total_time,iter);
        //     save_index++;
        // }
    }



    return ; // 

}


