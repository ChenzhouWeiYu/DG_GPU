#include "base/type.h"
// #include "base/exact.h"
#include "base/filesystem_manager.h"
#include "base/logger_system.h"
#include "base/io.h"
#include "mesh/mesh.h"
#include "matrix/matrix.h"
#include "dg/dg_basis/dg_basis.h"
#include "dg/dg_schemes/explicit_convection.h"
#include "dg/dg_schemes/explicit_convection_gpu.h"
// #include "dg/dg_schemes/implicit_diffusion.h"
// #include "dg/dg_limiters/positive_limiter.h"
#include "dg/dg_limiters/positive_limiter_gpu.h"
#include "dg/dg_limiters/weno_limiter_gpu.h"
// #include "DG/DG_Schemes/PWENOLimiter.h"
#include "solver/eigen_sparse_solver.h"
#include "mesh/device_mesh.h"

#include "dg/time_integrator.h"

#include "problem.h"
#include "tools.h"
// #include "save_to_hdf5.h"



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







template<uInt Order,typename FluxType>
void Run(uInt N, FilesystemManager& fsm, LoggerSystem& logger){

    logger.log_section_title("Setup Stage");

    logger.start_stage("Split Hex Mesh to Tet");

    const auto& cmesh = create_mesh(N);
    check_mesh(cmesh);
    DeviceMesh gpu_mesh;
    gpu_mesh.initialize_from(cmesh);  // 这部分完全 CPU 逻辑
    gpu_mesh.upload_to_gpu();   

    logger.end_stage();

    logger.print_mesh_info(gpu_mesh);



    using Basis = DGBasisEvaluator<Order>;
    using QuadC = typename AutoQuadSelector<Basis::OrderBasis, GaussLegendreTet::Auto>::type;
    constexpr uInt DoFs = 5*Basis::NumBasis;
    /* ======================================================= *\
    **   算子 和 限制器 的实例化
    \* ======================================================= */
    // ExplicitConvection<Basis::OrderBasis,FluxType> CPUconvection;
    ExplicitConvectionGPU<Basis::OrderBasis,FluxType> convection;

    // PositiveLimiter<Basis::OrderBasis, QuadC, false> CPUpositivelimiter(cmesh, param_gamma);
    PositiveLimiterGPU<Basis::OrderBasis, QuadC, false> positivelimiter(gpu_mesh, param_gamma);
    WENOLimiterGPU<Basis::OrderBasis, QuadC> wenolimiter(gpu_mesh);
    
    




    
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
    // CPUpositivelimiter.constructMinMax(U_n, U_n); 
    // CPUpositivelimiter.apply(U_n, U_n); 

    LongVectorDevice<DoFs> gpu_U_n = U_n.to_device();
    positivelimiter.constructMinMax(gpu_U_n); 
    positivelimiter.apply(gpu_U_n); 

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

    logger.log_section_title("Time Marching");
    Scalar total_time = 0.0, final_time = (init_x0y0[0] == 0.8 ? 0.8 : 0.3);
    std::vector<Scalar> save_time;
    for(uInt i=0; i<10; ++i) {
        save_time.push_back((i+1) * final_time * 0.1 );
    }


    for(const auto& p : save_time) std::cout<<std::setw(6)<<p<<"  "; std::cout<<std::endl;

    uInt save_index = 0;
    uInt iter = 0;
    TimeIntegrator<DoFs,Order> time_integrator(gpu_mesh,gpu_U_n,gpu_r_mass,0.5);
    // time_integrator.set_scheme(TimeIntegrationScheme::SSP_RK3);
    logger.log_explicit_step(uInt(-1), 0.0, 0.0, 0.0);
    while (total_time < final_time) {
        Scalar dt = compute_CFL_time_step<Order, QuadC, Basis>(
            cmesh, gpu_mesh, gpu_U_n, 0.5, param_gamma);
        // if (iter<100) dt *= 1e-2;

        // 截断到下一个 save_time 保证不会错过保存时间点
        if (save_index < save_time.size() && total_time + dt > save_time[save_index])
            dt = save_time[save_index] - total_time;

        if (total_time + dt > final_time)
            dt = final_time - total_time;
        
        positivelimiter.constructMinMax(gpu_U_n);
        time_integrator.advance(convection,total_time,dt);
        wenolimiter.apply(gpu_U_n);
        positivelimiter.apply(gpu_U_n);
        // positivelimiter.apply_2(gpu_U_n);

        total_time += dt;
        iter++;
        if(logger.log_explicit_step(iter, total_time, dt, save_time[save_index])){
            const std::string& filename = fsm.get_solution_file_h5(save_index+1, N);
            logger.log_save_solution(iter, total_time, filename);
            save_DG_solution_to_hdf5<QuadC,Basis>(cmesh, gpu_U_n.download(), filename,total_time,iter);
            save_index++;
        }

    }

    return ; // 

}


