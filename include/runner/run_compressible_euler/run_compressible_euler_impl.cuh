#pragma once
#include "base/type.h"
#include "base/exact.h"
#include "base/filesystem_manager.h"
#include "base/logger_system.h"
#include "base/io.h"
#include "mesh/mesh.h"
#include "matrix/matrix.h"

#include "dg/dg_basis/dg_basis.h"
#include "dg/dg_schemes/explicit_convection_gpu/explicit_convection_gpu.cuh"
#include "dg/dg_limiters/positive_limiters/positive_limiter_gpu.cuh"
#include "dg/dg_limiters/weno_limiters/weno_limiter_gpu.cuh"
#include "dg/dg_limiters/weno_limiters/pweight_weno_limiter_gpu.cuh"

#include "mesh/device_mesh.cuh"
#include "dg/time_integrator.cuh"
#include "runner/run_compressible_euler/run_compressible_euler_interface.cuh"
#include "runner/run_compressible_euler/cfl_tools.cuh"




template<uInt Order,typename FluxType, bool OnlyNeigbAvg>
void RunCompressibleEuler(uInt N, FilesystemManager& fsm, LoggerSystem& logger, uInt limiter_flag){

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
    using QuadF = typename AutoQuadSelector<Basis::OrderBasis, GaussLegendreTri::Auto>::type;

    constexpr uInt DoFs = 5*Basis::NumBasis;
    Scalar param_gamma = FluxType::get_gamma();
    if (param_gamma - get_gamma() > 1e-4){
        printf("数值通量的 \\gamma = %.4f, 而解析解提供的 \\gamma = %.4f", param_gamma, get_gamma());
        return;
    }
    /* ======================================================= *\
    **   算子 和 限制器 的实例化
    \* ======================================================= */
    ExplicitConvectionGPU<Basis::OrderBasis,FluxType> convection;
    PositiveLimiterGPU<Basis::OrderBasis, QuadC, QuadF, OnlyNeigbAvg> positivelimiter(gpu_mesh, param_gamma);
    WENOLimiterGPU<Basis::OrderBasis, QuadC, QuadF> wenolimiter(gpu_mesh);
    PWeightWENOLimiterGPU<Basis::OrderBasis, QuadC, QuadF> pweight_wenolimiter(gpu_mesh);
    
    
    
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
    Scalar total_time = 0.0;
    Scalar final_time = get_final_time();
    std::vector<Scalar> save_time = get_save_time();
    
    Scalar CFL = get_CFL(0);
    Scalar dt = compute_CFL_time_step<Order, QuadC, Basis>(cmesh, gpu_mesh, gpu_U_n, CFL, param_gamma);

    for(const auto& p : save_time) std::cout<<std::setw(6)<<p<<"  "; std::cout<<std::endl;

    uInt save_index = 0;
    uInt iter = 0;
    TimeIntegrator<DoFs,Order,OnlyNeigbAvg> time_integrator(gpu_mesh,gpu_U_n,gpu_r_mass,positivelimiter,wenolimiter,pweight_wenolimiter);
    time_integrator.set_scheme(get_time_intergrator_scheme());
    logger.log_explicit_step(uInt(-1), 0.0, 0.0, 0.0);
    while (total_time < final_time) {
        CFL = get_CFL(iter);
        if (iter < 3000 || iter % 1000 == 0) 
        dt = compute_CFL_time_step<Order, QuadC, Basis>(cmesh, gpu_mesh, gpu_U_n, CFL, param_gamma);
        Scalar curr_dt = dt;
        // 截断到下一个 save_time 保证不会错过保存时间点
        if (save_index < save_time.size() && total_time + dt > save_time[save_index])
            curr_dt = save_time[save_index] - total_time;
        if (total_time + curr_dt > final_time)
            curr_dt = final_time - total_time;
        // positivelimiter.constructMinMax(gpu_U_n);
        time_integrator.advance(convection, total_time, curr_dt, limiter_flag);
        // wenolimiter.apply(gpu_U_n);
        // positivelimiter.apply(gpu_U_n);
        total_time += curr_dt;
        iter++;
        // cudaDeviceSynchronize();
        if(logger.log_explicit_step(iter, total_time, curr_dt, save_time[save_index])){
            const std::string& filename = fsm.get_solution_file_h5(save_index+1, N);
            logger.log_save_solution(iter, total_time, filename);
            save_DG_solution_to_hdf5<QuadC,Basis>(cmesh, gpu_U_n.download(), filename,total_time,iter);
            save_index++;
        }
        /*
        Scalar h_tt, h_dt;
        cudaMemcpy(&h_tt, &total_time, 1 * sizeof(Scalar), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_dt, &dt, 1 * sizeof(Scalar), cudaMemcpyDeviceToHost);
        if(logger.log_explicit_step(iter, h_tt, h_dt, save_time[save_index])){
            const std::string& filename = fsm.get_solution_file_h5(save_index+1, N);
            logger.log_save_solution(iter, h_tt, filename);
            save_DG_solution_to_hdf5<QuadC,Basis>(cmesh, gpu_U_n.download(), filename,h_tt,iter);
            save_index++;
        }
        */

    }
}


