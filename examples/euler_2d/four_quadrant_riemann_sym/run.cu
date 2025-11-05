#include "base/type.h"
#include "base/exact.h"
#include "base/filesystem_manager.h"
#include "base/logger_system.h"
#include "base/io.h"
#include "mesh/mesh.h"
#include "matrix/matrix.h"

#include "dg/dg_basis/dg_basis.h"
#include "dg/dg_flux/euler_physical_flux.h"
#include "dg/dg_schemes/explicit_convection_gpu/explicit_convection_gpu.cuh"
#include "dg/dg_schemes/explicit_convection_gpu/explicit_convection_gpu_impl.cuh"
#include "dg/dg_schemes/explicit_convection_gpu/explicit_convection_gpu_cells_impl.cuh"
#include "dg/dg_schemes/explicit_convection_gpu/explicit_convection_gpu_faces_impl.cuh"

#include "dg/condition/condition_interface.h"
#include "dg/condition/double_mach.h"

#include "dg/physics/physics_base.h"
#include "dg/physics/ideal_gas_physics.h"
#include "dg/flux_schemes/flux_scheme_base.h"
#include "dg/flux_schemes/lax_friedrichs_flux.h"
#include "dg/flux_schemes/rotated_flux_scheme.h"
#include "dg/flux_schemes/stabilized_flux.h"
#include "dg/flux_schemes/hllc_flux.h"
#include "dg/flux_schemes/rsir_flux.h"

#include "dg/dg_limiters/positive_preserving_limiters/positive_preserving_limiter_gpu.cuh"
#include "dg/dg_limiters/positive_preserving_limiters/positive_preserving_limiter_gpu_impl.cuh"
#include "dg/dg_limiters/positive_preserving_limiters/positive_preserving_limiter_gpu_kernels.cuh"

#include "mesh/device_mesh.cuh"
#include "runner/run_compressible_euler/cfl_tools.cuh"

template<uInt Order, typename NumFlux, uInt Level, uInt WithEntropy>
void RunCompressibleEuler(uInt N, FilesystemManager& fsm, LoggerSystem& logger);
ComputingMesh create_mesh(uInt N);


template<typename Physics>
class IBCondition : public ConditionInterface<IBCondition<Physics>, Physics> {
public:
    using Base = ConditionInterface<IBCondition<Physics>, Physics>;
    using Base::Base;       // 构造函数
    using Base::computeRho;
    using Base::computeU;
    using Base::computeV;
    using Base::computeW;
    using Base::computeP;
    using Base::computeT;
    using Base::computeE;
    // using Base::physics_;

    HostDevice __forceinline__
    Scalar rhoImpl(const vector3f& xyz, Scalar t) const {
        return rho_xyz(xyz, t);
    }

    HostDevice __forceinline__
    Scalar uImpl(const vector3f& xyz, Scalar t) const {
        return u_xyz(xyz, t);
    }

    HostDevice __forceinline__
    Scalar vImpl(const vector3f& xyz, Scalar t) const {
        return v_xyz(xyz, t);
    }

    HostDevice __forceinline__
    Scalar wImpl(const vector3f& xyz, Scalar t) const {
        return w_xyz(xyz, t);
    }

    HostDevice __forceinline__
    Scalar pImpl(const vector3f& xyz, Scalar t) const {
        return p_xyz(xyz, t);
    }

    // e 由 p 推导
    HostDevice __forceinline__
    Scalar eImpl(const vector3f& xyz, Scalar t) const {
        Scalar rho = computeRho(xyz, t);
        Scalar u   = computeU(xyz, t);
        Scalar v   = computeV(xyz, t);
        Scalar w   = computeW(xyz, t);
        Scalar p   = computeP(xyz, t);
        // printf("p = %f gamma = %f\n", p, this->physics_.get_gamma());
        return p / (this->physics_.get_gamma() - 1) / rho + Scalar(0.5)*(u*u + v*v + w*w);
    }

    // computeImpl：组装守恒变量
    HostDevice __forceinline__
    DenseMatrix<5,1> computeImpl(const vector3f& xyz, Scalar t) const {
        Scalar rho = computeRho(xyz, t);
        Scalar u   = computeU(xyz, t);
        Scalar v   = computeV(xyz, t);
        Scalar w   = computeW(xyz, t);
        Scalar e   = computeE(xyz, t);
        return {rho, rho*u, rho*v, rho*w, rho*e};
    }
};


Scalar get_CFL(uInt iter){
    if (iter < 1000){
        return 0.5 * 0.001;
    }
    if (iter < 2000){
        return 0.5 * 0.01;
    }
    if (iter < 3000){
        return 0.5 * 0.1;
    }
    return 0.5;
}

Scalar get_final_time() {
    return 0.8;
}


std::vector<Scalar> get_save_time(){
    std::vector<Scalar> save_time;
    save_time.push_back(get_final_time() * 0.001 );
    save_time.push_back(get_final_time() * 0.003 );
    save_time.push_back(get_final_time() * 0.006 );
    save_time.push_back(get_final_time() * 0.01 );
    save_time.push_back(get_final_time() * 0.03 );
    save_time.push_back(get_final_time() * 0.06 );
    for(uInt i=0; i<10; ++i) {
        save_time.push_back((i+1) * get_final_time() * 0.1 );
    }
    return save_time;
}


template<uInt DoFs>
__global__ void update_solution(
    DenseMatrix<DoFs, 1>* U_n,
    const DenseMatrix<DoFs, 1>* R_in,
    const DenseMatrix<DoFs, 1>* r_mass,
    Scalar dt, uInt size)
{
    uInt cellId = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellId >= size) return;

    for (int i = 0; i < DoFs; ++i) {
        // Euler 更新，使用质量矩阵逆（r_mass）
        U_n[cellId](i, 0) -= dt * r_mass[cellId](i, 0) * R_in[cellId](i, 0);

        // 特定变量置零（如密度、动量等）
        if (i % 5 == 3) U_n[cellId](i, 0) = 0.0;
    }
}


#define Expand_For_Flux(Order) {\
    if(FluxType=="LF") RunCompressibleEuler<Order,LaxFriedrichsFlux<IdealGasPhysics>,(Order>1?2:1),false>(meshN, fsm, logger); \
    if(FluxType=="HLL") RunCompressibleEuler<Order,HLLFlux<IdealGasPhysics>,(Order>1?2:1),false>(meshN, fsm, logger); \
    if(FluxType=="HLLC") RunCompressibleEuler<Order,HLLCFlux<IdealGasPhysics>,(Order>1?2:1),false>(meshN, fsm, logger);\
    if(FluxType=="RSIR") RunCompressibleEuler<Order,RSIRFlux<IdealGasPhysics>,(Order>1?2:1),false>(meshN, fsm, logger); \
    if(FluxType=="RHLL") RunCompressibleEuler<Order,StabilizedFlux<HLLFlux<IdealGasPhysics>>,(Order>1?2:1),false>(meshN, fsm, logger); \
    if(FluxType=="RHLLC") RunCompressibleEuler<Order,StabilizedFlux<HLLCFlux<IdealGasPhysics>>,(Order>1?2:1),false>(meshN, fsm, logger);\
    if(FluxType=="RRSIR") RunCompressibleEuler<Order,StabilizedFlux<RSIRFlux<IdealGasPhysics>>,(Order>1?2:1),false>(meshN, fsm, logger); \
    if(FluxType=="ELF") RunCompressibleEuler<Order,LaxFriedrichsFlux<IdealGasPhysics>,(Order>1?2:1),true>(meshN, fsm, logger); \
    if(FluxType=="EHLL") RunCompressibleEuler<Order,HLLFlux<IdealGasPhysics>,(Order>1?2:1),true>(meshN, fsm, logger); \
    if(FluxType=="EHLLC") RunCompressibleEuler<Order,HLLCFlux<IdealGasPhysics>,(Order>1?2:1),true>(meshN, fsm, logger);\
    if(FluxType=="ERSIR") RunCompressibleEuler<Order,RSIRFlux<IdealGasPhysics>,(Order>1?2:1),true>(meshN, fsm, logger); \
    if(FluxType=="ERHLL") RunCompressibleEuler<Order,StabilizedFlux<HLLFlux<IdealGasPhysics>>,(Order>1?2:1),true>(meshN, fsm, logger); \
    if(FluxType=="ERHLLC") RunCompressibleEuler<Order,StabilizedFlux<HLLCFlux<IdealGasPhysics>>,(Order>1?2:1),true>(meshN, fsm, logger);\
    if(FluxType=="ERRSIR") RunCompressibleEuler<Order,StabilizedFlux<RSIRFlux<IdealGasPhysics>>,(Order>1?2:1),true>(meshN, fsm, logger); \
}

int main(int argc, char** argv){
    int cpus = get_phy_cpu();
    int order = std::stoi(argv[1]);
    int meshN = std::stoi(argv[2]);
    // if(argc > 3){
    //     cpus = std::stoi(argv[3]);
    // }
    std::string FluxType = "LF";
    if(argc > 3){
        std::cout << FluxType <<std::endl;
        std::cout << argv[3] <<std::endl;
        FluxType = argv[3];
    }
    omp_set_num_threads(cpus);

    // 文件管理系统
    FilesystemManager fsm("./Order_" + std::to_string(order) + "_Mesh_" + std::to_string(meshN)+"_"+FluxType);

    // 创建目录结构
    fsm.prepare_output_directory();

    // 日志系统
    LoggerSystem logger(fsm);
    logger.log_boxed_title("Discontinuous Galerkin Simulation");
    logger.set_indent(0);
    logger.print_header("Discontinuous Galerkin Simulation");
    logger.print_config(order, meshN, cpus);
    
                             
    // RunCompressibleEuler<1,LF75C,false>(meshN, fsm, logger, 0b01);
    
    if(order == 0) Expand_For_Flux(0);
    if(order == 1) Expand_For_Flux(1);
    if(order == 2) Expand_For_Flux(2);
    // if(order == 3) Expand_For_Flux(3);
    // if(order == 1) RunCompressibleEuler<1,HLLCFlux<IdealGasPhysics>,1>(meshN, fsm, logger);
    // if(order == 2) RunCompressibleEuler<2>(meshN, fsm, logger);
}





template<uInt Order, typename NumFlux, uInt Level, uInt WithEntropy>
void RunCompressibleEuler(uInt N, FilesystemManager& fsm, LoggerSystem& logger){

    logger.log_section_title("Setup Stage");

    logger.start_stage("Split Hex Mesh to Tet");

    const auto& cmesh = create_mesh(N);
    check_mesh(cmesh);
    DeviceMesh gpu_mesh;
    gpu_mesh.initialize_from(cmesh);  // 这部分完全 CPU 逻辑
    gpu_mesh.upload_to_gpu();   

    logger.end_stage();

    logger.print_mesh_info(gpu_mesh);



    
    /* ======================================================= *\
    **   算子 和 限制器 的实例化
    \* ======================================================= */
    using Basis = DGBasisEvaluator<Order>;
    using QuadC = typename AutoQuadSelector<Basis::OrderBasis, GaussLegendreTet::Auto>::type;
    using QuadF = typename AutoQuadSelector<Basis::OrderBasis, GaussLegendreTri::Auto>::type;



    IdealGasPhysics physics(1.4); // gamma = 1.4
    constexpr uInt Neqn = decltype(physics)::NEQN;
    constexpr uInt DoFs = decltype(physics)::NEQN*Basis::NumBasis;


    IBCondition<decltype(physics)> condition(physics);
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
        const auto& rhoU_coef = Basis::func2coef_with_bounds([&](vector3f Xi)->DenseMatrix<Neqn,1>{
            const vector3f& xyz = cell.transform_to_physical(Xi);
            return condition.compute(xyz, 0.0);
        });
        /* 写入到向量 U_n 的单元 cell 那一段*/
        for(uInt k=0;k<Basis::NumBasis;k++){
            MatrixView<DoFs,1,Neqn,1>(U_n[cellId],Neqn*k,0) = rhoU_coef[k];
        }
    }

    LongVectorDevice<DoFs> gpu_U_n = U_n.to_device();
    Scalar s0 = compute_initial_s0<IdealGasPhysics, Order>(gpu_U_n, physics, gpu_mesh.num_cells());

    ExplicitConvectionGPU<decltype(physics), NumFlux, decltype(condition), Basis::OrderBasis, QuadC, QuadF> convection(physics,condition);
    PositivityPreservingLimiterGPU<decltype(physics), Basis::OrderBasis, QuadC, QuadF, Level, WithEntropy> positive_limiter(gpu_mesh, physics, s0);

    positive_limiter.apply(gpu_U_n);




        
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
            for(uInt k=0; k<Neqn; ++k) {
                r_mass[cid](Neqn*i + k, 0) = 1.0/val;
            }
        }
    }
    
    LongVectorDevice<DoFs> gpu_r_mass = r_mass.to_device();
    LongVectorDevice<DoFs> U_1_(U_n.size());
    
    U_n = gpu_U_n.download();
    save_DG_solution_to_hdf5<QuadC,Basis>(cmesh, U_n, fsm.get_solution_file_h5(0, N));

    logger.log_section_title("Time Marching");
    Scalar total_time = 0.0;
    Scalar final_time = get_final_time();
    std::vector<Scalar> save_time = get_save_time();
    
    Scalar CFL = get_CFL(0);
    Scalar dt = compute_CFL_time_step<Order, QuadC, Basis>(cmesh, gpu_mesh, gpu_U_n, CFL, physics.get_gamma());

    for(const auto& p : save_time) std::cout<<std::setw(6)<<p<<"  "; std::cout<<std::endl;

    uInt save_index = 0;
    uInt iter = 0;
    // TimeIntegrator<DoFs,Order,OnlyNeigbAvg> time_integrator(gpu_mesh,gpu_U_n,gpu_r_mass,positivelimiter,wenolimiter,pweight_wenolimiter);
    // time_integrator.set_scheme(get_time_intergrator_scheme());
    logger.log_explicit_step(uInt(-1), 0.0, 0.0, 0.0);
    while (total_time < final_time) {
        CFL = get_CFL(iter);
        //  if (iter < 3000 || iter % 1000 == 0) 
        dt = compute_CFL_time_step<Order, QuadC, Basis>(cmesh, gpu_mesh, gpu_U_n, CFL, physics.get_gamma());
        Scalar curr_dt = dt;
        // 截断到下一个 save_time 保证不会错过保存时间点
        if (save_index < save_time.size() && total_time + dt > save_time[save_index])
            curr_dt = save_time[save_index] - total_time;
        if (total_time + curr_dt > final_time)
            curr_dt = final_time - total_time;
        std::cout << dt << " " << curr_dt << std::endl;



        uInt size = gpu_U_n.size();
        dim3 block(256);
        dim3 grid((size + block.x - 1) / block.x);

        convection.eval(gpu_mesh, gpu_U_n, U_1_, total_time + 0.5 * curr_dt);
        update_solution<<<grid, block>>>(gpu_U_n.d_blocks, U_1_.d_blocks, gpu_r_mass.d_blocks, curr_dt, size);
        positive_limiter.apply(gpu_U_n);





        total_time += curr_dt;
        iter++;


        if(logger.log_explicit_step(iter, total_time, curr_dt, save_time[save_index])){
            const std::string& filename = fsm.get_solution_file_h5(save_index+1, N);
            logger.log_save_solution(iter, total_time, filename);
            save_DG_solution_to_hdf5<QuadC,Basis>(cmesh, gpu_U_n.download(), filename,total_time,iter);
            save_index++;
        }

    }
}
