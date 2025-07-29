#include "base/type.h"
#include "base/filesystem_manager.h"
#include "base/logger_system.h"
#include "dg/dg_flux/euler_physical_flux.h"
// #include "base/io.h"
// #include "mesh/mesh.h"
// #include "matrix/matrix.h"
// #include "dg/dg_basis/dg_basis.h"
// #include "dg/dg_schemes/explicit_convection.h"
// #include "dg/dg_schemes/explicit_convection_gpu.h"
// #include "dg/dg_schemes/implicit_diffusion.h"
// #include "DG/DG_Schemes/PositiveLimiter.h"
// #include "DG/DG_Schemes/PositiveLimiterGPU.h"
// #include "DG/DG_Schemes/PWENOLimiter.h"
#include "solver/eigen_sparse_solver.h"
// #include "mesh/device_mesh.h"

#include "dg/time_integrator.h"

// #include "problem.h"
// #include "tools.h"
// #include "save_to_hdf5.h"


#define Expand_For_Flux(Order) {\
    if(FluxType=="LF") Run<Order,LF75C>(meshN, fsm, logger); \
    if(FluxType=="Roe") Run<Order,Roe75C>(meshN, fsm, logger); \
    if(FluxType=="HLL") Run<Order,HLL75C>(meshN, fsm, logger); \
    if(FluxType=="HLLC") Run<Order,HLLC75C>(meshN, fsm, logger);\
    if(FluxType=="HLLEM") Run<Order,HLLEM75C>(meshN, fsm, logger);\
}

template<uInt Order,typename FluxType>
void Run(uInt N, FilesystemManager& fsm, LoggerSystem& logger);


HostDevice Scalar get_gamma() {return 1.4;}

TimeIntegrationScheme get_time_intergrator_scheme() {
    return TimeIntegrationScheme::EULER;
}

Scalar get_CFL(){
    return 0.5;
}

Scalar get_final_time() {
    return 4.0;
}

std::vector<Scalar> get_save_time(){
    std::vector<Scalar> save_time;
    for(uInt i=0; i<8; ++i) {
        save_time.push_back((i+1) * 0.5 );
    }
    return save_time;
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
    Eigen::setNbThreads(cpus);

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
    
                             
    if(order == 1) Expand_For_Flux(1);
    if(order == 2) Expand_For_Flux(2);
    if(order == 3) Expand_For_Flux(3);
    // if(order == 4) Run<4>(meshN);
    // if(order == 5) Run<5>(meshN);
}

