#include "base/type.h"
#include "base/filesystem_manager.h"
#include "base/logger_system.h"
#include "dg/dg_flux/euler_physical_flux.h"
#include "runner/run_compressible_euler/run_compressible_euler_interface.cuh"



template<uInt Order,typename FluxType>
void RunCompressibleEuler(uInt N, FilesystemManager& fsm, LoggerSystem& logger);

#define Expand_For_Flux(Order) {\
    if(FluxType=="LF") RunCompressibleEuler<Order,LF75C,false>(meshN, fsm, logger, 0b01); \
    if(FluxType=="Roe") RunCompressibleEuler<Order,Roe75C,false>(meshN, fsm, logger, 0b01); \
    if(FluxType=="HLL") RunCompressibleEuler<Order,HLL75C,false>(meshN, fsm, logger, 0b01); \
    if(FluxType=="HLLC") RunCompressibleEuler<Order,HLLC75C,false>(meshN, fsm, logger, 0b01);\
    if(FluxType=="RHLLC") RunCompressibleEuler<Order,RHLLC75C,false>(meshN, fsm, logger, 0b01);\
    if(FluxType=="HLLEM") RunCompressibleEuler<Order,HLLEM75C,false>(meshN, fsm, logger, 0b01);\
    if(FluxType=="LF_WENO") RunCompressibleEuler<Order,LF75C,false>(meshN, fsm, logger, 0b11); \
    if(FluxType=="Roe_WENO") RunCompressibleEuler<Order,Roe75C,false>(meshN, fsm, logger, 0b11); \
    if(FluxType=="HLL_WENO") RunCompressibleEuler<Order,HLL75C,false>(meshN, fsm, logger, 0b11); \
    if(FluxType=="HLLC_WENO") RunCompressibleEuler<Order,HLLC75C,false>(meshN, fsm, logger, 0b11);\
    if(FluxType=="RHLLC_WENO") RunCompressibleEuler<Order,RHLLC75C,false>(meshN, fsm, logger, 0b11);\
    if(FluxType=="HLLEM_WENO") RunCompressibleEuler<Order,HLLEM75C,false>(meshN, fsm, logger, 0b11);\
}


TimeIntegrationScheme get_time_intergrator_scheme() {
    return TimeIntegrationScheme::EULER;
}

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
    return 4.0;
}

std::vector<Scalar> get_save_time(){
    std::vector<Scalar> save_time;
    for(uInt i=0; i<40; ++i) {
        save_time.push_back((i+1) * 0.1 );
    }
    return save_time;
}


/*
Scalar get_final_time() {
    return 0.001;
}

std::vector<Scalar> get_save_time(){
    std::vector<Scalar> save_time;
    for(uInt i=0; i<1; ++i) {
        save_time.push_back(get_final_time() * 1.0 );
    }
    return save_time;
}
*/



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
    
                             
    if(order == 1) Expand_For_Flux(1);
    if(order == 2) Expand_For_Flux(2);
    if(order == 3) Expand_For_Flux(3);
    // if(order == 4) Run<4>(meshN);
    // if(order == 5) Run<5>(meshN);
}

