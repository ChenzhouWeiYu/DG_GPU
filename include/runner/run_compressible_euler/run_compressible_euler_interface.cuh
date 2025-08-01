#pragma once
#include "base/type.h"
#include "mesh/computing_mesh.h"
// #include "mesh/cgal_mesh.h"
#include "dg/time_integrator.cuh"
#include "base/filesystem_manager.h"
#include "base/logger_system.h"


template<uInt Order,typename FluxType, bool OnlyNeigbAvg = false>
void RunCompressibleEuler(uInt N, FilesystemManager& fsm, LoggerSystem& logger, uInt limiter_flag = uInt(-1));

ComputingMesh create_mesh(uInt N);
HostDevice Scalar get_gamma();
TimeIntegrationScheme get_time_intergrator_scheme();

Scalar get_final_time();
std::vector<Scalar> get_save_time();

Scalar get_CFL(uInt iter = 0);