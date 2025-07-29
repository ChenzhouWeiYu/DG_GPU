#include "base/type.h"
#include "base/filesystem_manager.h"
#include "base/logger_system.h"
// #include "dg/time_integrator.h"
#include "rte_time_integrator.h"

#include "base/io.h"
#include "mesh/mesh.h"
#include "matrix/matrix.h"
#include "dg/dg_basis/dg_basis.h"
#include "dg/dg_schemes/explicit_rte_gpu_impl/explicit_rte_gpu.h"
#include "mesh/device_mesh.h"

#include "mesh.h"

TimeIntegrationScheme get_time_intergrator_scheme() {
    return TimeIntegrationScheme::SSP_RK3;
}

Scalar get_CFL(){
    return 0.5;
}

Scalar get_final_time() {
    return 0.2;
}

std::vector<Scalar> get_save_time(){
    std::vector<Scalar> save_time;
    for(uInt i=0; i<20; ++i) {
        save_time.push_back((i+1) * 0.01 );
    }
    return save_time;
}





template<uInt X3Order, uInt DoFs>
Scalar compute_CFL_time_step(
    const ComputingMesh& cpu_mesh,
    const DeviceMesh& gpu_mesh,
    const LongVectorDevice<DoFs>& coef_device,
    Scalar CFL)
{
    const uInt num_cells = gpu_mesh.num_cells();

    // h / lambda 最小值
    Scalar min_dt = std::numeric_limits<Scalar>::max();
    for (uInt i = 0; i < num_cells; ++i) {
        Scalar h_i = cpu_mesh.m_cells[i].m_h;
        Scalar lam = 1.0;
        if (lam > 1e-12)
            min_dt = std::min(min_dt, h_i / lam);
    }

    Scalar dt = CFL * min_dt / std::pow(2 * X3Order + 1, 1); // k=1
    return dt;
}






template<uInt X3Order,typename S2Mesh, bool OnlyNeigbAvg>
void RunRadiationTransfer(uInt N, FilesystemManager& fsm, LoggerSystem& logger, uInt limiter_flag){

    logger.log_section_title("Setup Stage");

    logger.start_stage("Split Hex Mesh to Tet");

    const auto& cmesh = create_mesh(N);
    check_mesh(cmesh);
    DeviceMesh gpu_mesh;
    gpu_mesh.initialize_from(cmesh);  // 这部分完全 CPU 逻辑
    gpu_mesh.upload_to_gpu();   

    logger.end_stage();

    logger.print_mesh_info(gpu_mesh);



    using X3Basis = DGBasisEvaluator<X3Order>;
    using X3QuadC = typename AutoQuadSelector<X3Basis::OrderBasis, GaussLegendreTet::Auto>::type;

    constexpr uInt S2Order = X3Order;
    using S2Basis = DGBasisEvaluator2D<S2Order>;
    using S2QuadC = typename AutoQuadSelector<S2Basis::OrderBasis, GaussLegendreTri::Auto>::type;
    
    
    constexpr uInt X3DoFs = X3Basis::NumBasis;
    constexpr uInt S2DoFs = S2Basis::NumBasis;
    constexpr uInt S2Cells = S2Mesh::num_cells;
    constexpr auto s2_cells = S2Mesh::s2_cells();
    constexpr uInt DoFs = X3DoFs * S2DoFs * S2Cells;
    
    /* ======================================================= *\
    **   算子 和 限制器 的实例化
    \* ======================================================= */
    ExplicitRTEGPU<X3Order, S2Order, X3QuadC, S2QuadC, S2Mesh> scheme;
    
    
    
    logger.start_stage("Set Initial Condition");
    /* ======================================================= *\
    **   设置初值
    \* ======================================================= */
    LongVector<DoFs> U_n(cmesh.m_cells.size());
    // #pragma omp parallel for schedule(dynamic)
    // for(uInt cid=0;cid<cmesh.m_cells.size();cid++){
    //     /* 获取单元 cell 的信息 */
    //     const auto& cell = cmesh.m_cells[cid];
    //     /* 单元 cell 上，计算初值的多项式插值系数 */
    //     const auto& X3_coef = X3Basis::func2coef([&](vector3f Xi)->Scalar{
    //         const vector3f& xyz = cell.transform_to_physical(Xi);
    //         Scalar x = xyz[0], y = xyz[1], z = xyz[2];
    //         Scalar r2 = x * x + y * y;
    //         Scalar r4 = r2 * r2;
    //         Scalar a = 16;
    //         Scalar Pi4 = M_PI * M_PI * M_PI * M_PI;
    //         return M_1_PI * 0.25 * a / (1 + a*a * Pi4 * 0.25 * r4);
    //     });
    //     /* 获取单元 cell 的信息 */
    //     const auto& S2_coef = S2Basis::func2coef([&](vector2f Xi)->Scalar{

    //         auto phys = S2Mesh::spherical_to_cartesian(Xi);  // Xi = (phi, mu)
    //         const vector3f Omega0 = {std::sqrt(1.0/3.0), std::sqrt(1.0/3.0), std::sqrt(1.0/3.0)};
    //         Scalar dot = phys[0]*Omega0[0] + phys[1]*Omega0[1] + phys[2]*Omega0[2];
    //         return 1.0;
    //     });

    //     for(uInt i = 0; i < X3DoFs; ++i) {
    //         for(uInt j = 0; j < S2DoFs; ++j) {
    //             U_n[cid](i*S2DoFs + j, 0) = X3_coef[i] * S2_coef[j];
    //         }
    //     }
    // }
    // === 角度方向的目标方向 Omega0（单位向量） === //
    const vector3f Omega0 = {std::sqrt(1.0 / 3.0), std::sqrt(1.0 / 3.0), std::sqrt(1.0 / 3.0)};

    // === 遍历所有空间单元 cid === //
    #pragma omp parallel for schedule(dynamic)
    for (uInt cid = 0; cid < cmesh.m_cells.size(); ++cid) {
        const auto& cell = cmesh.m_cells[cid];

        // ==== 定义空间函数 f(x) ==== //
        auto f_xyz = [&](vector3f Xi) -> Scalar {
            vector3f x = cell.transform_to_physical(Xi);
            vector3f xc = {0.5, 0.5, 0.0};  // 空间中心
            Scalar dx = x[0] - xc[0];
            Scalar dy = x[1] - xc[1];
            Scalar dz = x[2] - xc[2];
            Scalar r2 = dx*dx + dy*dy;
            return std::exp(-100.0 * r2);
        };

        // === 得到该单元上的空间系数展开 === //
        const auto X3_coef = X3Basis::func2coef(f_xyz);

        // === 遍历所有角度单元 aid === //
        for (uInt aid = 0; aid < S2Cells; ++aid) {
            const auto& s2_cell = s2_cells[aid];

            // ==== 定义角度函数 g(Ω) ==== //
            auto g_ang = [&](vector2f Xi) -> Scalar {
                vector2f phi_mu = s2_cell.map_to_physical(Xi); // Xi in reference triangle
                vector3f Omega = S2Mesh::spherical_to_cartesian(phi_mu);
                Scalar dot = Omega[0]*Omega0[0] + Omega[1]*Omega0[1] + Omega[2]*Omega0[2];
                return std::exp(50.0 * (dot - 1.0));  // Peak at Omega0
            };

            // === 得到该角度单元上的系数展开 === //
            const auto S2_coef = S2Basis::func2coef(g_ang);

            // === 写入张量乘积系数到 U_n[cid] === //
            for (uInt i = 0; i < X3DoFs; ++i) {
                for (uInt j = 0; j < S2DoFs; ++j) {
                    uInt idx = aid * X3DoFs * S2DoFs + i * S2DoFs + j;
                    U_n[cid](idx, 0) = X3_coef[i] * S2_coef[j];
                }
            }
        }
    }


    LongVectorDevice<DoFs> gpu_U_n = U_n.to_device();


    logger.end_stage();


    /* ======================================================= *\
    **   计算 (\phi_i, \phi_i) 作为质量矩阵
    **   正交基，只需要计算、保存对角元。  r_mass 表示是 倒数
    \* ======================================================= */
    LongVector<DoFs> r_mass(U_n.size());

    // 获取角度空间 S² 上的质量项对角元（由于是统一的静态 S2Basis，可预计算一次）
    std::array<Scalar,S2DoFs> s2_mass_diag;
    s2_mass_diag.fill(0.0);
    {
        constexpr auto S2points = S2QuadC::get_points();
        constexpr auto S2weights = S2QuadC::get_weights();
        for (uInt g = 0; g < S2QuadC::num_points; ++g) {
            const auto& uv = S2points[g];
            Scalar w = S2weights[g];
            auto phi = S2Basis::eval_all(uv[0], uv[1]);
            for (uInt j = 0; j < S2DoFs; ++j) {
                s2_mass_diag[j] += phi[j] * phi[j] * w;
            }
        }
    }
    std::array<Scalar,X3DoFs> x3_mass_diag;
    x3_mass_diag.fill(0.0);
    {
        constexpr auto X3points = X3QuadC::get_points();
        constexpr auto X3weights = X3QuadC::get_weights();
        for (uInt g = 0; g < X3QuadC::num_points; ++g) {
            const auto& p = X3points[g];
            Scalar w = X3weights[g];
            auto phi = X3Basis::eval_all(p[0], p[1], p[2]);
            for (uInt i = 0; i < X3DoFs; ++i) {
                x3_mass_diag[i] += phi[i] * phi[i] * w;
            }
        }
    }

    print(s2_mass_diag);
    print(x3_mass_diag);
    
    for (uInt cid = 0; cid < cmesh.m_cells.size(); ++cid) {
        const auto& cell = cmesh.m_cells[cid];
        Scalar detJacX3 = cell.compute_jacobian_det(); // = cell.m_volume * 6
        for (uInt aid = 0; aid < S2Cells; ++aid) {
            const auto& s2_cell = s2_cells[aid];
            Scalar detJacS2 = s2_cell.area * 2;
            for (uInt i = 0; i < X3DoFs; ++i) {
                for (uInt j = 0; j < S2DoFs; ++j) {
                    Scalar mass_diag = x3_mass_diag[i] * s2_mass_diag[j] * detJacX3 * detJacS2;
                    // print(vector4f{x3_mass_diag[i], s2_mass_diag[j], detJacX3, detJacS2});
                    r_mass[cid](aid * X3DoFs * S2DoFs + i * S2DoFs + j, 0) = 1.0 / mass_diag;
                }
            }
        }
    }

    // print(r_mass);
    LongVectorDevice<DoFs> gpu_r_mass = r_mass.to_device();

    
    U_n = gpu_U_n.download();
    save_RTE_solution_to_hdf5<X3QuadC,S2QuadC,X3Basis,S2Basis,S2Mesh>(cmesh, U_n, fsm.get_solution_file_h5(0, N));


    logger.log_section_title("Time Marching");
    Scalar total_time = 0.0;
    Scalar final_time = 0.1;
    std::vector<Scalar> save_time = {0.001,0.01,0.1};
    Scalar CFL = get_CFL();

    for(const auto& p : save_time) std::cout<<std::setw(6)<<p<<"  "; std::cout<<std::endl;

    uInt save_index = 0;
    uInt iter = 0;
    RTE_TimeIntegrator<X3Order,S2Order,X3QuadC,S2QuadC,S2Mesh> time_integrator(gpu_mesh,gpu_U_n,gpu_r_mass);
    time_integrator.set_scheme(TimeIntegrationScheme::EULER);
    logger.log_explicit_step(uInt(-1), 0.0, 0.0, 0.0);
    while (total_time < final_time) {
        Scalar dt = compute_CFL_time_step<X3Order, DoFs>(cmesh, gpu_mesh, gpu_U_n, CFL);

        // 截断到下一个 save_time 保证不会错过保存时间点
        if (save_index < save_time.size() && total_time + dt > save_time[save_index])
            dt = save_time[save_index] - total_time;

        if (total_time + dt > final_time)
            dt = final_time - total_time;
        
        time_integrator.advance(scheme,total_time,dt);

        total_time += dt;
        iter++;
        if(logger.log_explicit_step(iter, total_time, dt, save_time[save_index])){
            const std::string& filename = fsm.get_solution_file_h5(save_index+1, N);
            logger.log_save_solution(iter, total_time, filename);
            save_RTE_solution_to_hdf5<X3QuadC,S2QuadC,X3Basis,S2Basis,S2Mesh>(cmesh, gpu_U_n.download(), filename,total_time,iter);
            save_index++;
        }

    }
}









int main(int argc, char** argv){
    int cpus = get_phy_cpu();
    int order = std::stoi(argv[1]);
    int meshN = std::stoi(argv[2]);
    
    omp_set_num_threads(cpus);

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
    
                             
    if(order == 1) RunRadiationTransfer<1,S2MeshIcosahedral,false>(meshN,fsm,logger,0b00);
    if(order == 2) RunRadiationTransfer<2,S2MeshIcosahedral,false>(meshN,fsm,logger,0b00);
    // if(order == 3) RunRadiationTransfer<3,S2MeshIcosahedral,false>(meshN,fsm,logger,0b00);
    // if(order == 4) Run<4>(meshN);
    // if(order == 5) Run<5>(meshN);
}

