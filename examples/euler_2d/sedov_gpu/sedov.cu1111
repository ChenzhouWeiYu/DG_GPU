#include "base/type.h"
#include "base/filesystem_manager.h"
#include "base/LoggerSystem.hpp"
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

#include "problem.h"
#include "tools.h"

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


template<uInt DoFs>
void Eular_Step(auto& convection,
                const DeviceMesh& gpu_mesh, Scalar dt,
                LongVectorDevice<DoFs>& gpu_U_n,
                const LongVectorDevice<DoFs>& gpu_r_mass) 
{
    // LongVector<DoFs> U_1(gpu_U_n.size());
    // LongVectorDevice<DoFs> gpu_U_1 = U_1.to_device();
    LongVectorDevice<DoFs> gpu_U_1(gpu_U_n.size());

    convection.eval(gpu_mesh, gpu_U_n, gpu_U_1);

    const int num_blocks = (gpu_U_n.size() + 31) / 32;
    update_solution<DoFs><<<num_blocks, 32>>>(gpu_U_n.d_blocks, gpu_U_1.d_blocks, gpu_r_mass.d_blocks, dt, gpu_U_n.size());
    cudaDeviceSynchronize();
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
    gpu_mesh.initialize_from(cmesh);  // 这部分完全 CPU 逻辑
    gpu_mesh.upload_to_gpu();            // 上传
    


    // debug(gpu_mesh.num_cells());
    // debug(gpu_mesh.num_faces());

    // auto host_cells = gpu_mesh.host_cells();
    // auto host_faces = gpu_mesh.host_faces();

    // debug(host_cells[0].nodes);
    // debug(host_cells[0].centroid);
    // debug(host_cells[0].volume);
    // debug(host_cells[0].JacMat);
    // debug(host_cells[0].invJac);

    // debug(host_faces[0].nodes);
    // debug(host_faces[0].neighbor_cells);
    // debug(host_faces[0].normal);
    // debug(host_faces[0].area);

    // const auto& invJac = gpu_mesh.device_cells()[0].invJac;
    // DenseMatrix<3,3> A = {1,3,7,4,5,8,9,6,2};
    // DenseMatrix<3,3> AtA = DenseMatrix<3,3>::Identity();

    // DenseMatrix<3,3>* d_A;
    // DenseMatrix<3,3>* d_AtA;
    // cudaMalloc(&d_A, sizeof(DenseMatrix<3,3>));
    // cudaMalloc(&d_AtA, sizeof(DenseMatrix<3,3>));

    // // Host -> Device 拷贝输入矩阵
    // cudaMemcpy(d_A, &A, sizeof(DenseMatrix<3,3>), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_AtA, &AtA, sizeof(DenseMatrix<3,3>), cudaMemcpyHostToDevice);
    
    // debug(copy_device_to_host(d_A));
    // debug(copy_device_to_host(d_AtA));
    // mat_kernel<<<1,1>>>(d_A,d_AtA);
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess)
    //     std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;

    // cudaDeviceSynchronize();
    // debug(copy_device_to_host(d_A));
    // debug(copy_device_to_host(d_AtA));

    logger.end_stage();
    // return ; // 只测试网格划分



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
    CPUpositivelimiter.apply(U_n, U_n); 

    LongVectorDevice<DoFs> gpu_U_n = U_n.to_device();


    // // print(U_n[0].transpose());
    // // print(U_n[1].transpose());
    // // print(U_n[2].transpose());
    // // print(U_n[3].transpose());
    // print(U_n[30].transpose());
    // // print(U_n[31].transpose());
    // print(cmesh.m_cells[30].m_neighbors);
    // print(gpu_mesh.host_cells()[30].neighbor_cells);
    // for(uInt cid : cmesh.m_cells[30].m_neighbors){
    //     if (cid != uInt(-1)) print(U_n[cid].transpose());
    // }
    // print(U_n.dot(U_n));
    // positivelimiter.constructMinMax(gpu_U_n);


    // print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
    
    // // const auto& h_per_cell_min = positivelimiter.d_per_cell_min.download();
    // // const auto& h_per_cell_max = positivelimiter.d_per_cell_max.download();
    // // print(h_per_cell_min);
    // // print(h_per_cell_max);





    // U_n = gpu_U_n.download();
    // print(U_n.dot(U_n));

    // // positivelimiter.apply(gpu_U_n);
    // // U_n = gpu_U_n.download();
    // // print(U_n.dot(U_n));

    // const auto& U_h = reconstruct_solution<QuadC,Basis>(cmesh,U_n);
    // print(U_h);
    // debug(gpu_U_n.size());

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
    save_Uh_Us<QuadC,Basis>(cmesh,U_n,0.0,fsm.get_solution_file(0, N));
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
    Scalar total_time = 0.0, final_time = 1.0;
    std::vector<Scalar> save_time;
    for(uInt i=0; i<100; ++i) {
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
    uInt iter = 0, output_interval = 1000;

    while (total_time < final_time) {
        // 计算时间步长
        // 计算 CFL 数，确定最小时间步长
        auto [c_min,c_max] = std::minmax_element(cmesh.m_cells.begin(),cmesh.m_cells.end(),
            [](const CompTetrahedron& x, const CompTetrahedron& y){return x.m_h<y.m_h;});
        Scalar h_min = (*c_min).m_h;
        Scalar h_max = (*c_max).m_h;
        // debug(vector2f{h_min,h_max});
        // 波速
        const auto& U_h = reconstruct_solution<QuadC,Basis>(cmesh,U_n);
        std::vector<Scalar> wave_speeds(U_h.size());
        #pragma omp parallel for schedule(dynamic)
        for(uInt i=0; i<U_h.size(); ++i) {
            constexpr auto Qpoint = QuadC::get_points();
            constexpr auto Qweights = QuadC::get_weights();
            wave_speeds[i] = 0.0;
            for(uInt xgid=0; xgid<QuadC::num_points; ++xgid) {
                const auto& U_h_xg = U_h[i].template SubMat<5,1>(5*xgid, 0);
                // 计算波速
                const auto& wave_speed = AirFluxC::waveSpeedMagnitude(U_h_xg);
                wave_speeds[i] += wave_speed * Qweights[xgid] * 6.0; // 6.0 是体积的缩放因子
                // printf("%f ", wave_speed);
            }
            // printf("\n");
            // printf("%f \n", wave_speeds[i]);
        }
        Scalar lambda = *std::max_element(wave_speeds.begin(), wave_speeds.end());
        printf("lambda = %.6e    ", lambda);
        // if(iter>4) {
        //     return; // 只测试 CFL 条件
        // }
        // CFL : \delta t < (Ch)/((2p+1)^k \lambda)   k=1~2
        Scalar max_dt = h_min / lambda * 1.0 / std::pow(2*Order+1, 1); 
        // Scalar dt = compute_dt(cmesh, U_n);
        Scalar dt = max_dt * 0.5;// (1.0+iter)/(1000.0+iter);

        // 截断到下一个 save_time 保证不会错过保存时间点
        if (save_index < save_time.size() && total_time + dt > save_time[save_index])
            dt = save_time[save_index] - total_time;

        if (total_time + dt > final_time)
            dt = final_time - total_time;

        // 开始模拟
        print("开始模拟，dt = " + std::to_string(dt) + " time = " + std::to_string(total_time));

        // positivelimiter.constructMinMax(U_n, U_n); 
        // printf("%lf,%lf",min(U_n),max(U_n));
        // positivelimiter.constructMinMax(gpu_U_n);
        // U_n = gpu_U_n.download();
        // print(U_n.dot(U_n));
        // const auto& h_per_cell_min = positivelimiter.d_per_cell_min.download();
        // const auto& h_per_cell_max = positivelimiter.d_per_cell_max.download();
        // print(h_per_cell_min);
        // print(h_per_cell_max);
        
        // U_n = gpu_U_n.download();
        CPUpositivelimiter.constructMinMax(U_n); 
        printf("更新前 %.6e    ", U_n.dot(U_n));
        // print(CPUpositivelimiter.per_cell_min);
        // print(CPUpositivelimiter.per_cell_max);


        Eular_Step(convection, gpu_mesh,    dt, gpu_U_n, gpu_r_mass);
        // cudaDeviceSynchronize();

        // positivelimiter.apply(gpu_U_n);
        // cudaDeviceSynchronize();
        
        U_n = gpu_U_n.download();
        // cudaDeviceSynchronize();
        // print(U_n);
        printf("更新后 %.6e    ", U_n.dot(U_n));
        // return;

        // print(U_n.dot(U_n));
        CPUpositivelimiter.apply(U_n); 
        // print(U_n);
        printf("限制后 %.6e    ", U_n.dot(U_n));

        gpu_U_n.fill_from_host(U_n);
        // cudaDeviceSynchronize();

        // print(U_n.dot(U_n));
        printf("\n");
        // if(iter>3)
        // return;
        // print(U_n);
        // U_n = gpu_U_n.download();
        // printf("%lf,%lf",min(U_n),max(U_n));
        // positivelimiter.apply_2(U_n, U_n); 
        // printf("%lf,%lf",min(U_n),max(U_n));
        // gpu_U_n = U_n.to_device();
        // break;

        // advance(U_n, cmesh, gpu_mass, total_time, dt);

        total_time += dt;
        iter++;

        // 按迭代步保存
        if (save_time.size() == 0 &&  output_interval > 0 && iter % output_interval == 0) {
            U_n = gpu_U_n.download();
            print("按步数保存成功: iter=" + std::to_string(iter) + " time=" + std::to_string(total_time));
            save_Uh_Us<QuadC,Basis>(cmesh,U_n,total_time,fsm.get_solution_file(0, N));
            // break;
        }

        // 按时间保存
        if (save_index < save_time.size() && total_time >= save_time[save_index] - 1e-12) {
            U_n = gpu_U_n.download();
            print("按时间保存成功: time=" + std::to_string(save_time[save_index]));
            save_Uh_Us<QuadC,Basis>(cmesh,U_n,total_time,fsm.get_solution_file(save_index+1, N));
            save_index++;
        }
    }



    return ; // 


    // /* ======================================================= *\
    // **   开始迭代
    // **   第一层迭代，关于数值解的保存的间隔
    // **   间隔 Dt 时间保存一次
    // \* ======================================================= */
    // // print(std::array<std::string,8>{"#       time", "rel.err  rho",
    // //                 "rel.err  u", "rel.err  v", "rel.err  w", 
    // //                 "rel.err  e", "rel.err coef", "cpu time"});
    // logger.log_section_title("Time Marching");
    // Scalar total_time = 0.0;
    // for(uInt save_step = 0; save_step < 10; save_step++){    
    //     Scalar Dt = 0.1;
    //     Scalar max_dt = Dt * std::pow((1.1*5/N),(Order+1));
    //     max_dt = (1.1/N) * std::pow(1.0/(2.0*Order+1.0),1);
    //     Scalar dt = max_dt;
    //     uInt kk = 0;
        

    //     /* ======================================================= *\
    //     **   第二层迭代，关于 保存间隔 Dt 内的时间推进
    //     **   需要判断是否超过 Dt，超过了就截断
    //     **   sub_t 是当前的 Dt 间隔内的时间推进
    //     \* ======================================================= */
    //     Scalar sub_t = 0;
    //     do{ 
            
    //         /* 时间步长 dt 截断到 Dt 长度 */
    //         dt = std::min(max_dt, Dt-sub_t); 
    //         if(total_time < max_dt * 1.00e-6) dt = max_dt * 1.00e-6; else
    //         if(total_time < max_dt * 1.00e-5) dt = max_dt * 1.00e-5; else
    //         if(total_time < max_dt * 1.00e-4) dt = max_dt * 1.00e-4; else
    //         if(total_time < max_dt * 3.16e-3) dt = max_dt * 3.16e-3; else
    //         if(total_time < max_dt * 1.00e-2) dt = max_dt * 1.00e-2; else
    //         if(total_time < max_dt * 3.00e-2) dt = max_dt * 2.00e-2; else
    //         if(total_time < max_dt * 6.00e-2) dt = max_dt * 3.00e-2; else
    //         if(total_time < max_dt * 1.00e-1) dt = max_dt * 4.00e-2; else
    //         if(total_time < max_dt * 1.50e-1) dt = max_dt * 5.00e-2; else
    //         if(total_time < max_dt * 2.10e-1) dt = max_dt * 6.00e-2; else
    //         if(total_time < max_dt * 2.80e-1) dt = max_dt * 7.00e-2; else
    //         if(total_time < max_dt * 3.60e-1) dt = max_dt * 8.00e-2; else
    //         if(total_time < max_dt * 4.50e-1) dt = max_dt * 9.00e-2; else
    //         if(total_time < max_dt * 5.50e-1) dt = max_dt * 1.00e-1; else
    //         if(total_time < max_dt * 7.00e-1) dt = max_dt * 1.50e-1; else
    //         if(total_time < max_dt * 1.00e-0) dt = max_dt * 3.00e-1; else
    //         if(total_time < max_dt * 1.50e-0) dt = max_dt * 5.00e-1; else
    //         if(total_time < max_dt * 2.10e-0) dt = max_dt * 7.00e-1;
    //         // logging("Delta t = " + std::to_string(dt));
    //         sub_t += dt;
    //         total_time += dt; // 实际物理时间

    //         logger.log_time_step(save_step, total_time, dt);

            
    //         /* 记录下旧时刻的解 */
    //         LongVector<DoFs> U_k = U_n;

    //         /* 记录初始迭代步的 残差、初始的增量，用于非线性迭代停机条件 */
    //         Scalar init_delta = 0.0;
    //         Scalar prev_delta = 0.0;
    //         Scalar init_res_norm = 0.0;
    //         Scalar prev_res_norm = 0.0;

    //         LongVector<5*Basis::NumBasis> res_old(U_n.size());
    //         /* ======================================================= *\
    //         **   lambda 表达式，用于第三层迭代的 离散算子
    //         **   U_k : 上一步迭代的解，用于离散时的线性化，边界条件的 ghost 
    //         **   U_n : 当前时刻的解，用于时间离散的源项
    //         \* ======================================================= */
    //         const auto& get_matrix = [&](
    //             BlockSparseMatrix<5*Basis::NumBasis,5*Basis::NumBasis>& sparse_mat,
    //             LongVector<5*Basis::NumBasis>& rhs, 
    //             const LongVector<5*Basis::NumBasis>& U_k)
    //         {
    //             /* 用 保存的 dx 代入、离散 */ 
    //             convection.assemble(cmesh, U_k, total_time, sparse_mat, rhs);
    //             // diffusion.assemble(cmesh, U_k, total_time, sparse_mat, rhs);
                
    //             /* 补上质量矩阵，作为时间项的离散 */
    //             for(uInt cellId = 0;cellId<cmesh.m_cells.size();cellId++){
    //                 sparse_mat.add_block(cellId, cellId, DenseMatrix<DoFs,DoFs>::Diag(mass[cellId]/dt));
    //                 rhs[cellId] += mass[cellId]/dt * U_n[cellId];
    //             }
    //             /* 收集了所有 Block 后，组装为稀疏矩阵 */
    //             sparse_mat.finalize();
    //         };
    //         /* ======================================================= *\
    //         **   第三层迭代，关于 时间步 dt 内的 非线性迭代
    //         **   需要判断是否超过 Dt，超过了就截断
    //         \* ======================================================= */
    //         for(uInt picard_iter = 0; picard_iter < 2000; picard_iter++){
    //             logger.log_picard_iteration(picard_iter);
                
    //             // logging("Start discretization");
    //             logger.log_discretization_start();

    //             LongVector<5*Basis::NumBasis> rhs(U_n.size());
    //             BlockSparseMatrix<5*Basis::NumBasis,5*Basis::NumBasis> sparse_mat;
    //             /* 用 第 k 步迭代的解 U_k 进行离散 */ 
    //             get_matrix(sparse_mat, rhs, U_k);

    //             logger.log_discretization_end();

    //             // sparse_mat.output_as_scalar("12345.txt");
                
    //             // logging("Start linear solver");

    //             logger.log_krylov_start();
    //             /* 调用 Eigen 求解线性方程组 */
    //             EigenSparseSolver<DoFs,DoFs> solver(sparse_mat,rhs);
    //             LongVector<DoFs> U_k_tmp(U_k.size());
    //             solver.set_iterations(1000);
    //             solver.set_tol(1e-14);
    //             auto [krylov_iters, krylov_residual] = solver.DGMRES(U_k, U_k_tmp);
    //             logger.log_krylov_end(krylov_iters, krylov_residual);

    //             if (krylov_residual > 1e-12){
    //                 logger.log_krylov_fallback_start();
    //                 U_k_tmp = solver.SparseLU(U_k);
    //                 // logger.output("        Krylov solver failed, using LU instead.");
    //                 logger.log_krylov_fallback_end();
    //             }

    //             // logging("Linear solver finished");

    //             // if(picard_iter>3){
    //             //     LongVector<DoFs> rhs(U_n.size());
    //             //     BlockSparseMatrix<DoFs,DoFs> sparse_mat;
    //             //     get_matrix(sparse_mat, rhs, U_k_tmp);
    //             //     const auto& res_new = (sparse_mat.multiply(U_k_tmp)) - (rhs);
    //             //     Scalar a = 0;
    //             //     a = (res_new-res_old).dot(res_new)/(res_new-res_old).dot(res_new-res_old);
    //             //     a = std::max(a,-5.0);
    //             //     a = std::min(a,0.1);

    //             //     // a = -0.5;
                    
    //             //     U_k_tmp = a*U_k + (1-a)*U_k_tmp;
    //             //     // debug(a);
    //             //     // res_old = res_new;
    //             // }
                
    //             // /* 降维模拟的后处理 */
    //             for(uInt cid = 0; cid < cmesh.m_cells.size(); cid++){
    //                 U_k_tmp[cid][3 + 5*0] = 0;
    //                 for (uInt k = 1; k < Basis::NumBasis; k++) {
    //                     U_k_tmp[cid][3 + 5*k] = 0;
    //                 }
    //             }

    //             // /* 施加限制器，这里是只有保极值保正，需要先计算允许的 Min Max */
    //             // pwenolimiter.apply(U_k_tmp, U_k); 
    //             positivelimiter.constructMinMax(U_k_tmp, U_k); 
    //             positivelimiter.apply(U_k_tmp, U_k); 

    //             /* 计算 新的解 U_k_tmp 与 上一非线性迭代步 U_k 的增量 */
    //             Scalar delta = (U_k_tmp - U_k).dot(U_k_tmp - U_k);
    //             delta = std::sqrt(delta);

    //             /* 替换 U_k 为新解 U_k_tmp */
    //             U_k = U_k_tmp;

    //                             /* 关于停机条件的处理，包括第一次迭代时的设置 */
    //             if(picard_iter==0) {
    //                 init_delta = delta;
    //                 prev_delta = delta;
    //             }
    //             Scalar rate_delta = delta/prev_delta;  // 下降率充分大
    //             Scalar rel_delta = delta/init_delta;   // 相对变化率足够小
    //             Scalar rate_res_norm, rel_res_norm;
    //             prev_delta = delta;

    //             /* 计算非线性残差，但没有用于停机条件 */
    //             std::ostringstream oss;
    //             oss << delta;
    //             Scalar res_norm;
    //             // logging("Re-discretize, calculate nonlinear residuals");
    //             {
    //                 LongVector<5*Basis::NumBasis> rhs(U_n.size());
    //                 BlockSparseMatrix<5*Basis::NumBasis,5*Basis::NumBasis> sparse_mat;
    //                 get_matrix(sparse_mat, rhs, U_k);
    //                 res_old = sparse_mat.multiply(U_k) - rhs;
    //                 res_norm = res_old.dot(res_old);
    //                 res_norm = std::sqrt(res_norm);
    //                 if(picard_iter==0) {
    //                     init_res_norm = res_norm;
    //                     prev_res_norm = res_norm;
    //                 }
    //                 rate_res_norm = res_norm/prev_res_norm;
    //                 rel_res_norm = res_norm/init_res_norm;
    //                 prev_res_norm = res_norm;
    //                 // std::ostringstream oss;
    //                 // oss << std::sqrt(res_norm);
    //                 // logging("Picard iter " + std::to_string(picard_iter) + "\t ||N(U_k)|| = " + oss.str());

    //             }
    //             // logging("Picard iter " + std::to_string(picard_iter) + "\t ||delta U|| = " + oss.str());
    //             logger.log_nonlinear_residual(res_norm, delta);
    //             logger.log_convergence_check(delta, rel_delta, res_norm, rel_res_norm, rate_res_norm);
    //             logger.end_picard_iteration();
    //             if(delta < 1e-8 || rel_delta < 1e-8 || 
    //                 res_norm < 1e-8 || rel_res_norm < 1e-8 || 
    //                 (picard_iter>100 && rate_res_norm > 1-1e-4 && rel_res_norm < 1e-3) || 
    //                 (picard_iter>20 && rate_res_norm > 1-1e-6)) {
    //                 // logging("delta = " + std::to_string(delta) + "    rel_delta = " + std::to_string(rel_delta) + "    res_norm = " + std::to_string(res_norm) + "    rel_res_norm = " + std::to_string(rel_res_norm) + "    rate_res_norm = " + std::to_string(rate_res_norm));
    //                 break;
    //             }
    //             if(res_norm>1e8){
    //                 sub_t = Dt; // 可能是数值不稳定，跳出迭代
    //                 break; // 可能是数值不稳定，跳出迭代
    //             }
    //         }
    //         /* 非线性迭代结束，U_n 赋值为最后的 U_k，作为新的时间步 U_{n+1} */ 
    //         U_n = U_k;
    //         logger.end_stage();
    //         if(total_time < max_dt) {
                    
    //             std::ofstream fp(fsm.get_solution_file(1000000 + uInt(100000000*total_time), N));
    //             auto [U_h, U_s, error] = reconstruct_solution<QuadC,Basis>(cmesh, U_n, 0.0);

    //             for(uInt cellId=0;cellId<cmesh.m_cells.size();cellId++){
    //                 const auto& cell = cmesh.m_cells[cellId];
    //                 for(uInt xgId=0; xgId<QuadC::num_points; ++xgId) {
    //                     const auto& p = QuadC::points[xgId];
    //                     const auto& pos = cell.transform_to_physical(p);
    //                     const auto& bUh = U_h[cellId].template SubMat<5,1>(5*xgId,0);
    //                     const auto& bUs = U_s[cellId].template SubMat<5,1>(5*xgId,0);
    //                     fp <<std::setprecision(16)<< pos[0] << "  " <<std::setprecision(16)<< pos[1] << "  " <<std::setprecision(16)<< pos[2]
    //                         << "  " <<std::setprecision(16)<<  bUh[0] << "  " <<std::setprecision(16)<<  bUs[0] 
    //                         << "  " <<std::setprecision(16)<<  bUh[1] << "  " <<std::setprecision(16)<<  bUs[1] 
    //                         << "  " <<std::setprecision(16)<<  bUh[2] << "  " <<std::setprecision(16)<<  bUs[2]
    //                         << "  " <<std::setprecision(16)<<  bUh[3] << "  " <<std::setprecision(16)<<  bUs[3] 
    //                         << "  " <<std::setprecision(16)<<  bUh[4] << "  " <<std::setprecision(16)<<  bUs[4] << std::endl;
    //                 }
    //             }
    //             fp.close();
    //         }
    //         // logging("Iter  " + std::to_string(save_step+1) + " \t Total time: " + std::to_string(total_time));
    //     }while(sub_t < Dt);

    //     std::ofstream fp(fsm.get_solution_file(save_step+1, N));
    //     auto err_integral = [&](LongVector<5*QuadC::num_points> U_h,LongVector<5*QuadC::num_points> U_s){
    //         DenseMatrix<5,1> err_per_cells = DenseMatrix<5,1>::Zeros();
    //         DenseMatrix<5,1> sol_per_cells = 1e-47 * DenseMatrix<5,1>::Ones();
    //         for(uInt cellId=0;cellId<cmesh.m_cells.size();cellId++){
    //             const auto& cell = cmesh.m_cells[cellId];
    //             for(uInt xgId=0; xgId<QuadC::num_points; ++xgId) {
    //                 const auto& bUh = U_h[cellId].template SubMat<5,1>(5*xgId,0);
    //                 const auto& bUs = U_s[cellId].template SubMat<5,1>(5*xgId,0);
    //                 const auto& bUe = bUh - bUs;
    //                 const auto& weight = QuadC::weights[xgId] * cell.compute_jacobian_det();
    //                 // Scalar error_cell = uh[cellId][xgId] - us[cellId][xgId];

    //                 err_per_cells += pow(bUe,2) * weight;
    //                 sol_per_cells += pow(bUs,2) * weight;
    //             }
    //         }
    //         return pow(err_per_cells/sol_per_cells,0.5);
    //     };

    //     auto [U_h, U_s, error] = reconstruct_solution<QuadC,Basis>(cmesh, U_n, total_time);

    //     for(uInt cellId=0;cellId<cmesh.m_cells.size();cellId++){
    //         const auto& cell = cmesh.m_cells[cellId];
    //         for(uInt xgId=0; xgId<QuadC::num_points; ++xgId) {
    //             const auto& p = QuadC::points[xgId];
    //             const auto& pos = cell.transform_to_physical(p);
    //             const auto& bUh = U_h[cellId].template SubMat<5,1>(5*xgId,0);
    //             const auto& bUs = U_s[cellId].template SubMat<5,1>(5*xgId,0);
    //             fp <<std::setprecision(16)<< pos[0] << "  " <<std::setprecision(16)<< pos[1] << "  " <<std::setprecision(16)<< pos[2]
    //              << "  " <<std::setprecision(16)<<  bUh[0] << "  " <<std::setprecision(16)<<  bUs[0] 
    //              << "  " <<std::setprecision(16)<<  bUh[1] << "  " <<std::setprecision(16)<<  bUs[1] 
    //              << "  " <<std::setprecision(16)<<  bUh[2] << "  " <<std::setprecision(16)<<  bUs[2]
    //              << "  " <<std::setprecision(16)<<  bUh[3] << "  " <<std::setprecision(16)<<  bUs[3] 
    //              << "  " <<std::setprecision(16)<<  bUh[4] << "  " <<std::setprecision(16)<<  bUs[4] << std::endl;
    //         }
    //     }
    //     fp.close();

    //     const auto& U_err = err_integral(U_h,U_s);
    //     // print(std::array<Scalar,8>{total_time, U_err[0], U_err[1],U_err[2],U_err[3],U_err[4], 
    //     //         std::sqrt(error.dot(error)/U_n.dot(U_n)), chrone_clock()});
    //     // print(vector3f{curr_time, std::sqrt(err.dot(err)/rho.dot(rho)),chrone_clock()});
    // }
}


