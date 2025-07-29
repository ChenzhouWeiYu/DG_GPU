#include "base/type.h"
#include "rte_time_integrator.h"


// Kernel 封装
template<uInt DoFs>
__global__ void rk_stage(DenseMatrix<DoFs, 1>* U_out,
                            const DenseMatrix<DoFs, 1>* U_in,
                            const DenseMatrix<DoFs, 1>* R_in,
                            Scalar dt, uInt size);
template<uInt DoFs>
__global__ void rk_stage_mixed(DenseMatrix<DoFs, 1>* U_out,
                                const DenseMatrix<DoFs, 1>* U_a,
                                const DenseMatrix<DoFs, 1>* U_b,
                                const DenseMatrix<DoFs, 1>* R_b,
                                Scalar dt,
                                Scalar alpha_a, Scalar alpha_b, Scalar beta_b,
                                uInt size);





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
        // if (i % 5 == 3) U_n[cellId](i, 0) = 0.0;
    }
}


template<uInt DoFs>
__global__ void rk_stage1(
    DenseMatrix<DoFs, 1>* U_1,
    const DenseMatrix<DoFs, 1>* U_n,
    const DenseMatrix<DoFs, 1>* R_n,
    const DenseMatrix<DoFs, 1>* r_mass,
    Scalar dt, uInt size)
{
    uInt cellId = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellId >= size) return;

    for (int i = 0; i < DoFs; ++i) {
        // Stage 1: U_1 = U_n - dt * r_mass * R(U_n)
        U_1[cellId](i, 0) = U_n[cellId](i, 0) - dt * r_mass[cellId](i, 0) * R_n[cellId](i, 0);

        // if (i % 5 == 3) U_1[cellId](i, 0) = 0.0;
    }
}

template<uInt DoFs>
__global__ void rk_stage2(
    DenseMatrix<DoFs, 1>* U_2,
    const DenseMatrix<DoFs, 1>* U_n,
    const DenseMatrix<DoFs, 1>* U_1,
    const DenseMatrix<DoFs, 1>* R_1,
    const DenseMatrix<DoFs, 1>* r_mass,
    Scalar dt, uInt size)
{
    uInt cellId = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellId >= size) return;

    for (int i = 0; i < DoFs; ++i) {
        // Stage 2: U_2 = 3/4 U_n + 1/4 U_1 - 1/4 dt * r_mass * R(U_1)
        U_2[cellId](i, 0) = 0.75 * U_n[cellId](i, 0) + 0.25 * U_1[cellId](i, 0)
                         - 0.25 * dt * r_mass[cellId](i, 0) * R_1[cellId](i, 0);

        // if (i % 5 == 3) U_2[cellId](i, 0) = 0.0;
    }
}

template<uInt DoFs>
__global__ void rk_stage3(
    DenseMatrix<DoFs, 1>* U_n,
    const DenseMatrix<DoFs, 1>* U_n_old,
    const DenseMatrix<DoFs, 1>* U_2,
    const DenseMatrix<DoFs, 1>* R_2,
    const DenseMatrix<DoFs, 1>* r_mass,
    Scalar dt, uInt size)
{
    uInt cellId = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellId >= size) return;

    for (int i = 0; i < DoFs; ++i) {
        // Stage 3: U_{n+1} = 1/3 U_n + 2/3 U_2 - 2/3 dt * r_mass * R(U_2)
        U_n[cellId](i, 0) = (1.0 / 3.0) * U_n_old[cellId](i, 0)
                         + (2.0 / 3.0) * U_2[cellId](i, 0)
                         - (2.0 / 3.0) * dt * r_mass[cellId](i, 0) * R_2[cellId](i, 0);

        // if (i % 5 == 3) U_n[cellId](i, 0) = 0.0;
    }
}


template<uInt X3Order, uInt S2Order,typename GaussQuadCell,typename GaussQuadTri,typename S2Mesh>
RTE_TimeIntegrator<X3Order,S2Order,GaussQuadCell,GaussQuadTri,S2Mesh>::RTE_TimeIntegrator(
    const DeviceMesh& mesh,
    LongVectorDevice<DoFs>& U_n,
    const LongVectorDevice<DoFs>& r_mass)
    : mesh_(mesh), U_n_(U_n), r_mass_(r_mass)
{
    // 初始化内部缓存（大小与 U_n 一致）
    U_1_.resize(U_n.size());

    // 默认 Euler 格式
    scheme_ = TimeIntegrationScheme::EULER;
}



template<uInt X3Order, uInt S2Order,typename GaussQuadCell,typename GaussQuadTri,typename S2Mesh>
void RTE_TimeIntegrator<X3Order,S2Order,GaussQuadCell,GaussQuadTri,S2Mesh>::set_scheme(TimeIntegrationScheme scheme) {
    scheme_ = scheme;
    if(scheme == TimeIntegrationScheme::SSP_RK3){
        U_2_.resize(U_n_.size());
        U_temp_.resize(U_n_.size());
    }
}



template<uInt X3Order, uInt S2Order,typename GaussQuadCell,typename GaussQuadTri,typename S2Mesh>
void RTE_TimeIntegrator<X3Order,S2Order,GaussQuadCell,GaussQuadTri,S2Mesh>::advance(
    ExplicitRTEGPU<X3Order,S2Order,GaussQuadCell,GaussQuadTri,S2Mesh>& convection,
    Scalar curr_time, Scalar dt){
    const uInt size = U_n_.size();
    const int num_blocks = (size + 31) / 32;

    switch (scheme_) {
        case TimeIntegrationScheme::EULER: {
            // Euler 更新：U_n += dt * r_mass .* R(U_n)
            U_1_.fill_with_scalar(0.0);
            convection.eval(mesh_, U_n_, U_1_, curr_time);
            update_solution<<<num_blocks, 32>>>(U_n_.d_blocks, U_1_.d_blocks, r_mass_.d_blocks, dt, size);
            cudaDeviceSynchronize();
            break;
        }

        case TimeIntegrationScheme::SSP_RK3: {
            // Stage 1: U_1 = U_n - dt * r_mass .* R(U_n)
            U_1_.fill_with_scalar(0.0);
            convection.eval(mesh_, U_n_, U_1_, curr_time);
            rk_stage1<<<num_blocks, 32>>>(U_1_.d_blocks, U_n_.d_blocks, U_1_.d_blocks, r_mass_.d_blocks, dt, size);
            cudaDeviceSynchronize();

            // Stage 2: U_2 = 3/4 U_n + 1/4 U_1 - 1/4 dt * r_mass .* R(U_1)
            U_2_.fill_with_scalar(0.0);
            convection.eval(mesh_, U_1_, U_2_, curr_time + dt);
            rk_stage2<<<num_blocks, 32>>>(U_2_.d_blocks, U_n_.d_blocks, U_1_.d_blocks, U_2_.d_blocks, r_mass_.d_blocks, dt, size);
            cudaDeviceSynchronize();

            // Stage 3: U_n = 1/3 U_n + 2/3 U_2 - 2/3 dt * r_mass .* R(U_2)
            U_temp_.fill_with_scalar(0.0);
            convection.eval(mesh_, U_2_, U_temp_, curr_time + 0.5*dt);
            rk_stage3<<<num_blocks, 32>>>(U_n_.d_blocks, U_n_.d_blocks, U_2_.d_blocks, U_temp_.d_blocks, r_mass_.d_blocks, dt, size);
            cudaDeviceSynchronize();
            break;
        }

        default:
            std::cerr << "Unsupported time integration scheme." << std::endl;
            exit(-1);
    }
}

// Euler & RK3 更新 Kernel（内核函数）
template<uInt DoFs>
__global__ void rk_stage(
    DenseMatrix<DoFs, 1>* U_out,
    const DenseMatrix<DoFs, 1>* U_in,
    const DenseMatrix<DoFs, 1>* R_in,
    Scalar dt,
    uInt size)
{
    uInt cellId = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellId >= size) return;

    for (int i = 0; i < DoFs; ++i) {
        U_out[cellId](i, 0) = U_in[cellId](i, 0) - dt * R_in[cellId](i, 0);
        // if (i % 5 == 3) U_out[cellId](i, 0) = 0.0;  // 特定变量置零（可选）
    }
}

template<uInt DoFs>
__global__ void rk_stage_mixed(
    DenseMatrix<DoFs, 1>* U_out,
    const DenseMatrix<DoFs, 1>* U_a,
    const DenseMatrix<DoFs, 1>* U_b,
    const DenseMatrix<DoFs, 1>* R_b,
    Scalar dt,
    Scalar alpha_a, Scalar alpha_b, Scalar beta_b,
    uInt size)
{
    uInt cellId = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellId >= size) return;

    for (int i = 0; i < DoFs; ++i) {
        U_out[cellId](i, 0) = alpha_a * U_a[cellId](i, 0)
                           + alpha_b * U_b[cellId](i, 0)
                           - beta_b * dt * R_b[cellId](i, 0);
        // if (i % 5 == 3) U_out[cellId](i, 0) = 0.0;
    }
}



#define explict_template_instantiation(Order)\
template class RTE_TimeIntegrator<Order, Order,\
    typename AutoQuadSelector<Order, GaussLegendreTet::Auto>::type,\
    typename AutoQuadSelector<Order, GaussLegendreTri::Auto>::type>;

explict_template_instantiation(1)
explict_template_instantiation(2)
#undef explict_template_instantiation