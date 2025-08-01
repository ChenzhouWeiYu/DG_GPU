#include "base/type.h"
#include "dg/time_integrator.cuh"


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


template<uInt DoFs, uInt Order, bool OnlyNeigbAvg>
TimeIntegrator<DoFs, Order, OnlyNeigbAvg>::TimeIntegrator(
    const DeviceMesh& mesh,
    LongVectorDevice<DoFs>& U_n,
    const LongVectorDevice<DoFs>& r_mass,
    PositiveLimiterGPU<Order, typename AutoQuadSelector<Order, GaussLegendreTet::Auto>::type, typename AutoQuadSelector<Order, GaussLegendreTri::Auto>::type, OnlyNeigbAvg>& positivelimiter,
    WENOLimiterGPU<Order, typename AutoQuadSelector<Order, GaussLegendreTet::Auto>::type, typename AutoQuadSelector<Order, GaussLegendreTri::Auto>::type>& wenolimiter,
    PWeightWENOLimiterGPU<Order, typename AutoQuadSelector<Order, GaussLegendreTet::Auto>::type, typename AutoQuadSelector<Order, GaussLegendreTri::Auto>::type>& pweightwenolimiter)
    : mesh_(mesh), U_n_(U_n), r_mass_(r_mass), positivelimiter(positivelimiter), wenolimiter(wenolimiter), pweightwenolimiter(pweightwenolimiter)
{
    // 初始化内部缓存（大小与 U_n 一致）
    U_1_.resize(U_n.size());

    // 默认 Euler 格式
    scheme_ = TimeIntegrationScheme::EULER;
}

template<uInt DoFs, uInt Order, bool OnlyNeigbAvg>
void TimeIntegrator<DoFs, Order, OnlyNeigbAvg>::set_scheme(TimeIntegrationScheme scheme) {
    scheme_ = scheme;
    if(scheme == TimeIntegrationScheme::SSP_RK3){
        U_2_.resize(U_n_.size());
        U_temp_.resize(U_n_.size());
    }
}


template<uInt DoFs, uInt Order, bool OnlyNeigbAvg>
template<typename FluxType>
void TimeIntegrator<DoFs, Order, OnlyNeigbAvg>::advance(
    ExplicitConvectionGPU<Order,FluxType>& convection,
    Scalar curr_time,
    Scalar dt,
    uInt limiter_flag
)
{
    uInt size = U_n_.size();
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);

    switch (scheme_) {
        case TimeIntegrationScheme::EULER: {
            // Euler 更新：U_n += dt * r_mass .* R(U_n)
            if(limiter_flag & (1<<0)) positivelimiter.constructMinMax(U_n_);
            // U_1_.fill_with_scalar(0.0);
            convection.eval(mesh_, U_n_, U_1_, curr_time);
            update_solution<<<grid, block>>>(U_n_.d_blocks, U_1_.d_blocks, r_mass_.d_blocks, dt, size);
            // cudaDeviceSynchronize();
            if(limiter_flag & (1<<1)) pweightwenolimiter.apply(U_n_);
            if(limiter_flag & (1<<0)) positivelimiter.apply(U_n_);
            break;
        }

        case TimeIntegrationScheme::SSP_RK3: {
            // Stage 1: U_1 = U_n - dt * r_mass .* R(U_n)
            if(limiter_flag & 1<<0) positivelimiter.constructMinMax(U_n_);
            // U_1_.fill_with_scalar(0.0);
            convection.eval(mesh_, U_n_, U_1_, curr_time);
            rk_stage1<<<grid, block>>>(U_1_.d_blocks, U_n_.d_blocks, U_1_.d_blocks, r_mass_.d_blocks, dt, size);
            // cudaDeviceSynchronize();
            if(limiter_flag & 1<<1) pweightwenolimiter.apply(U_1_);
            if(limiter_flag & 1<<0) positivelimiter.apply(U_1_);

            // Stage 2: U_2 = 3/4 U_n + 1/4 U_1 - 1/4 dt * r_mass .* R(U_1)
            if(limiter_flag & 1<<0) positivelimiter.constructMinMax(U_1_);
            // U_2_.fill_with_scalar(0.0);
            convection.eval(mesh_, U_1_, U_2_, curr_time + dt);
            rk_stage2<<<grid, block>>>(U_2_.d_blocks, U_n_.d_blocks, U_1_.d_blocks, U_2_.d_blocks, r_mass_.d_blocks, dt, size);
            // cudaDeviceSynchronize();
            if(limiter_flag & 1<<1) pweightwenolimiter.apply(U_2_);
            if(limiter_flag & 1<<0) positivelimiter.apply(U_2_);

            // Stage 3: U_n = 1/3 U_n + 2/3 U_2 - 2/3 dt * r_mass .* R(U_2)
            if(limiter_flag & 1<<0) positivelimiter.constructMinMax(U_2_);
            // U_temp_.fill_with_scalar(0.0);
            convection.eval(mesh_, U_2_, U_temp_, curr_time + 0.5*dt);
            rk_stage3<<<grid, block>>>(U_n_.d_blocks, U_n_.d_blocks, U_2_.d_blocks, U_temp_.d_blocks, r_mass_.d_blocks, dt, size);
            // cudaDeviceSynchronize();
            if(limiter_flag & 1<<1) pweightwenolimiter.apply(U_n_);
            if(limiter_flag & 1<<0) positivelimiter.apply(U_n_);
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



// template class TimeIntegrator<5*DGBasisEvaluator<1>::NumBasis, 1>;
// template class TimeIntegrator<5*DGBasisEvaluator<2>::NumBasis, 2>;
// template class TimeIntegrator<5*DGBasisEvaluator<3>::NumBasis, 3>;
// template class TimeIntegrator<5*DGBasisEvaluator<4>::NumBasis, 4>;
// template class TimeIntegrator<5*DGBasisEvaluator<5>::NumBasis, 5>;

// template void TimeIntegrator<5*DGBasisEvaluator<1>::NumBasis, 1>::advance(ExplicitConvectionGPU<1,AirFluxC>&,Scalar,Scalar);
// template void TimeIntegrator<5*DGBasisEvaluator<1>::NumBasis, 1>::advance(ExplicitConvectionGPU<1,MonatomicFluxC>&,Scalar,Scalar);

#define Explicit_For_Flux(NAME,Order) \
template void TimeIntegrator<5*DGBasisEvaluator<Order>::NumBasis, Order, true>::advance(ExplicitConvectionGPU<Order,NAME##75C>&,Scalar,Scalar,uInt);\
template void TimeIntegrator<5*DGBasisEvaluator<Order>::NumBasis, Order, true>::advance(ExplicitConvectionGPU<Order,NAME##53C>&,Scalar,Scalar,uInt);\
template void TimeIntegrator<5*DGBasisEvaluator<Order>::NumBasis, Order, false>::advance(ExplicitConvectionGPU<Order,NAME##75C>&,Scalar,Scalar,uInt);\
template void TimeIntegrator<5*DGBasisEvaluator<Order>::NumBasis, Order, false>::advance(ExplicitConvectionGPU<Order,NAME##53C>&,Scalar,Scalar,uInt);\


#define explict_template_instantiation(Order)\
template class TimeIntegrator<5*DGBasisEvaluator<Order>::NumBasis, Order, true>;\
template class TimeIntegrator<5*DGBasisEvaluator<Order>::NumBasis, Order, false>;\
FOREACH_FLUX_TYPE(Explicit_For_Flux,Order)\


explict_template_instantiation(0)
explict_template_instantiation(1)
explict_template_instantiation(2)
explict_template_instantiation(3)
explict_template_instantiation(4)
explict_template_instantiation(5)
#undef explict_template_instantiation

#undef Explicit_For_Flux
