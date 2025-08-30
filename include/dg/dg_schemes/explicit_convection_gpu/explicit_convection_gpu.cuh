#pragma once

#include "base/type.h"
#include "dg/dg_basis/dg_basis.h"
#include "mesh/device_mesh.cuh"
#include "matrix/matrix.h"
#include "base/exact.h"
#include "dg/dg_flux/euler_physical_flux.h"
#include "matrix/long_vector_device.cuh"

// GPU 显式对流核
template<uInt Order=3, typename Flux = AirFluxC, 
         typename GaussQuadCell = GaussLegendreTet::Auto, 
         typename GaussQuadFace = GaussLegendreTri::Auto>
class ExplicitConvectionGPU {
private:
    using BlockMat = DenseMatrix<5,5>;
    using Basis = DGBasisEvaluator<Order>;
    
    using QuadC = typename std::conditional_t<
        std::is_same_v<GaussQuadCell, GaussLegendreTet::Auto>,
        typename AutoQuadSelector<Order, GaussLegendreTet::Auto>::type,
        GaussQuadCell
    >;
    using QuadF = typename std::conditional_t<
        std::is_same_v<GaussQuadFace, GaussLegendreTri::Auto>,
        typename AutoQuadSelector<Order, GaussLegendreTri::Auto>::type,
        GaussQuadFace
    >;

    static constexpr uInt N = Basis::NumBasis;
    std::vector<DeviceMesh> mgpu_mesh_;
    std::vector<LongVectorDevice<5*N>> mgpu_U_;
    std::vector<LongVectorDevice<5*N>> mgpu_rhs_;
    int dev_cnt_ = 1;
    // __device__ static 
    // vector3f transform_to_cell(const GPUTriangleFace& face, const vector2f& uv, uInt side) const;

public:
    ExplicitConvectionGPU() = default;
    ~ExplicitConvectionGPU() = default;

    ExplicitConvectionGPU(const DeviceMesh& mesh) {
        cudaGetDeviceCount(&dev_cnt_);
        mgpu_mesh_.resize(dev_cnt_);
        mgpu_U_.resize(dev_cnt_);
        mgpu_rhs_.resize(dev_cnt_);
        // std::cout << "device count: " << dev_cnt_ << std::endl;
        for(int g=0; g<dev_cnt_; ++g) {
            cudaSetDevice(g);

            // std::cout << "device " << g << " init" << std::endl;
            // mesh 拷贝
            mgpu_mesh_[g].initialize_from(mesh);
            mgpu_mesh_[g].upload_to_gpu();
            // std::cout << "device " << g << " init done" << std::endl;

            // std::cout << "device " << g << " U init" << std::endl;
            // 分配 GPU 向量
            mgpu_U_[g].resize(mesh.num_cells());
            mgpu_rhs_[g].resize(mesh.num_cells());
            cudaMemset(mgpu_U_[g].d_blocks, 0, mesh.num_cells()*sizeof(DenseMatrix<5*N,1>));
            cudaMemset(mgpu_rhs_[g].d_blocks, 0, mesh.num_cells()*sizeof(DenseMatrix<5*N,1>));
            // std::cout << "device " << g << " U init done" << std::endl;
            // cudaError_t err = cudaGetLastError();
            // if (err != cudaSuccess) {
            //     printf("ExplicitConvectionGPU   CUDA kernel launch error: %s, GPU id: %d\n", cudaGetErrorString(err), g);
            // }
            // cudaDeviceSynchronize();
        }
        cudaSetDevice(0);
    }
    // 3个 kernel launcher
    void eval_cells(const DeviceMesh& mesh, 
                    const LongVectorDevice<5*N>& U,
                    LongVectorDevice<5*N>& rhs);
                    
    void eval_internals(const DeviceMesh& mesh, 
                        const LongVectorDevice<5*N>& U,
                        LongVectorDevice<5*N>& rhs);
                        
    void eval_boundarys(const DeviceMesh& mesh, 
                        const LongVectorDevice<5*N>& U,
                        LongVectorDevice<5*N>& rhs, Scalar time = 0.0);

    void eval(const DeviceMesh& mesh, 
                        const LongVectorDevice<5*N>& U,
                        LongVectorDevice<5*N>& rhs, Scalar time = 0.0);
};









// #define Explicit_For_Flux(Order)\
// extern template class ExplicitConvectionGPU<Order,LF75C>;\
// extern template class ExplicitConvectionGPU<Order,LF53C>;\
// extern template class ExplicitConvectionGPU<Order,Roe75C>;\
// extern template class ExplicitConvectionGPU<Order,Roe53C>;\
// extern template class ExplicitConvectionGPU<Order,HLL75C>;\
// extern template class ExplicitConvectionGPU<Order,HLL53C>;\
// extern template class ExplicitConvectionGPU<Order,HLLC75C>;\
// extern template class ExplicitConvectionGPU<Order,HLLC53C>;\
// extern template class ExplicitConvectionGPU<Order,LaxFriedrichs75C>;\
// extern template class ExplicitConvectionGPU<Order,LaxFriedrichs53C>;

#define Explicit_For_Flux(NAME,Order) \
extern template class ExplicitConvectionGPU<Order,NAME##75C>;\
extern template class ExplicitConvectionGPU<Order,NAME##53C>;

FOREACH_FLUX_TYPE(Explicit_For_Flux,0)
FOREACH_FLUX_TYPE(Explicit_For_Flux,1)
FOREACH_FLUX_TYPE(Explicit_For_Flux,2)
FOREACH_FLUX_TYPE(Explicit_For_Flux,3)
FOREACH_FLUX_TYPE(Explicit_For_Flux,4)
FOREACH_FLUX_TYPE(Explicit_For_Flux,5)

#undef Explicit_For_Flux
// Explicit_For_Flux(0)
// Explicit_For_Flux(1)
// Explicit_For_Flux(2)
// Explicit_For_Flux(3)
// Explicit_For_Flux(4)
// Explicit_For_Flux(5)

// #undef Explicit_For_Flux