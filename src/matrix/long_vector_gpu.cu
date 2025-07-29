#include "base/type.h"
#include "matrix/long_vector.h"
#include "matrix/long_vector_device.h"
#include <cuda_runtime.h>

// Host -> Device 上传
template <uInt BlockSize>
LongVectorDevice<BlockSize> LongVector<BlockSize>::to_device() const {
    printf("从 CPU 上传到 GPU 向量   ");
    LongVectorDevice<BlockSize> dev;
    dev.length = blocks.size();
    cudaMalloc(&dev.d_blocks, dev.length * sizeof(DenseMatrix<BlockSize, 1>));
    cudaMemcpy(dev.d_blocks, blocks.data(), dev.length * sizeof(DenseMatrix<BlockSize, 1>), cudaMemcpyHostToDevice);
    return dev;
}

// Device -> Host 下载
template <uInt BlockSize>
LongVector<BlockSize> LongVectorDevice<BlockSize>::download() const {
    printf("从 GPU 下载到 CPU 向量   "); 
    LongVector<BlockSize> host(length);
    cudaMemcpy(host.blocks.data(), d_blocks, length * sizeof(DenseMatrix<BlockSize, 1>), cudaMemcpyDeviceToHost);
    return host;
}

template <uInt BlockSize>
__global__ void kernel_fill_with_scalar(DenseMatrix<BlockSize, 1>* d_blocks, size_t length, Scalar val) {
    uInt i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
        for (uInt j = 0; j < BlockSize; ++j)
            d_blocks[i](j, 0) = val;
    }
}

template <uInt BlockSize>
void LongVectorDevice<BlockSize>::fill_with_scalar(Scalar val) {
    // printf("用标量 %.3f 填充 GPU 向量，长度 %lu\n", val, length);

    // if (d_blocks) cudaFree(d_blocks);
    // cudaMalloc(&d_blocks, length * sizeof(DenseMatrix<BlockSize, 1>));

    if (!d_blocks) return;

    // 初始化每个 block 的值
    if (val == 0.0) {
        cudaMemset(d_blocks, 0, length * sizeof(DenseMatrix<BlockSize, 1>));
    } else {
        printf("目前非零的功能还没有测试！ \n");
        constexpr int BLOCK_SIZE = 32;
        int num_blocks = (length + BLOCK_SIZE - 1) / BLOCK_SIZE;
        kernel_fill_with_scalar<BlockSize><<<num_blocks, BLOCK_SIZE>>>(d_blocks, length, val);
        // cudaDeviceSynchronize();  // 可选：便于调试错误
    }
}


template class LongVector<1>;
template class LongVector<2>;
template class LongVector<3>;
template class LongVector<4>;
template class LongVector<5>;
template class LongVector<8>;
template class LongVector<10>;
template class LongVector<12>;
template class LongVector<13>;
template class LongVector<16>;
template class LongVector<20>;
template class LongVector<30>;
template class LongVector<31>;
template class LongVector<34>;
template class LongVector<35>;
template class LongVector<40>;
template class LongVector<50>;
template class LongVector<56>;
template class LongVector<60>;
template class LongVector<64>;
template class LongVector<70>;
template class LongVector<80>;
template class LongVector<84>;
template class LongVector<100>;
template class LongVector<105>;
template class LongVector<112>;
template class LongVector<115>;
template class LongVector<120>;
template class LongVector<125>;
template class LongVector<140>;
template class LongVector<168>;
template class LongVector<175>;
template class LongVector<180>;
template class LongVector<188>;
template class LongVector<200>;
template class LongVector<203>;
template class LongVector<224>;
template class LongVector<240>;
template class LongVector<252>;
template class LongVector<280>;
template class LongVector<287>;
template class LongVector<300>;
template class LongVector<308>;
template class LongVector<336>;
template class LongVector<360>;
template class LongVector<400>;
template class LongVector<420>;
template class LongVector<480>;
template class LongVector<600>;
template class LongVector<700>;
template class LongVector<720>;
template class LongVector<800>;
template class LongVector<900>;
template class LongVector<1120>;
template class LongVector<1200>;
template class LongVector<1440>;
template class LongVector<1680>;
template class LongVector<1800>;
template class LongVector<2000>;
template class LongVector<2100>;
template class LongVector<2400>;
template class LongVector<3000>;
template class LongVector<3360>;
template class LongVector<3600>;
template class LongVector<4000>;
template class LongVector<4200>;
template class LongVector<5040>;
template class LongVector<6000>;
template class LongVector<6300>;
template class LongVector<6720>;
template class LongVector<7000>;
template class LongVector<7200>;
template class LongVector<9000>;
template class LongVector<10080>;
template class LongVector<10500>;
template class LongVector<11200>;
template class LongVector<12000>;
template class LongVector<12600>;
template class LongVector<15120>;
template class LongVector<16800>;
template class LongVector<18000>;
template class LongVector<20160>;
template class LongVector<21000>;
template class LongVector<25200>;
template class LongVector<30240>;
template class LongVector<31500>;
template class LongVector<33600>;
template class LongVector<50400>;
template class LongVector<75600>;
template class LongVectorDevice<1>;
template class LongVectorDevice<2>;
template class LongVectorDevice<3>;
template class LongVectorDevice<4>;
template class LongVectorDevice<5>;
template class LongVectorDevice<8>;
template class LongVectorDevice<10>;
template class LongVectorDevice<12>;
template class LongVectorDevice<13>;
template class LongVectorDevice<16>;
template class LongVectorDevice<20>;
template class LongVectorDevice<30>;
template class LongVectorDevice<31>;
template class LongVectorDevice<34>;
template class LongVectorDevice<35>;
template class LongVectorDevice<40>;
template class LongVectorDevice<50>;
template class LongVectorDevice<56>;
template class LongVectorDevice<60>;
template class LongVectorDevice<64>;
template class LongVectorDevice<70>;
template class LongVectorDevice<80>;
template class LongVectorDevice<84>;
template class LongVectorDevice<100>;
template class LongVectorDevice<105>;
template class LongVectorDevice<112>;
template class LongVectorDevice<115>;
template class LongVectorDevice<120>;
template class LongVectorDevice<125>;
template class LongVectorDevice<140>;
template class LongVectorDevice<168>;
template class LongVectorDevice<175>;
template class LongVectorDevice<180>;
template class LongVectorDevice<188>;
template class LongVectorDevice<200>;
template class LongVectorDevice<203>;
template class LongVectorDevice<224>;
template class LongVectorDevice<240>;
template class LongVectorDevice<252>;
template class LongVectorDevice<280>;
template class LongVectorDevice<287>;
template class LongVectorDevice<300>;
template class LongVectorDevice<308>;
template class LongVectorDevice<336>;
template class LongVectorDevice<360>;
template class LongVectorDevice<400>;
template class LongVectorDevice<420>;
template class LongVectorDevice<480>;
template class LongVectorDevice<600>;
template class LongVectorDevice<700>;
template class LongVectorDevice<720>;
template class LongVectorDevice<800>;
template class LongVectorDevice<900>;
template class LongVectorDevice<1120>;
template class LongVectorDevice<1200>;
template class LongVectorDevice<1440>;
template class LongVectorDevice<1680>;
template class LongVectorDevice<1800>;
template class LongVectorDevice<2000>;
template class LongVectorDevice<2100>;
template class LongVectorDevice<2400>;
template class LongVectorDevice<3000>;
template class LongVectorDevice<3360>;
template class LongVectorDevice<3600>;
template class LongVectorDevice<4000>;
template class LongVectorDevice<4200>;
template class LongVectorDevice<5040>;
template class LongVectorDevice<6000>;
template class LongVectorDevice<6300>;
template class LongVectorDevice<6720>;
template class LongVectorDevice<7000>;
template class LongVectorDevice<7200>;
template class LongVectorDevice<9000>;
template class LongVectorDevice<10080>;
template class LongVectorDevice<10500>;
template class LongVectorDevice<11200>;
template class LongVectorDevice<12000>;
template class LongVectorDevice<12600>;
template class LongVectorDevice<15120>;
template class LongVectorDevice<16800>;
template class LongVectorDevice<18000>;
template class LongVectorDevice<20160>;
template class LongVectorDevice<21000>;
template class LongVectorDevice<25200>;
template class LongVectorDevice<30240>;
template class LongVectorDevice<31500>;
template class LongVectorDevice<33600>;
template class LongVectorDevice<50400>;
template class LongVectorDevice<75600>;