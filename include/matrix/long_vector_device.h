#pragma once
#include "base/type.h"
#include "matrix/dense_matrix.h"

template <uInt BlockSize>
class LongVector ;

template <uInt BlockSize>
class LongVectorDevice {
public:
    DenseMatrix<BlockSize, 1>* d_blocks = nullptr;
    uInt length = 0;

    LongVectorDevice() = default;

    HostDevice LongVectorDevice(uInt len) : length(len) {
        // printf("构造 GPU 向量，长度 %lu\n", len);
        cudaMalloc(&d_blocks, length * sizeof(DenseMatrix<BlockSize, 1>));
    }

    HostDevice ~LongVectorDevice() {
        if (d_blocks) cudaFree(d_blocks);
    }

    HostDevice DenseMatrix<BlockSize,1>& operator[](uInt i) { return d_blocks[i]; }
    HostDevice const DenseMatrix<BlockSize,1>& operator[](uInt i) const { return d_blocks[i]; }
    HostDevice uInt size() const { return length; }

    HostDevice void resize(uInt len) {
        if (d_blocks) cudaFree(d_blocks);
        length = len;
        cudaMalloc(&d_blocks, length * sizeof(DenseMatrix<BlockSize, 1>));
    }

        // ------------------ 填充接口 ------------------

    // 填充为某个标量（所有 entry = val）
    void fill_with_scalar(Scalar val);
    //  {
    //     printf("用标量 %.3f 填充 GPU 向量，长度 %lu\n", val, length);
    //     printf("目前这个功能还没有测试！ \n");
    //     // if (d_blocks) cudaFree(d_blocks);
    //     // cudaMalloc(&d_blocks, length * sizeof(DenseMatrix<BlockSize, 1>));
    //     // DenseMatrix<BlockSize, 1> block = DenseMatrix<BlockSize, 1>::Ones() * val;
    //     // for (uInt i = 0; i < length; ++i)
    //     //     cudaMemcpy(&d_blocks[i], &block, sizeof(DenseMatrix<BlockSize, 1>), cudaMemcpyHostToDevice);
    // }

    // 填充为全 0
    void fill_zeros() {
        // printf("填充为零，长度 %lu\n", length);
        fill_with_scalar(0.0);
    }

    // 用一个固定的 block 填充
    void fill_with_block(const DenseMatrix<BlockSize, 1>& val) {
        printf("用给定 DenseMatrix 段填充，长度 %lu\n", length);
        if (d_blocks) cudaFree(d_blocks);
        cudaMalloc(&d_blocks, length * sizeof(DenseMatrix<BlockSize, 1>));
        for (uInt i = 0; i < length; ++i)
            cudaMemcpy(&d_blocks[i], &val, sizeof(DenseMatrix<BlockSize, 1>), cudaMemcpyHostToDevice);
    }

    // 从 CPU 的 LongVector<BlockSize> 复制数据
    void fill_from_host(const LongVector<BlockSize>& h_vec) {
        printf("从 CPU 向量复制，长度 %lu\n", h_vec.size());
        if (d_blocks) cudaFree(d_blocks);
        length = h_vec.size();
        cudaMalloc(&d_blocks, length * sizeof(DenseMatrix<BlockSize, 1>));
        cudaMemcpy(d_blocks, h_vec.blocks.data(),
                   length * sizeof(DenseMatrix<BlockSize, 1>),
                   cudaMemcpyHostToDevice);
    }

    // 为初始化而构造的接口（可选）
    explicit LongVectorDevice(const LongVector<BlockSize>& h_vec) {
        printf("构造 GPU 向量（来自 CPU LongVector），长度 %lu\n", h_vec.size());
        length = h_vec.size();
        cudaMalloc(&d_blocks, length * sizeof(DenseMatrix<BlockSize, 1>));
        cudaMemcpy(d_blocks, h_vec.blocks.data(),
                   length * sizeof(DenseMatrix<BlockSize, 1>),
                   cudaMemcpyHostToDevice);
    }

    // 显式释放（可选）
    void free() {
        if (d_blocks) cudaFree(d_blocks);
        d_blocks = nullptr;
        length = 0;
    }



    // 四则运算
    #define ELEMENT_WISE_OP(op)\
    __device__ LongVectorDevice& operator op##=(const LongVectorDevice& rhs) {\
        for(uInt i=0; i<length; ++i)\
            d_blocks[i] op##= rhs.d_blocks[i];\
        return *this;\
    }\
    __device__ LongVectorDevice operator op(const LongVectorDevice& rhs) const {\
        LongVectorDevice res(*this);\
        return res op##= rhs;\
    }\
    __device__ LongVectorDevice& operator op##=(Scalar val) {\
        for(uInt i=0; i<length; ++i)\
            d_blocks[i] op##= val;\
        return *this;\
    }\
    __device__ LongVectorDevice operator op(Scalar val) const {\
        LongVectorDevice result(*this);\
        return result op##= val;\
    }

    ELEMENT_WISE_OP(+)
    ELEMENT_WISE_OP(-)
    ELEMENT_WISE_OP(*)
    ELEMENT_WISE_OP(/)
    #undef ELEMENT_WISE_OP


    __device__ Scalar dot(const LongVectorDevice& rhs) const {
        Scalar sum = 0.0;
        for (uInt i = 0; i < length; i++)
            sum += d_blocks[i].dot(rhs[i]);
        return sum;
    }

    __device__ Scalar norm() const {
        return sqrt(dot(*this));
    }

    // 从 device 端下载到 host
    LongVector<BlockSize> download() const;
};


// 放在类外部！注意：需要在 device 端定义
template<uInt DoFs>
__device__ inline LongVectorDevice<DoFs> operator*(Scalar val, const LongVectorDevice<DoFs>& vec) {
    return vec * val;
}
// template<uInt DoFs>
// __device__ inline LongVectorDevice<DoFs> operator/(Scalar val, const LongVectorDevice<DoFs>& vec) {
//     return vec * val;
// }





extern template class LongVectorDevice<1>;
extern template class LongVectorDevice<2>;
extern template class LongVectorDevice<3>;
extern template class LongVectorDevice<4>;
extern template class LongVectorDevice<5>;
extern template class LongVectorDevice<8>;
extern template class LongVectorDevice<10>;
extern template class LongVectorDevice<12>;
extern template class LongVectorDevice<13>;
extern template class LongVectorDevice<16>;
extern template class LongVectorDevice<20>;
extern template class LongVectorDevice<30>;
extern template class LongVectorDevice<31>;
extern template class LongVectorDevice<34>;
extern template class LongVectorDevice<35>;
extern template class LongVectorDevice<40>;
extern template class LongVectorDevice<50>;
extern template class LongVectorDevice<56>;
extern template class LongVectorDevice<60>;
extern template class LongVectorDevice<64>;
extern template class LongVectorDevice<70>;
extern template class LongVectorDevice<80>;
extern template class LongVectorDevice<84>;
extern template class LongVectorDevice<100>;
extern template class LongVectorDevice<105>;
extern template class LongVectorDevice<112>;
extern template class LongVectorDevice<115>;
extern template class LongVectorDevice<125>;
extern template class LongVectorDevice<140>;
extern template class LongVectorDevice<168>;
extern template class LongVectorDevice<175>;
extern template class LongVectorDevice<188>;
extern template class LongVectorDevice<203>;
extern template class LongVectorDevice<224>;
extern template class LongVectorDevice<252>;
extern template class LongVectorDevice<280>;
extern template class LongVectorDevice<287>;
extern template class LongVectorDevice<308>;
extern template class LongVectorDevice<336>;
extern template class LongVectorDevice<420>;