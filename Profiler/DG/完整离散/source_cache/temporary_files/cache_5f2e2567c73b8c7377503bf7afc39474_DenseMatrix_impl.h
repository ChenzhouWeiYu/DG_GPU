#pragma once
#include "base/type.h"
#include "matrix/dense_matrix.h"


// // LU分解: 无主元，返回一个新矩阵，内部存储 L 和 U
// template<uInt M, uInt N>
// template<uInt MM, uInt NN,typename>
// HostDevice inline DenseMatrix<M,N> DenseMatrix<M,N>::lu() const {
//     DenseMatrix<M,N> LU = *this;
//     for (uInt k = 0; k < N; ++k) {
//         for (uInt i = k+1; i < N; ++i) {
//             LU(i,k) /= LU(k,k);
//             for (uInt j = k+1; j < N; ++j) {
//                 LU(i,j) -= LU(i,k) * LU(k,j);
//             }
//         }
//     }
//     return LU;
// }

// // 利用LU求解方程 Ax = b
// template<uInt M, uInt N>
// template<uInt MM, uInt NN,typename>
// HostDevice inline DenseMatrix<M,1> DenseMatrix<M,N>::solve
// (const DenseMatrix<M,1>& b, const DenseMatrix<M,N>& LU) const {
//     DenseMatrix<M,1> y, x;

//     // 前向代入 Ly = b
//     for (uInt i = 0; i < N; ++i) {
//         y(i,0) = b(i,0);
//         for (uInt j = 0; j < i; ++j) {
//             y(i,0) -= LU(i,j) * y(j,0);
//         }
//     }

//     // 后向代入 Ux = y
//     for (int i = N-1; i >= 0; --i) {
//         x(i,0) = y(i,0);
//         for (uInt j = i+1; j < N; ++j) {
//             x(i,0) -= LU(i,j) * x(j,0);
//         }
//         x(i,0) /= LU(i,i);
//     }
//     return x;
// }

// // 自动进行LU分解后求解
// template<uInt M, uInt N>
// template<uInt MM, uInt NN,typename>
// HostDevice inline DenseMatrix<M,1> DenseMatrix<M,N>::solve(const DenseMatrix<M,1>& b) const {
//     DenseMatrix<M,N> LU = lu();
//     return solve(b, LU);
// }

// // 计算逆矩阵: 自动LU分解
// template<uInt M, uInt N>
// template<uInt MM, uInt NN,typename>
// HostDevice inline DenseMatrix<M,N> DenseMatrix<M,N>::inverse() const {
//     DenseMatrix<M,N> LU = lu();
//     return inverse(LU);
// }

// // 已有LU分解，计算逆矩阵
// template<uInt M, uInt N>
// template<uInt MM, uInt NN,typename>
// HostDevice inline DenseMatrix<M,N> DenseMatrix<M,N>::inverse(const DenseMatrix<M,N>& LU) const {
//     DenseMatrix<M,N> inv;
//     DenseMatrix<M,1> e, col;
//     for (uInt i = 0; i < N; ++i) {
//         // 生成单位向量 e_i
//         for (uInt j = 0; j < N; ++j) e(j,0) = (i == j ? 1 : 0);
//         col = solve(e, LU);
//         // 把 col 写入逆矩阵第 i 列
//         for (uInt j = 0; j < N; ++j) {
//             inv(j,i) = col(j,0);
//         }
//     }
//     return inv;
// }

template<>      // 对类模板
template<>      // 对成员函数模板
HostDevice inline DenseMatrix<3,3> DenseMatrix<3,3>::inverse() const {
    DenseMatrix<3,3> ret;
    const DenseMatrix<3,3>& A = *this;
    Scalar det = 0;
    ret(0,0) = A(1,1)*A(2,2) - A(1,2)*A(2,1);
    ret(1,0) = A(1,2)*A(2,0) - A(1,0)*A(2,2);
    ret(2,0) = A(1,0)*A(2,1) - A(1,1)*A(2,0);
    ret(0,1) = A(2,1)*A(0,2) - A(2,2)*A(0,1);
    ret(1,1) = A(2,2)*A(0,0) - A(2,0)*A(0,2);
    ret(2,1) = A(2,0)*A(0,1) - A(2,1)*A(0,0);
    ret(0,2) = A(0,1)*A(1,2) - A(0,2)*A(1,1);
    ret(1,2) = A(0,2)*A(1,0) - A(0,0)*A(1,2);
    ret(2,2) = A(0,0)*A(1,1) - A(0,1)*A(1,0);
    det += A(0,0)*ret(0,0);
    det += A(1,0)*ret(0,1);
    det += A(2,0)*ret(0,2);
    ret /= det;
    return ret;
}





// 矩阵乘法运算符
// template<uInt M, uInt N>
// template <uInt K>
// HostDevice DenseMatrix<M, K> DenseMatrix<M,N>::multiply(const DenseMatrix<N, K>& rhs) const {
//     DenseMatrix<M, K> result;

//     if constexpr (std::is_same_v<Scalar,double> && use_AVX2) {
//         if constexpr (M == N && N == K && (M & (M - 1)) == 0) {
//             // 仅当尺寸为2的幂时使用Strassen
//             avx2::strassen_matrix_mult<M>(this->data_ptr(), rhs.data_ptr(), result.data_ptr());
//         } else {
//             avx2::matrix_mult<M, N, K>(this->data_ptr(), rhs.data_ptr(), result.data_ptr());
//         }
//     }
//     else{
//         for (uInt k = 0; k < N; k++) {
//             for (uInt i = 0; i < M; i++) {
//                 for (uInt j = 0; j < K; j++) {
//                     result(i,j) += (*this)(i,k) * rhs(k,j);
//                 }
//             }
//         }
//     }
//     return result;
// }
// // 矩阵-向量乘法
// template<uInt M, uInt N>
// HostDevice DenseMatrix<M, 1> DenseMatrix<M,N>::multiply(const DenseMatrix<N, 1>& rhs) const {
//     DenseMatrix<M, 1> result;
//     if constexpr (std::is_same_v<Scalar,double> && use_AVX2) {
//         avx2::matrix_vector_mult<M, N>(this->data_ptr(), rhs.data_ptr(), result.data_ptr());
//     }
//     else{
//         for (uInt i = 0; i < M; i++) {
//             for (uInt k = 0; k < N; k++) {
//                 result(i,0) += (*this)(i,k) * rhs(k,0);
//             }
//         }
//     }
//     return result;
// }
// template<uInt M, uInt N>
// HostDevice DenseMatrix<M, 1> DenseMatrix<M,N>::multiply(const std::array<Scalar,N>& rhs) const {
//     DenseMatrix<M, 1> result;
//     if constexpr (std::is_same_v<Scalar,double> && use_AVX2 && N>=4) {
//         avx2::matrix_vector_mult<M, N>(this->data_ptr(), rhs.data(), result.data_ptr());
//     }
//     else{
//         for (uInt i = 0; i < M; i++) {
//             for (uInt k = 0; k < N; k++) {
//                 result(i,0) += (*this)(i,k) * rhs[k];
//             }
//         }
//     }
//     return result;
// }