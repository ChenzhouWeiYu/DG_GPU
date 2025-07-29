#pragma once

// ================= gemm gemv 的 AVX2 实现 =================
namespace avx2 {
    template <uInt M, uInt N, uInt K>
    inline void generic_matrix_mult(const Scalar64* A, const Scalar64* B, Scalar64* C);
    template <uInt N>
    void strassen_matrix_mult(const Scalar64* A, const Scalar64* B, Scalar64* C);
    template <uInt N>
    void strassen_step(const Scalar64* A11, const Scalar64* A12,
                      const Scalar64* A21, const Scalar64* A22,
                      const Scalar64* B11, const Scalar64* B12,
                      const Scalar64* B21, const Scalar64* B22,
                      Scalar64* C11, Scalar64* C12,
                      Scalar64* C21, Scalar64* C22,
                      Scalar64* M1, Scalar64* M2, Scalar64* M3,
                      Scalar64* M4, Scalar64* M5, Scalar64* M6,
                      Scalar64* M7, uInt size);
    template <uInt N>
    inline void matrix_add(const Scalar64* A, const Scalar64* B, Scalar64* C);

    template <uInt N>
    inline void matrix_sub(const Scalar64* A, const Scalar64* B, Scalar64* C);

    template <uInt N>
    inline void matrix_add_sub(const Scalar64* M1, const Scalar64* M2,
                              const Scalar64* M3, const Scalar64* M4,
                              Scalar64* C);

    // 矩阵乘法核心(支持任意尺寸)
    template <uInt M, uInt N, uInt K>
    inline void matrix_mult(const Scalar64* A, const Scalar64* B, Scalar64* C) {
        constexpr uInt BLOCK_SIZE = 64;
        if constexpr (M >= BLOCK_SIZE && N >= BLOCK_SIZE && K >= BLOCK_SIZE) {
            strassen_matrix_mult<M, N, K>(A, B, C);
        } else {
            generic_matrix_mult<M, N, K>(A, B, C);
        }
    }

    // 通用矩阵乘法(带AVX2优化)
    template <uInt M, uInt N, uInt K>
    inline void generic_matrix_mult(const Scalar64* A, const Scalar64* B, Scalar64* C) {
        constexpr uInt AVX2_REG_SIZE = 4; // 每个AVX2寄存器存4个double
        constexpr uInt UNROLL_FACTOR = 2; // 4

        // #pragma omp parallel for collapse(2)
        for (uInt i = 0; i < M; ++i) {
            for (uInt j = 0; j < K; j += UNROLL_FACTOR * AVX2_REG_SIZE) {
                for (uInt k = 0; k < N; ++k) {
                    __m256d a = _mm256_broadcast_sd(&A[i * N + k]);

                    for (uInt jj = 0; jj < UNROLL_FACTOR; ++jj) {
                        const uInt col = j + jj * AVX2_REG_SIZE;
                        if (col >= K) break;

                        __m256d b = _mm256_loadu_pd(&B[k * K + col]);
                        __m256d c = _mm256_loadu_pd(&C[i * K + col]);
                        c = _mm256_fmadd_pd(a, b, c);
                        _mm256_storeu_pd(&C[i * K + col], c);
                    }
                }
            }
        }
    }

    // 矩阵-向量乘法优化
    template <uInt M, uInt N>
    inline void matrix_vector_mult(const Scalar64* A, const Scalar64* x, Scalar64* y) {
        constexpr uInt AVX2_REG_SIZE = 4;
        constexpr uInt REMAIN = N % AVX2_REG_SIZE;

        // #pragma omp parallel for
        for (uInt i = 0; i < M; ++i) {
            __m256d sum = _mm256_setzero_pd();
            uInt j = 0;
            // debug(j);
            // AVX2主体部分
            for (; j + AVX2_REG_SIZE <= N; j += AVX2_REG_SIZE) {
                __m256d a = _mm256_loadu_pd(&A[i * N + j]);
                __m256d b = _mm256_loadu_pd(&x[j]);
                sum = _mm256_fmadd_pd(a, b, sum);
            }
            // debug(j);
            // 处理余数
            Scalar64 partial_sum = 0;
            for (; j < N; ++j) {
                partial_sum += A[i * N + j] * x[j];
            }
            // debug(j);
            // 合并结果
            alignas(32) double temp[4];
            _mm256_store_pd(temp, sum);
            y[i] = temp[0] + temp[1] + temp[2] + temp[3] + partial_sum;
            // debug(j);
        }
    }

    // 内积优化
    inline Scalar64 dot_product(const Scalar64* a, const Scalar64* b, uInt n) {
        constexpr uInt AVX2_REG_SIZE = 4;
        __m256d sum = _mm256_setzero_pd();
        uInt i = 0;

        // AVX2主体部分
        for (; i + AVX2_REG_SIZE <= n; i += AVX2_REG_SIZE) {
            __m256d va = _mm256_loadu_pd(&a[i]);
            __m256d vb = _mm256_loadu_pd(&b[i]);
            sum = _mm256_fmadd_pd(va, vb, sum);
        }

        // 处理余数
        Scalar64 partial_sum = 0;
        for (; i < n; ++i) {
            partial_sum += a[i] * b[i];
        }

        // 合并结果
        alignas(32) double temp[4];
        _mm256_store_pd(temp, sum);
        return temp[0] + temp[1] + temp[2] + temp[3] + partial_sum;
    }

    // ================= Strassen算法实现 =================
    template <uInt N>
    void strassen_matrix_mult(const Scalar64* A, const Scalar64* B, Scalar64* C) {
        constexpr uInt MIN_SIZE = 64;
        if (N <= MIN_SIZE) {
            generic_matrix_mult<N, N, N>(A, B, C);
            return;
        }

        constexpr uInt half = N / 2;
        constexpr uInt quarter = half / 2;
        const Scalar64* A11 = A;
        const Scalar64* A12 = A + half;
        const Scalar64* A21 = A + half * N;
        const Scalar64* A22 = A21 + half;
        
        const Scalar64* B11 = B;
        const Scalar64* B12 = B + half;
        const Scalar64* B21 = B + half * N;
        const Scalar64* B22 = B21 + half;
        
        Scalar64* C11 = C;
        Scalar64* C12 = C + half;
        Scalar64* C21 = C + half * N;
        Scalar64* C22 = C21 + half;

        // 分配临时内存
        std::vector<Scalar64> M1(N * N), M2(N * N), M3(N * N), 
                           M4(N * N), M5(N * N), M6(N * N), M7(N * N);

        // Strassen递归步骤
        avx2::strassen_step<half>(A11, A12, A21, A22, 
                           B11, B12, B21, B22,
                           C11, C12, C21, C22,
                           M1.data(), M2.data(), M3.data(),
                           M4.data(), M5.data(), M6.data(),
                           M7.data(), half);
    }

    template <uInt N>
    void strassen_step(const Scalar64* A11, const Scalar64* A12,
                      const Scalar64* A21, const Scalar64* A22,
                      const Scalar64* B11, const Scalar64* B12,
                      const Scalar64* B21, const Scalar64* B22,
                      Scalar64* C11, Scalar64* C12,
                      Scalar64* C21, Scalar64* C22,
                      Scalar64* M1, Scalar64* M2, Scalar64* M3,
                      Scalar64* M4, Scalar64* M5, Scalar64* M6,
                      Scalar64* M7, uInt size) {
        // 计算中间矩阵M1-M7
        matrix_add<N>(A11, A22, M1);
        matrix_add<N>(B11, B22, M2);
        strassen_matrix_mult<N>(M1, M2, M1); // M1 = (A11+A22)(B11+B22)

        matrix_add<N>(A21, A22, M2);
        strassen_matrix_mult<N>(M2, B11, M2); // M2 = (A21+A22)B11

        matrix_sub<N>(B12, B22, M3);
        strassen_matrix_mult<N>(A11, M3, M3); // M3 = A11(B12-B22)

        matrix_sub<N>(B21, B11, M4);
        strassen_matrix_mult<N>(A22, M4, M4); // M4 = A22(B21-B11)

        matrix_add<N>(A11, A12, M5);
        strassen_matrix_mult<N>(M5, B22, M5); // M5 = (A11+A12)B22

        matrix_sub<N>(A21, A11, M6);
        matrix_add<N>(B11, B12, M7);
        strassen_matrix_mult<N>(M6, M7, M6); // M6 = (A21-A11)(B11+B12)

        matrix_sub<N>(A12, A22, M7);
        matrix_add<N>(B21, B22, M5);
        strassen_matrix_mult<N>(M7, M5, M7); // M7 = (A12-A22)(B21+B22)

        // 组合结果矩阵
        matrix_add_sub<N>(M1, M4, M5, M7, C11); // C11 = M1 + M4 - M5 + M7
        matrix_add<N>(M3, M5, C12);             // C12 = M3 + M5
        matrix_add<N>(M2, M4, C21);             // C21 = M2 + M4
        matrix_add_sub<N>(M1, M3, M2, M6, C22); // C22 = M1 - M2 + M3 + M6
    }

    // 辅助函数
    template <uInt N>
    inline void matrix_add(const Scalar64* A, const Scalar64* B, Scalar64* C) {
        #pragma omp simd
        for (uInt i = 0; i < N*N; ++i)
            C[i] = A[i] + B[i];
    }

    template <uInt N>
    inline void matrix_sub(const Scalar64* A, const Scalar64* B, Scalar64* C) {
        #pragma omp simd
        for (uInt i = 0; i < N*N; ++i)
            C[i] = A[i] - B[i];
    }

    template <uInt N>
    inline void matrix_add_sub(const Scalar64* M1, const Scalar64* M2,
                              const Scalar64* M3, const Scalar64* M4,
                              Scalar64* C) {
        #pragma omp simd
        for (uInt i = 0; i < N*N; ++i)
            C[i] = M1[i] + M2[i] - M3[i] + M4[i];
    }
}