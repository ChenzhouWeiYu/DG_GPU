#include "Type.h"
#include <array>
#include <immintrin.h>
#include <type_traits>

// using Scalar = double;
constexpr size_t SIMD_WIDTH = 4; // AVX256每个向量包含4个double
constexpr size_t MAX_UNROLL_N = 16;






#define DEFINE_OPERATOR(op, Type) \
template<size_t N> \
HostDevice std::array<Type, N> operator op(const std::array<Type, N>& a, const std::array<Type, N>& b) { \
    std::array<Type, N> res; \
    size_t i=0;\
    if constexpr (N >= SIMD_WIDTH)\
    for(; i<N-(SIMD_WIDTH-1); i+=SIMD_WIDTH) { \
            res[i+0] = a[i+0] op b[i+0]; \
            res[i+1] = a[i+1] op b[i+1]; \
            res[i+2] = a[i+2] op b[i+2]; \
            res[i+3] = a[i+3] op b[i+3]; \
    } \
    if constexpr (N % SIMD_WIDTH != 0) { \
        for(; i<N; ++i) \
            res[i] = a[i] op b[i]; \
    } \
    return res; \
} \
template<size_t N> \
HostDevice std::array<Type, N> operator op(const std::array<Type, N>& a, Type b) { \
    std::array<Type, N> res; \
    size_t i=0;\
    if constexpr (N >= SIMD_WIDTH)\
    for(; i<N-(SIMD_WIDTH-1); i+=SIMD_WIDTH) { \
            res[i+0] = a[i+0] op b; \
            res[i+1] = a[i+1] op b; \
            res[i+2] = a[i+2] op b; \
            res[i+3] = a[i+3] op b; \
    } \
    if constexpr (N % SIMD_WIDTH != 0) { \
        for(; i<N; ++i) \
            res[i] = a[i] op b; \
    } \
    return res; \
} \
template<size_t N> \
HostDevice std::array<Type, N> operator op(Type a, const std::array<Type, N>& b) { \
    std::array<Type, N> res; \
    size_t i=0;\
    if constexpr (N >= SIMD_WIDTH)\
    for(; i<N-(SIMD_WIDTH-1); i+=SIMD_WIDTH) { \
            res[i+0] = a op b[i+0]; \
            res[i+1] = a op b[i+1]; \
            res[i+2] = a op b[i+2]; \
            res[i+3] = a op b[i+3]; \
    } \
    if constexpr (N % SIMD_WIDTH != 0) { \
        for(; i<N; ++i) \
            res[i] = a op b[i]; \
    } \
    return res; \
} \
template<size_t N> \
HostDevice std::array<Type, N>& operator op##=(std::array<Type, N>& a, const std::array<Type, N>& b) { \
    size_t i=0;\
    if constexpr (N >= SIMD_WIDTH)\
    for(; i<N-(SIMD_WIDTH-1); i+=SIMD_WIDTH) { \
            a[i+0] op##= b[i+0]; \
            a[i+1] op##= b[i+1]; \
            a[i+2] op##= b[i+2]; \
            a[i+3] op##= b[i+3]; \
    } \
    if constexpr (N % SIMD_WIDTH != 0) { \
        for(; i<N; ++i) \
            a[i] op##= b[i]; \
    } \
    return a; \
}\
template<size_t N> \
HostDevice std::array<Type, N>& operator op##=(std::array<Type, N>& a, const Type b) { \
    size_t i=0;\
    if constexpr (N >= SIMD_WIDTH)\
    for(; i<N-(SIMD_WIDTH-1); i+=SIMD_WIDTH) { \
            a[i+0] op##= b; \
            a[i+1] op##= b; \
            a[i+2] op##= b; \
            a[i+3] op##= b; \
    } \
    if constexpr (N % SIMD_WIDTH != 0) { \
        for(; i<N; ++i) \
            a[i] op##= b; \
    } \
    return a; \
}

DEFINE_OPERATOR(+, double)
DEFINE_OPERATOR(-, double)
DEFINE_OPERATOR(*, double)
DEFINE_OPERATOR(/, double)
DEFINE_OPERATOR(+, long double)
DEFINE_OPERATOR(-, long double)
DEFINE_OPERATOR(*, long double)
DEFINE_OPERATOR(/, long double)



// 点积
inline Scalar hsum_avx(__m256d v) {
    __m128d low = _mm256_castpd256_pd128(v);
    __m128d high = _mm256_extractf128_pd(v, 1);
    low = _mm_add_pd(low, high);
    __m128d shuf = _mm_shuffle_pd(low, low, 0x1);
    __m128d sum = _mm_add_pd(low, shuf);
    return _mm_cvtsd_f64(sum);
}
template<size_t N>
HostDevice inline Scalar vec_dot(const std::array<Scalar, N>& a, const std::array<Scalar, N>& b) {
    Scalar sum = 0;
    // if constexpr (N >= SIMD_WIDTH && std::is_same_v<Scalar,double> && use_AVX2) {
    //     __m256d sumv = _mm256_setzero_pd();
    //     for(size_t i=0; i<N; i+=SIMD_WIDTH) {
    //         auto av = _mm256_loadu_pd(a.data()+i);
    //         auto bv = _mm256_loadu_pd(b.data()+i);
    //         sumv = _mm256_fmadd_pd(av, bv, sumv);
    //     }
    //     sum = hsum_avx(sumv);
    //     if constexpr (N % SIMD_WIDTH != 0) {
    //         for(size_t i=N - N%SIMD_WIDTH; i<N; ++i)
    //             sum += a[i] * b[i];
    //     }
    // } else 
    {
        for(size_t i=0; i<N; ++i)
            sum += a[i] * b[i];
    }
    return sum;
}

// 外积
template<size_t N, size_t M>
HostDevice inline auto vec_outer(const std::array<Scalar, N>& a, const std::array<Scalar, M>& b) {
    std::array<std::array<Scalar, M>, N> res;
    for(size_t i=0; i<N; ++i)
        for(size_t j=0; j<M; ++j)
            res[i][j] = a[i] * b[j];
    return res;
}

// 三维叉乘
HostDevice inline auto vec_cross(const std::array<Scalar, 3>& a, const std::array<Scalar, 3>& b) {
    return std::array<Scalar, 3>{
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    };
}


template<size_t N>
HostDevice inline Scalar vec_length(const std::array<Scalar, N>& p1) {
    return std::sqrt(vec_dot(p1,p1));
}

template<size_t N>
HostDevice inline Scalar distance(const std::array<Scalar, N>& p1,const std::array<Scalar, N>& p2) {
    return vec_length(p2-p1);
}

template<size_t N>
HostDevice inline auto vec_unit(const std::array<Scalar, N>& p1) {
    return p1/std::sqrt(vec_dot(p1,p1));
}