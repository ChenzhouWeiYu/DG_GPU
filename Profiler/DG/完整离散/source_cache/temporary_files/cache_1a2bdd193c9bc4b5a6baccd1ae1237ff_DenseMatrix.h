#pragma once
#include "base/type.h"
#include "AVX2.h"

template <uInt M, uInt N, uInt SubM, uInt SubN>
class MatrixView;

template <uInt M, uInt N>
class DenseMatrix {
private:
    static constexpr uInt Rows = M;
    static constexpr uInt Cols = N;
    static constexpr uInt Size = M*N;
    
    alignas(32) std::array<Scalar, Size> data; // 行优先存储

public:
    // 访问元素
    HostDevice Scalar& operator()(uInt row, uInt col) { return data[row*N + col]; }
    HostDevice const Scalar& operator()(uInt row, uInt col) const { return data[row*N + col]; }

    HostDevice Scalar& operator[](uInt idx) { return data[idx]; }
    HostDevice const Scalar& operator[](uInt idx) const { return data[idx]; }

    HostDevice Scalar* data_ptr() { return data.data(); }
    HostDevice const Scalar* data_ptr() const { return data.data(); }

    // 获取矩阵的行数和列数
    HostDevice uInt size() const{return Size;}
    HostDevice uInt rows() const{return Rows;}
    HostDevice uInt cols() const{return Cols;}

    // 构造函数
    HostDevice DenseMatrix(){
        for(auto& v : data) v = 0.0;
    };
    HostDevice DenseMatrix(std::initializer_list<Scalar> init) {
        for(uInt i=0; i<Size; ++i) data[i] = 0.0;
        uInt i = 0;
        for(auto it = init.begin(); it != init.end() && i < Size; ++it, ++i)
            data[i] = *it;
    }

    HostDevice DenseMatrix(const std::array<Scalar, Size>& init) {
        for(uInt i=0; i<Size; ++i) data[i] = 0.0;
        uInt i = 0;
        for(auto it = init.begin(); it != init.end() && i < Size; ++it, ++i)
            data[i] = *it;
    }
    HostDevice DenseMatrix(const std::array<std::array<Scalar,N>, M>& init) {
        for(uInt i = 0; i < M; i++)
        for(uInt j = 0; j < N; j++)
        data[i*N+j] = init[i][j];
    }

    HostDevice DenseMatrix& operator=(const DenseMatrix& rhs){
        for(uInt i=0; i<M*N; ++i)
            data[i] = rhs[i];
        return *this;
    }
    HostDevice DenseMatrix(const DenseMatrix& rhs){
        for(uInt i=0; i<M*N; ++i)
            data[i] = rhs[i];
    }

    // 矩阵视图
    template <uInt SubM, uInt SubN>
    HostDevice MatrixView<M,N,SubM,SubN> View(uInt sr, uInt sc) {
        return MatrixView<M,N,SubM,SubN>(*this, sr, sc);
    }

    // template <uInt SubM, uInt SubN>
    // HostDevice DenseMatrix<SubM,SubN> SubMat(uInt sr, uInt sc) {
    //     return MatrixView<M,N,SubM,SubN>(*this, sr, sc);
    // }
    
    template <uInt SubM, uInt SubN>
    HostDevice DenseMatrix<SubM,SubN> SubMat(uInt sr, uInt sc) const {
        DenseMatrix<SubM,SubN> sub;
        for (uInt i = 0; i < SubM; i++)
            for (uInt j = 0; j < SubN; j++)
                sub(i,j) = (*this)(sr + i, sc + j);
        return sub;
    }

    // 赋值操作  
    HostDevice DenseMatrix& operator=(Scalar val) {
        for(auto& v : data) v = val;
        return *this;
    }
    HostDevice operator std::array<Scalar, Size>() const {
        return data; // data 是 std::array，可以直接返回
    }

    // 转置、迹、LU分解等操作
    HostDevice DenseMatrix<N,M> transpose() const{
        DenseMatrix<N,M> result;
        for(uInt i=0;i<M;i++) for(uInt j=0;j<N;j++) result(j,i)=(*this)(i,j);
        return result;
    }
    HostDevice Scalar trace() const{
        if constexpr (M==N){
            Scalar result = 0.0;
            for(uInt i=0;i<M;i++) result+=(*this)(i,i);
            return result;
        }
        else return 0.0;
    }

    // LU分解: 无主元，返回一个新矩阵，内部存储 L 和 U
    template<uInt MM = M, uInt NN = N,typename = std::enable_if_t<MM == NN>>
    HostDevice DenseMatrix<M,N> lu() const {
        DenseMatrix<M,N> LU = *this;
        for (uInt k = 0; k < N; ++k) {
            for (uInt i = k+1; i < N; ++i) {
                LU(i,k) /= LU(k,k);
                for (uInt j = k+1; j < N; ++j) {
                    LU(i,j) -= LU(i,k) * LU(k,j);
                }
            }
        }
        return LU;
    }

    // 自动进行LU分解后求解
    template<uInt MM = M, uInt NN = N,typename = std::enable_if_t<MM == NN>>
    HostDevice DenseMatrix<M,1> solve(const DenseMatrix<M,1>& b) const {
        DenseMatrix<M,N> LU = lu();
        return solve(b, LU);
    }
    // 利用LU求解方程 Ax = b
    template<uInt MM = M, uInt NN = N,typename = std::enable_if_t<MM == NN>>
    HostDevice DenseMatrix<M,1> solve(const DenseMatrix<M,1>& b, const DenseMatrix<M,N>& LU) const{
        DenseMatrix<M,1> y, x;

        // 前向代入 Ly = b
        for (uInt i = 0; i < N; ++i) {
            y(i,0) = b(i,0);
            for (uInt j = 0; j < i; ++j) {
                y(i,0) -= LU(i,j) * y(j,0);
            }
        }

        // 后向代入 Ux = y
        for (int i = N-1; i >= 0; --i) {
            x(i,0) = y(i,0);
            for (uInt j = i+1; j < N; ++j) {
                x(i,0) -= LU(i,j) * x(j,0);
            }
            x(i,0) /= LU(i,i);
        }
        return x;
    }
    // 计算逆矩阵: 自动LU分解
    template<uInt MM = M, uInt NN = N,typename = std::enable_if_t<MM == NN>>
    HostDevice DenseMatrix<M,N> inverse() const{
        DenseMatrix<M,N> LU = lu();
        return inverse(LU);
    }
    // 已有LU分解，计算逆矩阵
    template<uInt MM = M, uInt NN = N,typename = std::enable_if_t<MM == NN>>
    HostDevice DenseMatrix<M,N> inverse(const DenseMatrix<M,N>& LU) const{
        DenseMatrix<M,N> inv;
        DenseMatrix<M,1> e, col;
        for (uInt i = 0; i < N; ++i) {
            // 生成单位向量 e_i
            for (uInt j = 0; j < N; ++j) e(j,0) = (i == j ? 1 : 0);
            col = solve(e, LU);
            // 把 col 写入逆矩阵第 i 列
            for (uInt j = 0; j < N; ++j) {
                inv(j,i) = col(j,0);
            }
        }
        return inv;
    }


    // 逐元素运算 
    // 宏实现 +-*/ 的快速替换 
    #define ELEMENT_WISE_OP(op)\
    HostDevice DenseMatrix& operator op##=(const DenseMatrix& rhs) {\
        for(uInt i=0; i<M*N; ++i) data[i] op##= rhs.data[i];\
        return *this;\
    }\
    HostDevice DenseMatrix operator op(const DenseMatrix& rhs) const {\
        DenseMatrix result(*this);\
        return result op##= rhs;\
    }\
    HostDevice DenseMatrix& operator op##=(Scalar val) {\
        for(auto& d : data) d op##= val;\
        return *this;\
    }\
    HostDevice friend DenseMatrix operator op(Scalar val, const DenseMatrix& rhs) { \
        DenseMatrix res(rhs); \
        for (auto& v : res.data) v = val op v; \
        return res; \
    }\
    HostDevice DenseMatrix operator op(Scalar val) const {\
        DenseMatrix result(*this);\
        return result op##= val;\
    }
    
    ELEMENT_WISE_OP(+)
    ELEMENT_WISE_OP(-)
    ELEMENT_WISE_OP(*)
    ELEMENT_WISE_OP(/)
    #undef ELEMENT_WISE_OP
    HostDevice DenseMatrix operator -() const {
        return Zeros() - (*this);
    }
    

    // 矩阵乘法运算符
    // template <uInt K>
    // HostDevice DenseMatrix<M, K> multiply(const DenseMatrix<N, K>& rhs) const ;
    // HostDevice DenseMatrix<M, 1> multiply(const DenseMatrix<N, 1>& rhs) const ;
    // HostDevice DenseMatrix<M, 1> multiply(const std::array<Scalar,N>& rhs) const;
    template <uInt K>
    HostDevice DenseMatrix<M, K> multiply(const DenseMatrix<N, K>& rhs) const {
        DenseMatrix<M, K> result;
        for (uInt k = 0; k < N; k++) {
            for (uInt i = 0; i < M; i++) {
                for (uInt j = 0; j < K; j++) {
                    result(i,j) += (*this)(i,k) * rhs(k,j);
                }
            }
        }
        return result;
    }
    // 矩阵-向量乘法
    HostDevice DenseMatrix<M, 1> multiply(const DenseMatrix<N, 1>& rhs) const {
        DenseMatrix<M, 1> result;
        for (uInt i = 0; i < M; i++) {
            for (uInt k = 0; k < N; k++) {
                result(i,0) += (*this)(i,k) * rhs(k,0);
            }
        }
        return result;
    }
    HostDevice DenseMatrix<M, 1> multiply(const std::array<Scalar,N>& rhs) const {
        DenseMatrix<M, 1> result;
        for (uInt i = 0; i < M; i++) {
            for (uInt k = 0; k < N; k++) {
                result(i,0) += (*this)(i,k) * rhs[k];
            }
        }
        return result;
    }

    // 向量级运算
    template<int U = N,typename = std::enable_if_t<U ==1>>
    HostDevice Scalar dot(const DenseMatrix<M, 1>& rhs) const { return vec_dot(data, rhs.data);}
    HostDevice Scalar length() const { return vec_length(data);}
    HostDevice Scalar norm() const {return vec_length(data);}

    // 流输出
    friend std::ostream& operator<<(std::ostream& os, const DenseMatrix& mat) {
        for(uInt i=0; i<M; ++i) {
            for(uInt j=0; j<N; ++j)
                os << mat(i,j) << " ";
            os << "\n";
        }
        return os;
    }

private:
    

public:
    // ================= 数学函数 =================
    #define ELEMENT_WISE_FUNC(FUNC)\
    HostDevice friend DenseMatrix FUNC(const DenseMatrix& mat) {\
        DenseMatrix result;\
        for(uInt i=0; i<M*N; ++i)\
            result.data[i] = std::FUNC(mat.data[i]);\
        return result;\
    }
    ELEMENT_WISE_FUNC(exp)
    ELEMENT_WISE_FUNC(log)
    ELEMENT_WISE_FUNC(sin)
    ELEMENT_WISE_FUNC(cos)
    ELEMENT_WISE_FUNC(tan)
    // ELEMENT_WISE_FUNC(max)
    // ELEMENT_WISE_FUNC(min)
    ELEMENT_WISE_FUNC(abs)
    ELEMENT_WISE_FUNC(tanh)
    #undef ELEMENT_WISE_FUNC

    HostDevice friend DenseMatrix pow(const DenseMatrix& mat, Scalar beta) {
        DenseMatrix result;
        for(uInt i=0; i<M*N; ++i)
            result.data[i] = std::pow(mat.data[i], beta);
        return result;
    }

    HostDevice friend Scalar max(const DenseMatrix& mat) {
        Scalar m = mat.data[0];
        for(const auto& b : mat.data)
            m = m > b? m : b;
        return m;
    }
    
    HostDevice friend Scalar min(const DenseMatrix& mat) {
        Scalar m = mat.data[0];
        for(const auto& b : mat.data)
            m = m < b? m : b;
        return m;
    }
    // 类似实现log, sqrt, sin等...

    // ================= 特殊矩阵生成 =================
    template<uInt MM = M, uInt NN = N,typename = std::enable_if_t<MM == NN>>
    HostDevice static DenseMatrix Identity() {
        static_assert(M == N, "Identity matrix must be square");
        DenseMatrix mat;
        for(uInt i=0; i<M; ++i) mat(i,i) = 1.0;
        return mat;
    }
    template<uInt MM = M, uInt NN = N,typename = std::enable_if_t<MM == NN>>
    HostDevice static DenseMatrix Diag(DenseMatrix<N,1> diag) {
        static_assert(M == N, "Identity matrix must be square");
        DenseMatrix mat;
        for(uInt i=0; i<M; ++i) mat(i,i) = diag(i,0);
        return mat;
    }
    template<uInt MM = M, uInt NN = N,typename = std::enable_if_t<MM == NN>>
    HostDevice static DenseMatrix Diag(std::array<Scalar,N> diag) {
        static_assert(M == N, "Identity matrix must be square");
        DenseMatrix mat;
        for(uInt i=0; i<M; ++i) mat(i,i) = diag[i];
        return mat;
    }
    HostDevice static DenseMatrix Ones() {
        DenseMatrix mat;
        for(uInt i=0; i<Size; ++i) mat[i] = 1.0;
        return mat;
    }
    HostDevice static DenseMatrix Zeros() {
        DenseMatrix mat;
        for(uInt i=0; i<Size; ++i) mat[i] = 0.0;
        return mat;
    }
    HostDevice static DenseMatrix Random() {
        DenseMatrix mat;
        for(uInt i=0; i<Size; ++i) mat[i] = 1.0;
        return mat;
    }

};


#include "DenseMatrix_impl.h"







// ================= 矩阵视图（零拷贝子矩阵） =================
template <uInt M, uInt N, uInt SubM, uInt SubN>
class MatrixView {
    DenseMatrix<M,N>& parent;
    uInt start_row;
    uInt start_col;

public:
    HostDevice MatrixView(DenseMatrix<M,N>& mat, uInt sr, uInt sc) 
        : parent(mat), start_row(sr), start_col(sc) 
    {
        // if(sr + SubM > M || sc + SubN > N)
        //     throw std::out_of_range("Submatrix out of bounds");
    }

    HostDevice MatrixView(const DenseMatrix<M,N>& mat, uInt sr, uInt sc) 
        : parent(mat), start_row(sr), start_col(sc) 
    {
        // if(sr + SubM > M || sc + SubN > N)
        //     throw std::out_of_range("Submatrix out of bounds");
    }

    HostDevice Scalar& operator()(uInt i, uInt j) {
        return parent(start_row + i, start_col + j);
    }

    HostDevice Scalar& operator[](uInt idx) {
        return (*this)(idx/SubN, idx%SubN);
    }

    HostDevice const Scalar& operator()(uInt i, uInt j) const {
        return parent(start_row + i, start_col + j);
    }

    HostDevice const Scalar& operator[](uInt idx) const {
        return (*this)(idx/SubN, idx%SubN);
    }

    template <uInt K>
    HostDevice auto operator*(const MatrixView<N,K,SubN,K>& rhs) {
        DenseMatrix<SubM,K> result;
        // 优化后的子矩阵乘法...
        return result;  
    }
        // ================= 逐元素运算 =================
    // 宏实现 +-*/ 的快速替换 
    #define ELEMENT_WISE_OP(op)\
    HostDevice MatrixView& operator op##=(const DenseMatrix<SubM,SubN>& rhs) {\
        for(uInt i=0; i<SubM; ++i){\
            for(uInt j=0; j<SubN; ++j){\
                parent(start_row + i, start_col + j) op##= rhs(i,j);\
            }\
        } \
        return *this;\
    }\
    HostDevice DenseMatrix<SubM,SubN> operator op(const DenseMatrix<SubM,SubN>& rhs) {\
        DenseMatrix<SubM,SubN> result;\
        for(uInt i=0; i<SubM; ++i){\
            for(uInt j=0; j<SubN; ++j){\
                result(i,j) = parent(start_row + i, start_col + j) op rhs(i,j);\
            }\
        } \
        return result;\
    }\
    HostDevice DenseMatrix<SubM,SubN> operator op(const MatrixView& rhs) {\
        DenseMatrix<SubM,SubN> result;\
        for(uInt i=0; i<SubM; ++i){\
            for(uInt j=0; j<SubN; ++j){\
                result(i,j) = parent(start_row + i, start_col + j) op rhs.parent(start_row + i, start_col + j);\
            }\
        } \
        return result;\
    }\
    HostDevice const DenseMatrix<SubM,SubN>& operator op(const DenseMatrix<SubM,SubN>& rhs) const {\
        DenseMatrix<SubM,SubN> result;\
        for(uInt i=0; i<SubM; ++i){\
            for(uInt j=0; j<SubN; ++j){\
                result(i,j) = parent(start_row + i, start_col + j) op rhs(i,j);\
            }\
        } \
        return result;\
    }\
    HostDevice const DenseMatrix<SubM,SubN>& operator op(const MatrixView& rhs) const {\
        DenseMatrix<SubM,SubN> result;\
        for(uInt i=0; i<SubM; ++i){\
            for(uInt j=0; j<SubN; ++j){\
                result(i,j) = parent(start_row + i, start_col + j) op rhs.parent(start_row + i, start_col + j);\
            }\
        } \
        return result;\
    }

    HostDevice MatrixView& operator =(const DenseMatrix<SubM,SubN>& rhs) {
        for(uInt i=0; i<SubM; ++i){
            for(uInt j=0; j<SubN; ++j){
                parent(start_row + i, start_col + j) = rhs(i,j);
            }
        } 
        return *this;
    }

    HostDevice operator DenseMatrix<SubM,SubN>() const {
        DenseMatrix<SubM,SubN> result;
        for(uInt i=0; i<SubM; ++i){
            for(uInt j=0; j<SubN; ++j){
                result(i,j) = parent(start_row + i, start_col + j);
            }
        } 
        return result;
    }
    HostDevice static DenseMatrix<SubM,SubN> Sub(const DenseMatrix<M,N>& mat, uInt sr, uInt sc) {
        DenseMatrix<SubM,SubN> result;
        for(uInt i=0; i<SubM; ++i){
            for(uInt j=0; j<SubN; ++j){
                result(i,j) = mat(sr + i, sc + j);
            }
        } 
        return result;
    }
    
    ELEMENT_WISE_OP(+)
    ELEMENT_WISE_OP(-)
    ELEMENT_WISE_OP(*)
    ELEMENT_WISE_OP(/)
    #undef ELEMENT_WISE_OP
};

// ================= 使用示例 =================
/*
DenseMatrix<64,64> a = DenseMatrix<64,64>::Identity();
DenseMatrix<64,64> b = DenseMatrix<64,64>::Ones();

// 自动选择Strassen算法
auto c = a * b; 

// 子矩阵操作
MatrixView<64,64,32,32> sub_a(a, 0, 0);
MatrixView<64,64,32,32> sub_b(b, 16, 16);
auto sub_c = sub_a * sub_b;

// 并行向量运算
LongVector<64> v1, v2;
v1 += v2 * 3.14; // 并行执行
Scalar ip = v1.dot(v2); // SIMD+OpenMP优化
*/








// ================ 模板实例化 =================
#include "DenseMatrix_inst.h"