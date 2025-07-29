#pragma once
#include "matrix/dense_matrix.h"

template <uInt BlockSize>
class LongVectorDevice ;


// 长向量包装器
template <uInt BlockSize>
class LongVector {
// private:
public:
    std::vector<DenseMatrix<BlockSize,1>> blocks;

public:
    LongVector() = default;
    LongVector(uInt length){blocks.resize(length);for(auto&b:blocks)b=0.0;};

    // 访问接口
    DenseMatrix<BlockSize,1>& operator[](uInt i) { return blocks[i]; }
    const DenseMatrix<BlockSize,1>& operator[](uInt i) const { return blocks[i]; }
    uInt size() const{return blocks.size();}
    void resize(uInt size) {blocks.resize(size);}
    
    // 上传到 GPU
    LongVectorDevice<BlockSize> to_device() const;

    // 四则运算
    #define ELEMENT_WISE_OP(op)\
    LongVector& operator op##=(const LongVector& rhs) {\
        for(uInt i=0; i<blocks.size(); ++i)\
            blocks[i] op##= rhs.blocks[i];\
        return *this;\
    }\
    LongVector operator op(const LongVector& rhs) const {\
        LongVector res(*this);\
        return res op##= rhs;\
    }\
    LongVector& operator op##=(Scalar val) {\
        for(auto& d : blocks) d op##= val;\
        return *this;\
    }\
    friend LongVector operator op(Scalar val, LongVector& lhs) {\
        for(uInt i=0; i<lhs.blocks.size(); ++i)\
            lhs.blocks[i] = val op lhs.blocks[i];\
        return lhs;\
    }\
    LongVector operator op(Scalar val) const {\
        LongVector result(*this);\
        return result op##= val;\
    }

    ELEMENT_WISE_OP(+)
    ELEMENT_WISE_OP(-)
    ELEMENT_WISE_OP(*)
    ELEMENT_WISE_OP(/)
    #undef ELEMENT_WISE_OP

    // 内积
    Scalar dot(const LongVector& rhs) const {
        Scalar sum = 0;
        for(uInt i=0; i<blocks.size(); ++i) {
            for(uInt j=0; j<BlockSize; ++j)
                sum += blocks[i](j,0) * rhs.blocks[i](j,0);
        }
        return sum;
    }

    Scalar norm() const {
        return std::sqrt((*this).dot(*this));
    }

    // 逐元素数学函数
    #define ELEMENT_WISE_FUNC(FUNC)\
    friend LongVector FUNC(const LongVector& lv) {\
        LongVector res;\
        res.blocks.reserve(lv.blocks.size());\
        for(const auto& b : lv.blocks)\
            res.blocks.push_back(FUNC(b));\
        return res;\
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

    friend LongVector pow(const LongVector& lv, Scalar beta) {
        LongVector res;
        res.blocks.reserve(lv.blocks.size());
        for(const auto& b : lv.blocks)
            res.blocks.push_back(pow(b,beta));
        return res;
    }

    friend Scalar max(const LongVector& lv) {
        Scalar m;
        for(const auto& b : lv.blocks)
            m = m > max(b)? m : max(b);
        return m;
    }
    
    friend Scalar min(const LongVector& lv) {
        Scalar m;
        for(const auto& b : lv.blocks)
            m = m < min(b)? m : min(b);
        return m;
    }

    // 流输出
    friend std::ostream& operator<<(std::ostream& os, const LongVector& vec) {
        for(const auto& b:vec.blocks) {
            os << "(";
            for(uInt i = 0;i < b.size(); i++)
                os << b(i,0) << "  ";
            os << ")\n" ;
        }
        return os;
    }

};





extern template class LongVector<1>;
extern template class LongVector<2>;
extern template class LongVector<3>;
extern template class LongVector<4>;
extern template class LongVector<5>;
extern template class LongVector<8>;
extern template class LongVector<10>;
extern template class LongVector<12>;
extern template class LongVector<13>;
extern template class LongVector<16>;
extern template class LongVector<20>;
extern template class LongVector<30>;
extern template class LongVector<31>;
extern template class LongVector<34>;
extern template class LongVector<35>;
extern template class LongVector<40>;
extern template class LongVector<50>;
extern template class LongVector<56>;
extern template class LongVector<60>;
extern template class LongVector<64>;
extern template class LongVector<70>;
extern template class LongVector<80>;
extern template class LongVector<84>;
extern template class LongVector<100>;
extern template class LongVector<105>;
extern template class LongVector<112>;
extern template class LongVector<115>;
extern template class LongVector<125>;
extern template class LongVector<140>;
extern template class LongVector<168>;
extern template class LongVector<175>;
extern template class LongVector<188>;
extern template class LongVector<203>;
extern template class LongVector<224>;
extern template class LongVector<252>;
extern template class LongVector<280>;
extern template class LongVector<287>;
extern template class LongVector<308>;
extern template class LongVector<336>;
extern template class LongVector<420>;