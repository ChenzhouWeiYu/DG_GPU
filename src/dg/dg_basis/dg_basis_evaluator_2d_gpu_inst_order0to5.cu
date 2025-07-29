#include "dg/dg_basis/dg_basis.h"
#include "dg/dg_basis/dg_basis_evaluator_2d.h"
#include "dg/dg_basis/dg_basis_evaluator_2d_impl.h"


#define explict_template_for_NumPhyDim(Order,N,Type) \
template DenseMatrix<N,1> DGBasisEvaluator2D<Order>::coef2filed(const std::array<DenseMatrix<N,1>, NumBasis>& coef, const Type x, const Type y);\
template DenseMatrix<N,1> DGBasisEvaluator2D<Order>::coef2filed(const std::array<DenseMatrix<N,1>, NumBasis>& coef, const std::array<Type,2>& p);\
template DenseMatrix<N,1> DGBasisEvaluator2D<Order>::coef2filed(const DenseMatrix<N*NumBasis,1>& coef, const Type x, const Type y);\
template DenseMatrix<N,1> DGBasisEvaluator2D<Order>::coef2filed(const DenseMatrix<N*NumBasis,1>& coef, const std::array<Type,2>& p);\
template std::vector<DenseMatrix<N,1>> DGBasisEvaluator2D<Order>::coef2filed(const DenseMatrix<N*NumBasis,1>& coef, const std::vector<std::array<Type,2>>& p);


#define explict_template_instantiation(Order) \
template class DGBasisEvaluator2D<Order>;\
template Scalar DGBasisEvaluator2D<Order>::coef2filed(const std::array<Scalar, NumBasis>& coef, const Scalar x, const Scalar y);\
template Scalar DGBasisEvaluator2D<Order>::coef2filed(const std::array<Scalar, NumBasis>& coef, const std::array<Scalar,2>& p);\
template std::array<Scalar, DGBasisEvaluator2D<Order>::NumBasis> DGBasisEvaluator2D<Order>::eval_all(const Scalar x, const Scalar y);\
template std::array<std::array<Scalar,2>, DGBasisEvaluator2D<Order>::NumBasis> DGBasisEvaluator2D<Order>::grad_all(const Scalar x, const Scalar y);\
explict_template_for_NumPhyDim(Order,4,Scalar)\
explict_template_for_NumPhyDim(Order,5,Scalar)



explict_template_instantiation(0) 
explict_template_instantiation(1) 
explict_template_instantiation(2) 
explict_template_instantiation(3) 
explict_template_instantiation(4) 
explict_template_instantiation(5) 
// explict_template_instantiation(6) 
// explict_template_instantiation(7) 
// explict_template_instantiation(8) 
// explict_template_instantiation(9) 
// explict_template_instantiation(10)

#undef explict_template_instantiation

#undef explict_template_for_NumPhyDim