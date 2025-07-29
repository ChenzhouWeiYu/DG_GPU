#include "dg/dg_basis/dg_basis.h"
#include "dg/dg_basis/dg_basis_evaluator.h"
#include "dg/dg_basis/dg_basis_evaluator_impl.h"


#define explict_template_for_NumPhyDim(Order,N,Type) \
template DenseMatrix<N,1> DGBasisEvaluator<Order>::coef2filed(const std::array<DenseMatrix<N,1>, NumBasis>& coef, const Type x, const Type y, const Type z);\
template DenseMatrix<N,1> DGBasisEvaluator<Order>::coef2filed(const std::array<DenseMatrix<N,1>, NumBasis>& coef, const std::array<Type,3>& p);\
template DenseMatrix<N,1> DGBasisEvaluator<Order>::coef2filed(const DenseMatrix<N*NumBasis,1>& coef, const Type x, const Type y, const Type z);\
template DenseMatrix<N,1> DGBasisEvaluator<Order>::coef2filed(const DenseMatrix<N*NumBasis,1>& coef, const std::array<Type,3>& p);\
template std::vector<DenseMatrix<N,1>> DGBasisEvaluator<Order>::coef2filed(const DenseMatrix<N*NumBasis,1>& coef, const std::vector<std::array<Type,3>>& p);


#define explict_template_instantiation(Order) \
template class DGBasisEvaluator<Order>;\
template Scalar DGBasisEvaluator<Order>::coef2filed(const std::array<Scalar, NumBasis>& coef, const Scalar x, const Scalar y, const Scalar z);\
template Scalar DGBasisEvaluator<Order>::coef2filed(const std::array<Scalar, NumBasis>& coef, const std::array<Scalar,3>& p);\
template std::array<Scalar, DGBasisEvaluator<Order>::NumBasis> DGBasisEvaluator<Order>::eval_all(const Scalar x, const Scalar y, const Scalar z);\
template std::array<std::array<Scalar,3>, DGBasisEvaluator<Order>::NumBasis> DGBasisEvaluator<Order>::grad_all(const Scalar x, const Scalar y, const Scalar z);\
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