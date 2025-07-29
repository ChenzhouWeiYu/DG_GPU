#pragma once
#include "base/type.h"
#include "dg/dg_basis/dg_basis.h"
#include "mesh/mesh.h"
#include "matrix/matrix.h"
#include "base/exact.h"
#include "dg/dg_flux/euler_physical_flux.h"

template<uInt Order=3, typename Flux = AirFluxC, 
        typename GaussQuadCell = GaussLegendreTet::Auto, 
        typename GaussQuadFace = GaussLegendreTri::Auto>
class ImplicitConvection {
private:
    using Basis = DGBasisEvaluator<Order>;
    static constexpr uInt N = Basis::NumBasis;
    using Mat5x1 = DenseMatrix<5,1>;
    using Mat5x5 = DenseMatrix<5,5>;
    using BlkMat = DenseMatrix<5*N,5*N>;

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

    vector3f transform_to_cell(const CompTriangleFace& face, const vector2f& uv, uInt side) const ;

public:
    void assemble(const ComputingMesh& mesh, 
                 const LongVector<5*N>& old_solution,
                 const Scalar curr_time,
                 BlockSparseMatrix<5*N,5*N>& sparse_mat,
                 LongVector<5*N>& sparse_rhs);
    void assemble_cells(const ComputingMesh& mesh, 
                 const LongVector<5*N>& old_solution,
                 const Scalar curr_time,
                 BlockSparseMatrix<5*N,5*N>& sparse_mat,
                 LongVector<5*N>& sparse_rhs);
    void assemble_internals(const ComputingMesh& mesh, 
                 const LongVector<5*N>& old_solution,
                 const Scalar curr_time,
                 BlockSparseMatrix<5*N,5*N>& sparse_mat,
                 LongVector<5*N>& sparse_rhs);
    void assemble_boundarys(const ComputingMesh& mesh, 
                 const LongVector<5*N>& old_solution,
                 const Scalar curr_time,
                 BlockSparseMatrix<5*N,5*N>& sparse_mat,
                 LongVector<5*N>& sparse_rhs);
                 
};


#define Explicit_For_Flux(Order) \
extern template class ImplicitConvection<Order,AirFluxC>;\
extern template class ImplicitConvection<Order,MonatomicFluxC>;\
extern template class ImplicitConvection<Order+1,AirFluxC>;\
extern template class ImplicitConvection<Order+1,MonatomicFluxC>;\
extern template class ImplicitConvection<Order+2,AirFluxC>;\
extern template class ImplicitConvection<Order+2,MonatomicFluxC>;

Explicit_For_Flux(0)
Explicit_For_Flux(3)

#undef Explicit_For_Flux