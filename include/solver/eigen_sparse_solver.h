#pragma once
#include "base/type.h"
#include "matrix/sparse_matrix.h"
#include "Eigen/SparseCore"
// #include "Eigen/Sparse"
// #include "unsupported/Eigen/IterativeSolvers"

using EigenCSR = Eigen::SparseMatrix<Scalar,Eigen::RowMajor>;
using EigenCSC = Eigen::SparseMatrix<Scalar,Eigen::ColMajor>;
using Triplet =  Eigen::Triplet<Scalar>;


template <uInt BlockRows, uInt BlockCols>
class EigenSparseSolver {
public:
    EigenSparseSolver(const BlockSparseMatrix<BlockRows,BlockCols>& mat, const LongVector<BlockRows>& rhs);

    LongVector<BlockCols> SparseLU(const LongVector<BlockCols>& x0);
    LongVector<BlockCols> BiCGSTAB(const LongVector<BlockCols>& x0);
    LongVector<BlockCols> DGMRES(const LongVector<BlockCols>& x0);
    LongVector<BlockCols> DGMRES(const LongVector<BlockCols>& x0,uInt& iter, Scalar& residual);
    std::tuple<uInt,Scalar> DGMRES(const LongVector<BlockCols>& x0, LongVector<BlockCols>& ret);

    void set_tol(Scalar tol = 1e-12){m_tol = tol;};
    void set_iterations(uInt it){m_maxiters = it;};
    void set_restart(uInt it = 30){m_restart = it;};

private:
    std::vector<Triplet> m_tripletList;
    Scalar m_tol = 1e-12;
    uInt m_maxiters = uInt(-1);
    uInt m_restart = uInt(-1);
    // EigenCSR m_CSRmat;
    // EigenCSC m_CSCmat;
    Eigen::VectorXd m_eigenRhs;
    // Eigen::SparseLU<EigenCSC> m_lu;
    // Eigen::BiCGSTAB<EigenCSR> m_bicg;
    // Eigen::DGMRES<EigenCSR> m_gmres;
};