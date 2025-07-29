#pragma once
#include "solver/eigen_sparse_solver.h"
#include "Eigen/Sparse"
#include "unsupported/Eigen/IterativeSolvers"

template <uInt BlockRows, uInt BlockCols>
EigenSparseSolver<BlockRows,BlockCols>::EigenSparseSolver(const BlockSparseMatrix<BlockRows,BlockCols>& mat, const LongVector<BlockRows>& rhs){
    uInt ELL_row_length = mat.storage.ell_blocks.size();
    uInt ELL_max_row = mat.storage.ell_max_per_row;
    uInt CSR_nnz = mat.storage.csr_blocks.size();
    uInt NNZ = (ELL_row_length * ELL_max_row + CSR_nnz) * BlockRows * BlockCols;
    uInt ROWs = mat.num_block_rows * BlockRows;

    m_tripletList.reserve(NNZ);
    m_eigenRhs.resize(ROWs);

    for(uInt brow=0; brow<mat.num_block_rows; ++brow){
        for(uInt i=0; i<mat.storage.ell_max_per_row; ++i){
            const uInt bcol = mat.storage.ell_cols[brow][i];
            if(bcol == mat.invalid_index) continue;
            const auto& block = mat.storage.ell_blocks[brow][i];
            for(uInt row=0;row<block.rows();row++){
                for(uInt col=0;col<block.cols();col++){
                    // if(std::abs(block(row,col))>1e-15)
                    m_tripletList.push_back(Triplet(brow*block.rows()+row,bcol*block.cols()+col,block(row,col)));
                }
            }
        }

        const uInt start = mat.storage.csr_row_ptr[brow];
        const uInt end = mat.storage.csr_row_ptr[brow+1];
        for(uInt idx = start; idx < end; ++idx) {
            const uInt bcol = mat.storage.csr_cols[idx];
            const auto& block = mat.storage.csr_blocks[idx];

            for(uInt row=0;row<block.rows();row++){
                for(uInt col=0;col<block.cols();col++){
                    // if(std::abs(block(row,col))>1e-15)
                    m_tripletList.push_back(Triplet(brow*block.rows()+row,bcol*block.cols()+col,block(row,col)));
                }
            }
        }
    }

    for(uInt r=0;r<rhs.size();r++){
        auto& block = rhs[r];
        for(uInt rr=0;rr<block.size();rr++){
            m_eigenRhs[r * block.size() + rr] = block[rr];
        }
    }
}



template <uInt BlockRows, uInt BlockCols>
LongVector<BlockCols> EigenSparseSolver<BlockRows,BlockCols>::SparseLU(const LongVector<BlockCols>& x0){
    EigenCSC m_CSCmat(m_eigenRhs.size(), m_eigenRhs.size());
    Eigen::SparseLU<EigenCSC> m_splu;
    
    m_CSCmat.setFromTriplets(m_tripletList.begin(), m_tripletList.end());
    m_splu.compute(m_CSCmat);
    
    if(m_splu.info()!=Eigen::Success) {
        throw std::runtime_error("Matrix decomposition failed");
    }
    const Eigen::VectorXd& m_eigenX = m_splu.solve(m_eigenRhs);

    LongVector<BlockCols> dx(x0.size());
    for(uInt r=0;r<dx.size();r++){
        auto& block = dx[r];
        for(uInt rr=0;rr<block.size();rr++){
            block[rr] = m_eigenX[r * block.size() + rr];
        }
    }
    return dx;
}


template <uInt BlockRows, uInt BlockCols>
LongVector<BlockCols> EigenSparseSolver<BlockRows,BlockCols>::BiCGSTAB(const LongVector<BlockCols>& x0){
    EigenCSR m_CSRmat(m_eigenRhs.size(), m_eigenRhs.size());
    Eigen::BiCGSTAB<EigenCSR> m_bicg;
    m_bicg.setTolerance(m_tol);
    if(m_maxiters != uInt(-1)) m_bicg.setMaxIterations(m_maxiters);

    m_CSRmat.setFromTriplets(m_tripletList.begin(), m_tripletList.end());
    m_bicg.compute(m_CSRmat);

    Eigen::VectorXd m_eigenX0;
    m_eigenX0.resize(m_eigenRhs.size());
    for(uInt r=0;r<x0.size();r++){
        const auto& block = x0[r];
        for(uInt rr=0;rr<block.size();rr++){
            m_eigenX0[r * block.size() + rr] = block[rr];
        }
    }

    const Eigen::VectorXd& m_eigenX = m_bicg.solveWithGuess(m_eigenRhs,m_eigenX0);

    LongVector<BlockCols> dx(x0.size());
    for(uInt r=0;r<dx.size();r++){
        auto& block = dx[r];
        for(uInt rr=0;rr<block.size();rr++){
            block[rr] = m_eigenX[r * block.size() + rr];
        }
    }
    return dx;
}


template <uInt BlockRows, uInt BlockCols>
LongVector<BlockCols> EigenSparseSolver<BlockRows,BlockCols>::DGMRES(const LongVector<BlockCols>& x0){
    EigenCSR m_CSRmat(m_eigenRhs.size(), m_eigenRhs.size());
    Eigen::DGMRES<EigenCSR> m_gmres;
    m_gmres.setTolerance(m_tol);
    if(m_maxiters != uInt(-1)) m_gmres.setMaxIterations(m_maxiters);
    if(m_restart != uInt(-1)) m_gmres.set_restart(m_restart);

    m_CSRmat.setFromTriplets(m_tripletList.begin(), m_tripletList.end());
    m_gmres.compute(m_CSRmat);

    Eigen::VectorXd m_eigenX0;
    m_eigenX0.resize(m_eigenRhs.size());
    for(uInt r=0;r<x0.size();r++){
        const auto& block = x0[r];
        for(uInt rr=0;rr<block.size();rr++){
            m_eigenX0[r * block.size() + rr] = block[rr];
        }
    }


    const Eigen::VectorXd& m_eigenX = m_gmres.solveWithGuess(m_eigenRhs,m_eigenX0);
    //std::cout << "#iterations: " << m_gmres.iterations() << std::endl;
    //std::cout << "estimated error: " << m_gmres.error() << std::endl;
    LongVector<BlockCols> dx(x0.size());
    for(uInt r=0;r<dx.size();r++){
        auto& block = dx[r];
        for(uInt rr=0;rr<block.size();rr++){
            block[rr] = m_eigenX[r * block.size() + rr];
        }
    }
    return dx;
}
template <uInt BlockRows, uInt BlockCols>
LongVector<BlockCols> EigenSparseSolver<BlockRows,BlockCols>::DGMRES(const LongVector<BlockCols>& x0, uInt& iter, Scalar& residual){
    EigenCSR m_CSRmat(m_eigenRhs.size(), m_eigenRhs.size());
    Eigen::DGMRES<EigenCSR> m_gmres;
    m_gmres.setTolerance(m_tol);
    if(m_maxiters != uInt(-1)) m_gmres.setMaxIterations(m_maxiters);
    if(m_restart != uInt(-1)) m_gmres.set_restart(m_restart);

    m_CSRmat.setFromTriplets(m_tripletList.begin(), m_tripletList.end());
    m_gmres.compute(m_CSRmat);

    Eigen::VectorXd m_eigenX0;
    m_eigenX0.resize(m_eigenRhs.size());
    for(uInt r=0;r<x0.size();r++){
        const auto& block = x0[r];
        for(uInt rr=0;rr<block.size();rr++){
            m_eigenX0[r * block.size() + rr] = block[rr];
        }
    }


    const Eigen::VectorXd& m_eigenX = m_gmres.solveWithGuess(m_eigenRhs,m_eigenX0);
    iter = m_gmres.iterations();
    residual = m_gmres.error();
    //std::cout << "#iterations: " << m_gmres.iterations() << std::endl;
    //std::cout << "estimated error: " << m_gmres.error() << std::endl;
    LongVector<BlockCols> dx(x0.size());
    for(uInt r=0;r<dx.size();r++){
        auto& block = dx[r];
        for(uInt rr=0;rr<block.size();rr++){
            block[rr] = m_eigenX[r * block.size() + rr];
        }
    }
    return dx;
}

template <uInt BlockRows, uInt BlockCols>
std::tuple<uInt,Scalar> EigenSparseSolver<BlockRows,BlockCols>::DGMRES(const LongVector<BlockCols>& x0,LongVector<BlockCols>& dx){
    EigenCSR m_CSRmat(m_eigenRhs.size(), m_eigenRhs.size());
    Eigen::DGMRES<EigenCSR> m_gmres;
    m_gmres.setTolerance(m_tol);
    if(m_maxiters != uInt(-1)) m_gmres.setMaxIterations(m_maxiters);
    if(m_restart != uInt(-1)) m_gmres.set_restart(m_restart);

    m_CSRmat.setFromTriplets(m_tripletList.begin(), m_tripletList.end());
    m_gmres.compute(m_CSRmat);

    Eigen::VectorXd m_eigenX0;
    m_eigenX0.resize(m_eigenRhs.size());
    for(uInt r=0;r<x0.size();r++){
        const auto& block = x0[r];
        for(uInt rr=0;rr<block.size();rr++){
            m_eigenX0[r * block.size() + rr] = block[rr];
        }
    }


    const Eigen::VectorXd& m_eigenX = m_gmres.solveWithGuess(m_eigenRhs,m_eigenX0);
    //std::cout << "#iterations: " << m_gmres.iterations() << std::endl;
    //std::cout << "estimated error: " << m_gmres.error() << std::endl;
    //LongVector<BlockCols> dx(x0.size());
    for(uInt r=0;r<dx.size();r++){
        auto& block = dx[r];
        for(uInt rr=0;rr<block.size();rr++){
            block[rr] = m_eigenX[r * block.size() + rr];
        }
    }
    return {m_gmres.iterations(), m_gmres.error()};
}
