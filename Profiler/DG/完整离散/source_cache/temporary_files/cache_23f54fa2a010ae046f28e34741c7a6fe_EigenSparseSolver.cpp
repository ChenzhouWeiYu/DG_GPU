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




template class EigenSparseSolver<1,1>;
template class EigenSparseSolver<2,2>;
template class EigenSparseSolver<3,3>;
template class EigenSparseSolver<4,4>;
template class EigenSparseSolver<5,5>;
template class EigenSparseSolver<6,6>;
template class EigenSparseSolver<7,7>;
template class EigenSparseSolver<8,8>;
template class EigenSparseSolver<9,9>;
template class EigenSparseSolver<10,10>;
template class EigenSparseSolver<11,11>;
template class EigenSparseSolver<12,12>;
template class EigenSparseSolver<13,13>;
template class EigenSparseSolver<14,14>;
template class EigenSparseSolver<15,15>;
template class EigenSparseSolver<16,16>;
template class EigenSparseSolver<17,17>;
template class EigenSparseSolver<18,18>;
template class EigenSparseSolver<19,19>;
template class EigenSparseSolver<20,20>;
template class EigenSparseSolver<24,24>;
template class EigenSparseSolver<28,28>;
template class EigenSparseSolver<30,30>;
template class EigenSparseSolver<31,31>;
template class EigenSparseSolver<32,32>;
template class EigenSparseSolver<34,34>;
template class EigenSparseSolver<35,35>;
template class EigenSparseSolver<36,36>;
template class EigenSparseSolver<40,40>;
template class EigenSparseSolver<44,44>;
template class EigenSparseSolver<48,48>;
template class EigenSparseSolver<50,50>;
template class EigenSparseSolver<52,52>;
template class EigenSparseSolver<56,56>;
template class EigenSparseSolver<60,60>;
template class EigenSparseSolver<64,64>;
template class EigenSparseSolver<68,68>;
template class EigenSparseSolver<70,70>;
template class EigenSparseSolver<72,72>;
template class EigenSparseSolver<76,76>;
template class EigenSparseSolver<80,80>;
template class EigenSparseSolver<84,84>;
template class EigenSparseSolver<90,90>;
template class EigenSparseSolver<100,100>;
template class EigenSparseSolver<105,105>;
template class EigenSparseSolver<110,110>;
template class EigenSparseSolver<112,112>;
template class EigenSparseSolver<115,115>;
template class EigenSparseSolver<120,120>;
template class EigenSparseSolver<125,125>;
template class EigenSparseSolver<130,130>;
template class EigenSparseSolver<140,140>;
template class EigenSparseSolver<150,150>;
template class EigenSparseSolver<160,160>;
template class EigenSparseSolver<165,165>;
template class EigenSparseSolver<168,168>;
template class EigenSparseSolver<170,170>;
template class EigenSparseSolver<175,175>;
template class EigenSparseSolver<180,180>;
template class EigenSparseSolver<188,188>;
template class EigenSparseSolver<190,190>;
template class EigenSparseSolver<200,200>;
template class EigenSparseSolver<203,203>;
template class EigenSparseSolver<210,210>;
template class EigenSparseSolver<220,220>;
template class EigenSparseSolver<224,224>;
template class EigenSparseSolver<240,240>;
template class EigenSparseSolver<245,245>;
template class EigenSparseSolver<252,252>;
template class EigenSparseSolver<260,260>;
template class EigenSparseSolver<280,280>;
template class EigenSparseSolver<286,286>;
template class EigenSparseSolver<287,287>;
template class EigenSparseSolver<300,300>;
template class EigenSparseSolver<308,308>;
template class EigenSparseSolver<315,315>;
template class EigenSparseSolver<320,320>;
template class EigenSparseSolver<330,330>;
template class EigenSparseSolver<336,336>;
template class EigenSparseSolver<340,340>;
template class EigenSparseSolver<350,350>;
template class EigenSparseSolver<360,360>;
template class EigenSparseSolver<380,380>;
template class EigenSparseSolver<385,385>;
template class EigenSparseSolver<392,392>;
template class EigenSparseSolver<400,400>;
template class EigenSparseSolver<416,416>;
template class EigenSparseSolver<420,420>;
template class EigenSparseSolver<440,440>;
template class EigenSparseSolver<444,444>;
template class EigenSparseSolver<448,448>;
template class EigenSparseSolver<455,455>;
template class EigenSparseSolver<480,480>;
template class EigenSparseSolver<490,490>;
template class EigenSparseSolver<495,495>;
template class EigenSparseSolver<504,504>;
template class EigenSparseSolver<525,525>;
template class EigenSparseSolver<560,560>;
template class EigenSparseSolver<572,572>;
template class EigenSparseSolver<579,579>;
template class EigenSparseSolver<588,588>;
template class EigenSparseSolver<595,595>;
template class EigenSparseSolver<600,600>;
template class EigenSparseSolver<615,615>;
template class EigenSparseSolver<616,616>;
template class EigenSparseSolver<630,630>;
template class EigenSparseSolver<660,660>;
template class EigenSparseSolver<665,665>;
template class EigenSparseSolver<672,672>;
template class EigenSparseSolver<700,700>;
template class EigenSparseSolver<720,720>;
template class EigenSparseSolver<728,728>;
template class EigenSparseSolver<756,756>;
template class EigenSparseSolver<780,780>;
template class EigenSparseSolver<784,784>;
template class EigenSparseSolver<825,825>;
template class EigenSparseSolver<840,840>;
template class EigenSparseSolver<858,858>;
template class EigenSparseSolver<880,880>;
template class EigenSparseSolver<896,896>;
template class EigenSparseSolver<924,924>;
template class EigenSparseSolver<952,952>;
template class EigenSparseSolver<960,960>;
template class EigenSparseSolver<990,990>;
template class EigenSparseSolver<1008,1008>;
template class EigenSparseSolver<1023,1023>;
template class EigenSparseSolver<1064,1064>;
template class EigenSparseSolver<1078,1078>;
template class EigenSparseSolver<1080,1080>;
template class EigenSparseSolver<1092,1092>;
template class EigenSparseSolver<1100,1100>;
template class EigenSparseSolver<1120,1120>;
template class EigenSparseSolver<1144,1144>;
template class EigenSparseSolver<1155,1155>;
template class EigenSparseSolver<1176,1176>;
template class EigenSparseSolver<1200,1200>;
template class EigenSparseSolver<1260,1260>;
template class EigenSparseSolver<1320,1320>;
template class EigenSparseSolver<1344,1344>;
template class EigenSparseSolver<1428,1428>;
template class EigenSparseSolver<1430,1430>;
template class EigenSparseSolver<1440,1440>;
template class EigenSparseSolver<1485,1485>;
template class EigenSparseSolver<1512,1512>;
template class EigenSparseSolver<1540,1540>;
template class EigenSparseSolver<1560,1560>;
template class EigenSparseSolver<1596,1596>;
template class EigenSparseSolver<1650,1650>;
template class EigenSparseSolver<1680,1680>;
template class EigenSparseSolver<1716,1716>;
template class EigenSparseSolver<1760,1760>;
template class EigenSparseSolver<1800,1800>;
template class EigenSparseSolver<1815,1815>;
template class EigenSparseSolver<1920,1920>;
template class EigenSparseSolver<1980,1980>;
template class EigenSparseSolver<2002,2002>;
template class EigenSparseSolver<2040,2040>;
template class EigenSparseSolver<2145,2145>;
template class EigenSparseSolver<2160,2160>;
template class EigenSparseSolver<2200,2200>;
template class EigenSparseSolver<2280,2280>;
template class EigenSparseSolver<2288,2288>;
template class EigenSparseSolver<2310,2310>;
template class EigenSparseSolver<2400,2400>;
template class EigenSparseSolver<2420,2420>;
template class EigenSparseSolver<2475,2475>;
template class EigenSparseSolver<2574,2574>;
template class EigenSparseSolver<2640,2640>;
template class EigenSparseSolver<2805,2805>;
template class EigenSparseSolver<2860,2860>;
template class EigenSparseSolver<2970,2970>;
template class EigenSparseSolver<3080,3080>;
template class EigenSparseSolver<3135,3135>;
template class EigenSparseSolver<3146,3146>;
template class EigenSparseSolver<3300,3300>;
template class EigenSparseSolver<3432,3432>;
template class EigenSparseSolver<3520,3520>;
template class EigenSparseSolver<3718,3718>;
template class EigenSparseSolver<3740,3740>;
template class EigenSparseSolver<3960,3960>;
template class EigenSparseSolver<4004,4004>;
template class EigenSparseSolver<4180,4180>;
template class EigenSparseSolver<4290,4290>;
template class EigenSparseSolver<4400,4400>;
template class EigenSparseSolver<4576,4576>;
template class EigenSparseSolver<4862,4862>;
template class EigenSparseSolver<5148,5148>;
template class EigenSparseSolver<5434,5434>;
template class EigenSparseSolver<5720,5720>;