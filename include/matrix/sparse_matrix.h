#pragma once
#include "matrix/dense_matrix.h"
#include "matrix/long_vector.h"


template <uInt BlockRows, uInt BlockCols>
class BlockSparseMatrix {
public:
    // COO格式临时存储
    struct COOBlock {
        uInt row;
        uInt col;
        DenseMatrix<BlockRows, BlockCols> block;
        
        // 哈希支持
        bool operator==(const COOBlock& other) const {
            return row == other.row && col == other.col;
        }
    };
    
    struct COOHash {
        size_t operator()(const COOBlock& b) const {
            return (static_cast<size_t>(b.row) << 32) | b.col;
        }
    };

    std::unordered_set<COOBlock, COOHash> coo_blocks; // 自动去重

    // ELL+CSR存储
    struct {
        // ELL部分
        // std::vector<DenseMatrix<BlockRows, BlockCols>> ell_blocks;
        // std::vector<uInt> ell_cols;
        // uInt ell_max_per_row = 0;
        std::vector<std::vector<DenseMatrix<BlockRows, BlockCols>>> ell_blocks;
        std::vector<std::vector<uInt>> ell_cols;
        uInt ell_max_per_row = 0;
        
        // CSR部分
        std::vector<uInt> csr_row_ptr;
        std::vector<DenseMatrix<BlockRows, BlockCols>> csr_blocks;
        std::vector<uInt> csr_cols;
    } storage;

    uInt num_block_rows = 0;
    bool finalized = false;

public:
    // ================= 矩阵组装接口 =================
    void add_block(uInt row, uInt col, const DenseMatrix<BlockRows, BlockCols>& blk) {
        check_assembly_state();
        COOBlock key{row, col, {}};
        
        if(auto it = coo_blocks.find(key); it != coo_blocks.end()) {
            // 块已存在，累加
            const_cast<COOBlock&>(*it).block += blk; // 去const修改
        } else {
            // 插入新块
            coo_blocks.insert({row, col, blk});
            num_block_rows = std::max(num_block_rows, row+1);
        }
    }

    // ================= 矩阵定型 =================
    void finalize() {
        // debug("111");
        check_assembly_state();
        // debug("2222");
        build_storage();
        // debug("33333");
        coo_blocks.clear();
        // debug("444444");
        finalized = true;
    }

    // ================= 稀疏矩阵向量乘 =================
    LongVector<BlockRows> multiply(const LongVector<BlockCols>& vec) const {
        check_finalized();
        LongVector<BlockRows> result(num_block_rows); // 预分配空间

        // ELL部分并行计算
        // #pragma omp parallel for schedule(dynamic, 16)
        // for(uInt brow = 0; brow < num_block_rows; ++brow) {
        //     const uInt start_idx = brow * storage.ell_max_per_row;
            
        //     for(uInt i = 0; i < storage.ell_max_per_row; ++i) {
        //         const uInt idx = start_idx + i;
        //         if(idx >= storage.ell_blocks.size()) break;
                
        //         const uInt bcol = storage.ell_cols[idx];
        //         // debug(storage.ell_blocks[idx]);
        //         // debug(vec[bcol]);
        //         result[brow] += storage.ell_blocks[idx].multiply(vec[bcol]);
        //     }
        // }
        // #pragma omp parallel for
        for(uInt brow=0; brow<num_block_rows; ++brow){
            for(uInt i=0; i<storage.ell_max_per_row; ++i){
                const uInt bcol = storage.ell_cols[brow][i];
                if(bcol == invalid_index) continue; // 关键跳过
                
                result[brow] += storage.ell_blocks[brow][i].multiply(vec[bcol]);
            }
        }

        // CSR部分并行计算
        // #pragma omp parallel for schedule(dynamic, 16)
        for(uInt brow = 0; brow < num_block_rows; ++brow) {
            const uInt start = storage.csr_row_ptr[brow];
            const uInt end = storage.csr_row_ptr[brow+1];
            
            for(uInt idx = start; idx < end; ++idx) {
                // debug(storage.csr_blocks[idx].multiply(vec[idx]));
                result[brow] += storage.csr_blocks[idx].multiply(vec[storage.csr_cols[idx]]);
            }
        }

        return result;
    }

private:
    // ================= 存储构建 =================
    void build_storage() {
        // 统计行非零元分布
        std::vector<uInt> row_counts(num_block_rows, 0);
        for(const auto& blk : coo_blocks)
            row_counts[blk.row]++;
        
        // 确定ELL最大长度（取前95%分位数）
        std::vector<uInt> sorted_counts(row_counts);
        std::sort(sorted_counts.begin(), sorted_counts.end(), std::greater<>());
        
        // debug(size_t(sorted_counts.size()*0.95));
        // debug(sorted_counts.size());
        storage.ell_max_per_row = sorted_counts[size_t(sorted_counts.size()*0.95)];
        // storage.ell_max_per_row = 0;


        // 初始化ELL为二维结构
        storage.ell_blocks.resize(num_block_rows);
        storage.ell_cols.resize(num_block_rows);
        storage.csr_row_ptr.resize(num_block_rows + 1);
        std::vector<uInt> ell_counters(num_block_rows, 0);
        std::vector<std::vector<std::pair<uInt, DenseMatrix<BlockRows, BlockCols>>>> csr_tmp(num_block_rows);

        // 填充有效数据
        for(uInt row=0; row<num_block_rows; ++row){
            storage.ell_blocks[row].reserve(storage.ell_max_per_row);
            storage.ell_cols[row].reserve(storage.ell_max_per_row);
        }

        // 填充数据时
        for(const auto& blk : coo_blocks) {
            if(storage.ell_blocks[blk.row].size() < storage.ell_max_per_row){
                storage.ell_blocks[blk.row].push_back(blk.block);
                storage.ell_cols[blk.row].push_back(blk.col);
            } else {
                csr_tmp[blk.row].emplace_back(blk.col, blk.block);
            }
        }

        // 填充无效元素
        for(uInt row=0; row<num_block_rows; ++row){
            while(storage.ell_blocks[row].size() < storage.ell_max_per_row){
                storage.ell_blocks[row].emplace_back(); // 默认构造空矩阵
                storage.ell_cols[row].push_back(invalid_index);
            }
        }
        // // 准备存储
        // storage.ell_blocks.resize(num_block_rows * storage.ell_max_per_row);
        // storage.ell_cols.resize(storage.ell_blocks.size(), invalid_index);
        // storage.csr_row_ptr.resize(num_block_rows + 1);

        // // 填充数据
        // std::vector<uInt> ell_counters(num_block_rows, 0);
        // std::vector<std::vector<std::pair<uInt, DenseMatrix<BlockRows, BlockCols>>>> csr_tmp(num_block_rows);
        // // debug("123");
        // for(const auto& blk : coo_blocks) {
        //     const uInt row = blk.row;
        //     if(row >= num_block_rows) {
        //         std::cerr << "Invalid row index: " << row 
        //                 << " (max allowed: " << num_block_rows-1 << ")\n";
        //         continue;
        //     }
        //     if(ell_counters[row] < storage.ell_max_per_row) {
        //         const uInt idx = row * storage.ell_max_per_row + ell_counters[row]++;
        //         storage.ell_blocks[idx] = blk.block;
        //         storage.ell_cols[idx] = blk.col;
        //     } else {
        //         // debug("123");
        //         csr_tmp[row].emplace_back(blk.col, blk.block);
        //     }
        // }
        // debug("456");
        // 构建CSR
        storage.csr_row_ptr[0] = 0;
        for(uInt row = 0; row < num_block_rows; ++row) {
            std::sort(csr_tmp[row].begin(), csr_tmp[row].end(), 
                     [](auto& a, auto& b) { return a.first < b.first; });
            
            for(const auto& [col, blk] : csr_tmp[row]) {
                storage.csr_blocks.push_back(blk);
                storage.csr_cols.push_back(col);
            }
            storage.csr_row_ptr[row+1] = storage.csr_blocks.size();
        }
    }

    // ================= 工具方法 =================
public:
    static constexpr uInt invalid_index = std::numeric_limits<uInt>::max();

    void check_assembly_state() const {
        if(finalized) throw std::runtime_error("Matrix already finalized");
    }

    void check_finalized() const {
        if(!finalized) throw std::runtime_error("Matrix not finalized");
    }

    friend std::ostream& operator<<(std::ostream& os, const BlockSparseMatrix& mat) {
        // print("11111");
        for(uInt brow=0; brow<mat.num_block_rows; ++brow){
            for(uInt i=0; i<mat.storage.ell_max_per_row; ++i){
                const uInt bcol = mat.storage.ell_cols[brow][i];
                if(bcol == invalid_index) continue;
                const auto& block = mat.storage.ell_blocks[brow][i];
                os << brow << "  " << bcol << "    ";
                for(uInt k = 0;k<block.size();k++){
                    os << block[k] << "  ";
                }
                os << "\n";
            }
            // print(brow);
            const uInt start = mat.storage.csr_row_ptr[brow];
            const uInt end = mat.storage.csr_row_ptr[brow+1];
            for(uInt idx = start; idx < end; ++idx) {
                const uInt bcol = mat.storage.csr_cols[idx];
                const auto& block = mat.storage.csr_blocks[idx];
                os << brow << "  " << bcol << "    ";
                for(uInt k = 0;k<block.size();k++){
                    os << block[k] << "  ";
                }
                os << "\n";
            }
            // print(brow);

        }
        return os;
    }


public:
    void output_as_scalar(std::string filename) {
        std::ofstream os;
        os.open(filename);
        for(uInt brow=0; brow<num_block_rows; ++brow){
            for(uInt i=0; i<storage.ell_max_per_row; ++i){
                const uInt bcol = storage.ell_cols[brow][i];
                if(bcol == invalid_index) continue;
                const auto& block = storage.ell_blocks[brow][i];
                for(uInt row=0;row<block.rows();row++){
                    for(uInt col=0;col<block.cols();col++){
                        os << brow*block.rows()+row << "  " 
                            << bcol*block.cols()+col << "    "
                            << block(row,col) << "\n";
                    }
                }
            }

            const uInt start = storage.csr_row_ptr[brow];
            const uInt end = storage.csr_row_ptr[brow+1];
            for(uInt idx = start; idx < end; ++idx) {
                const uInt bcol = storage.csr_cols[idx];
                const auto& block = storage.csr_blocks[idx];

                for(uInt row=0;row<block.rows();row++){
                    for(uInt col=0;col<block.cols();col++){
                        os << brow*block.rows()+row << "  " 
                            << bcol*block.cols()+col << "    "
                            << block(row,col) << "\n";
                    }
                }
            }

        }
        os.close();
        // return os;
    }
};