#pragma once
#include "DenseMatrix.h"
#include "LongVector.h"


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

};