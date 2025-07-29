#!/bin/bash

# 创建临时文件存储所有维度
tmpfile=$(mktemp)

function combine() {
    local p=$1
    echo $(( (p + 1) * (p + 2) * (p + 3) / 6 ))
}


# 遍历 orderU 和 varNum
for (( orderU=0; orderU<=10; orderU++ )); do
    for varNum in {1..20}; do
        echo $(( $varNum * $(combine $orderU) )) >> "$tmpfile"
    done

    # 公式 1: orderU * 3 + orderU
    echo $(( $(combine $orderU) * 3 + $(combine $orderU) )) >> "$tmpfile"

    # 公式 2: orderU * 3 + orderU - 1
    echo $(( $(combine $orderU) * 3 + $(combine $orderU-1) )) >> "$tmpfile"

    # 公式 3: orderU * 3 + orderU - 2 （仅当 orderU > 1）
    if (( orderU > 1 )); then
        echo $(( $(combine $orderU) * 3 + $(combine $orderU-2) )) >> "$tmpfile"
    fi

    # 如果你还有更多公式，可以继续加在这里...
done

# 去重并排序
sorted_dims=$(sort -nu "$tmpfile")

# 输出结果为 template class 行
for dim in $sorted_dims; do
    echo "template class EigenSparseSolver<$dim,$dim>;"
done

# 清理临时文件
rm -f "$tmpfile"