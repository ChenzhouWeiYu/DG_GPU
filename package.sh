#!/bin/bash

# package.sh - 支持完全模式和简化模式的打包脚本

# 默认是简化模式
LITE_MODE=1
WITH_EXTE=0
WITH_IPYNB=0

# 解析命令行参数
for arg in "$@"; do
    if [[ "$arg" == "--lite" ]]; then
        LITE_MODE=1
    fi
    if [[ "$arg" == "--full" ]]; then
        LITE_MODE=0
    fi
    if [[ "$arg" == "--with-external" ]]; then
        WITH_EXTE=1
    fi
    if [[ "$arg" == "--with-ipynb" ]]; then
        WITH_IPYNB=1
    fi
done

# 获取项目路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/.."
cd "$PROJECT_DIR" || { echo "无法进入项目目录"; exit 1; }

# 自动生成时间戳包名
TIMESTAMP=$(date "+%Y%m%d_%H%M")

if (( LITE_MODE )); then
    nametar="FVE_DG_lite_${TIMESTAMP}.tar.bz2"
else
    nametar="FVE_DG_full_${TIMESTAMP}.tar.bz2"
fi

# 基础排除规则（适用于所有模式）
EXCLUDE_OPTIONS=(
    --exclude=Profiler
    --exclude=FVE/*
    --exclude=build
    --exclude=*/.ipynb_checkpoints
    --exclude=QuadData/*.pdf
)

# 如果是简化模式，再加一条排除数值解文件夹
if (( LITE_MODE )); then
    # EXCLUDE_OPTIONS+=(--exclude=Order_*/solution/*.txt)
    # EXCLUDE_OPTIONS+=(--exclude=Order_*/*.txt)
    EXCLUDE_OPTIONS+=(--exclude=Order_*)
    EXCLUDE_OPTIONS+=(--exclude=*.log)
    EXCLUDE_OPTIONS+=(--exclude=*.vtk)
    EXCLUDE_OPTIONS+=(--exclude=obj/run.o)
    EXCLUDE_OPTIONS+=(--exclude=obj/mesh.o)
    EXCLUDE_OPTIONS+=(--exclude=obj/problem.o)
    EXCLUDE_OPTIONS+=(--exclude=run)
fi

if (( WITH_EXTE )); then
    echo With_external
else
    EXCLUDE_OPTIONS+=(--exclude=external)
fi

if (( WITH_IPYNB )); then
    echo With_ipynb
else
    EXCLUDE_OPTIONS+=(--exclude=*.ipynb)
fi

# 查找并清理可执行文件的符号表
function strip_executables() {
    find "FVE_DG_GPU" -type f -executable -not -path "*/\.*" 2>/dev/null | while read -r file; do
        if file "$file" | grep -q 'ELF\|Mach-O'; then
            echo "[STRIP] 正在清理符号表: $file"
            strip --strip-all "$file" 2>/dev/null || true
        fi
    done
}

echo "[$(date)] 开始打包 FVE_DG..."

# Step 1: 清理可执行文件符号表
strip_executables

# Step 2: 打包成 .tar.bz2
if command -v pbzip2 &> /dev/null; then
    echo "[$(date)] 使用 pbzip2 并行压缩..."
    tar -cvf "$nametar" "${EXCLUDE_OPTIONS[@]}" FVE_DG_GPU \
        --use-compress-program="pbzip2 -9"
elif command -v bzip2 &> /dev/null; then
    echo "[$(date)] 使用 bzip2 压缩..."
    tar -cvjf "$nametar" "${EXCLUDE_OPTIONS[@]}" FVE_DG_GPU
else
    echo "[$(date)] 未找到 bzip2/pbzip2，尝试使用 7z 替代..."
    if command -v 7z &> /dev/null; then
        name7z="${nametar/.tar.bz2/.7z}"
        # 构建 exclude 列表用于 7z
        EXCLUDE_LIST=""
        for ex in "${EXCLUDE_OPTIONS[@]}"; do
            flag="${ex%%=*}"
            pattern="${ex#*=}"
            if [[ "$flag" == "--exclude" ]]; then
                EXCLUDE_LIST+=" -xr!$pattern"
            else
                echo "警告：7z 不支持此排除规则: $ex"
            fi
        done
        eval "7z a -t7z '$name7z' FVE_DG_GPU $EXCLUDE_LIST > /dev/null"
        echo "[$(date)] 已打包为 7z 格式: $name7z"
    else
        echo "[ERROR] 未安装 tar/bzip2/pbzip2/7z，请先安装这些工具！"
        exit 1
    fi
fi

# Step 3: 完成提示
echo "[$(date)] 打包完成!"
ls -lh "$nametar" "$name7z" 2>/dev/null || ls -lh "$nametar"
