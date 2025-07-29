# common.mk - 公共配置文件
# 必须放在项目根目录下
# 被各 example 目录下的 Makefile include
# 自动探测自身位置，并设置 ROOT_DIR

# 获取当前 common.mk 的绝对路径（Makefile 中第一个被读取的文件）
ifndef COMMON_MK_ALREADY_INCLUDED
COMMON_MK_ALREADY_INCLUDED := 1

.DEFAULT_GOAL := all
# 获取 common.mk 文件的真实路径
override THIS_MAKEFILE := $(realpath $(lastword $(MAKEFILE_LIST)))
override COMMON_MK := $(THIS_MAKEFILE)
override ROOT_DIR := $(dir $(COMMON_MK))

# 去除末尾斜杠，防止路径拼接错误
override ROOT_DIR := $(patsubst %/,%,$(ROOT_DIR))


HDF5INCLUDE := -I/usr/include/hdf5/serial
HDF5FLAGS += -lhdf5 -lhdf5_cpp -lcurl 

# 编译器与标志
CXX := g++
CXXFLAGS := -std=c++17 -O3 -march=native -ffast-math #-g
CXXFLAGS += -Wall -Wno-unused-variable -Wno-unused-but-set-variable -Wno-comments -Wdeprecated-declarations
CXXFLAGS += -fopenmp -mfma -mavx2 -DCGAL_HEADER_ONLY -DCGAL_DISABLE_GMP  
CXXFLAGS += -MMD -MP 
CXXFLAGS += $(HDF5FLAGS)

LDFLAGS := -fopenmp -flto
LDFLAGS += $(HDF5FLAGS)


CLANGFLAGS := -mllvm -polly -polly-parallel
# CXXFLAGS += $(CLANGFLAGS)

# CUDA 配置
NVCC := nvcc
NVCCFLAGS := -std=c++17 -O3 --use_fast_math #-g -lineinfo
NVCCFLAGS += -Xcompiler -fopenmp --expt-relaxed-constexpr
NVCCFLAGS += -I$(ROOT_DIR)/include -I$(ROOT_DIR)/external 
NVCCFLAGS += -Xcompiler -MMD -Xcompiler -MP
NVCCFLAGS += -gencode arch=compute_75,code=sm_75

NVCCFLAGS += $(HDF5FLAGS)

CUDA_LIB := -L/usr/local/cuda/lib64 -lcudart 
CUDA_LIB += $(HDF5FLAGS)

LDFLAGS += $(CUDA_LIB)

# 包含路径
INCLUDE_FLAGS := -I$(ROOT_DIR)/include -I$(ROOT_DIR)/external 
INCLUDE_FLAGS += $(HDF5INCLUDE)


# 构建输出目录（所有 src/.o 都放这里）
BUILD_DIR := $(ROOT_DIR)/build
OBJ_DIR := $(BUILD_DIR)/obj

# src 路径
SRC_DIR := $(ROOT_DIR)/src

# 自动扫描 src 下所有 .cpp 文件
SRC_SUBDIRS := $(shell find $(SRC_DIR) -type d)
SHARED_SRCS := $(foreach dir,$(SRC_SUBDIRS),$(wildcard $(dir)/*.cpp))

# 对应的 .o 文件路径（build/obj/src/...）
SHARED_OBJS := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SHARED_SRCS))

# 工具函数：创建目录
MKDIR_P := mkdir -p

# .PHONY: prebuild
prebuild:
	@mkdir -p $(BUILD_DIR)
	@find $(BUILD_DIR) -name '*.d' -exec sed -i '/tmpxft/d' {} +


# # 共享源文件的编译规则
# $(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
# 	@$(MKDIR_P) $(dir $@)
# 	@start_time=$$(date +%s); \
#     echo "[$<] 开始编译 at $$start_time"; \
# 	$(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) -c $< -o $@ ; \
# 	end_time=$$(date +%s); \
#     elapsed=$$((end_time - start_time)); \
#     echo "[$<] 编译完成，耗时 $$elapsed 秒" | tee -a compile_times.log
# # 添加 CUDA 源文件的扫描
# CUDA_SRCS := $(foreach dir,$(SRC_SUBDIRS),$(wildcard $(dir)/*.cu))
# CUDA_OBJS := $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(CUDA_SRCS))

# # CUDA 源文件的编译规则
# $(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
# 	@$(MKDIR_P) $(dir $@)
# 	@start_time=$$(date +%s); \
#     echo "[$<] 开始编译 at $$start_time" ; \
# 	$(NVCC) $(NVCCFLAGS) -dc $< -o $@ ; \
# 	end_time=$$(date +%s); \
#     elapsed=$$((end_time - start_time)); \
#     echo "[$<] 编译完成，耗时 $$elapsed 秒" | tee -a compile_times.log

# 共享源文件的编译规则
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@$(MKDIR_P) $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) -c $< -o $@
# 添加 CUDA 源文件的扫描
CUDA_SRCS := $(foreach dir,$(SRC_SUBDIRS),$(wildcard $(dir)/*.cu))
CUDA_OBJS := $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(CUDA_SRCS))

# CUDA 源文件的编译规则
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@$(MKDIR_P) $(dir $@)
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@ 

# 把所有 OBJ 汇总在一起方便后续链接
# ALL_OBJS := $(SHARED_OBJS) $(CUDA_OBJS)
# SHARED_OBJS += $(CUDA_OBJS)

# # 清理所有文件
# cleanall: 完全清除 build 目录
cleanall:
	rm -rf $(BUILD_DIR)
clean-dg:
	rm -rf $(OBJ_DIR)/dg
clean-atrix:
	rm -rf $(OBJ_DIR)/matrix

# clean <Module>: 清除指定模块对应的 .o 文件
# 示例：make clean DG   → 删除 build/obj/src/DG/
# 获取 src 下所有模块名（src/DG, src/Euler 等）


endif # COMMON_MK_ALREADY_INCLUDED
