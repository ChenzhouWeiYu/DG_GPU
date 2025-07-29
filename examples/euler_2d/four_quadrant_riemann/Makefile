# example/Euler2D/FourQuadrantRiemann/Makefile


.DEFAULT_GOAL := all

# 加载公共配置
ROOT_DIR := $(shell until [ -d "include" -o "$$PWD" = "/" ]; do cd ..; done; echo $$PWD)
include $(ROOT_DIR)/common.mk

# 当前目录源文件
LOCAL_SRCS := $(wildcard *.cpp)
LOCAL_OBJS := $(addprefix obj/, $(notdir $(LOCAL_SRCS:.cpp=.o)))

# CUDA 源文件支持
LOCAL_CUDA_SRCS := $(wildcard *.cu)
LOCAL_CUDA_OBJS := $(addprefix obj/, $(notdir $(LOCAL_CUDA_SRCS:.cu=.o)))

# 最终目标
TARGET := FourQuadrantRiemann

# .PHONY: prebuild_this
prebuild_this:
	@find . -name '*.d' -exec sed -i '/tmpxft/d' {} +

all: prebuild prebuild_this $(TARGET)

# 本地 C++ 编译规则
obj/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(CLANGFLAGS) $(INCLUDE_FLAGS) -c $< -o $@

# CUDA 编译规则
obj/%.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS)  -dc $< -o $@

# 链接
# $(TARGET): $(LOCAL_OBJS) $(CUDA_OBJS) $(SHARED_OBJS)
# 	$(CXX) $(LDFLAGS) $^ -o $@
$(TARGET): $(LOCAL_OBJS) $(LOCAL_CUDA_OBJS) $(CUDA_OBJS) $(SHARED_OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

# 清理规则
clean:
	rm -rf obj $(TARGET)

# 自动包含依赖文件
# -include $(LOCAL_OBJS:.o=.d)
# -include $(LOCAL_CUDA_OBJS:.o=.d)
# -include $(SHARED_OBJS:.o=.d)
# -include $(CUDA_OBJS:.o=.d)
