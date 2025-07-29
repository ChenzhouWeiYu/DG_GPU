# Poiseuille/Makefile


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
TARGET := Sedov

all: $(TARGET)

# 本地 C++ 编译规则
obj/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) -c $< -o $@

# CUDA 编译规则
obj/%.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -ftime-report -c $< -o $@

# # 链接
$(TARGET): $(LOCAL_OBJS) $(SHARED_OBJS)
	$(CXX) $(LDFLAGS) $^ -o $@
# # 链接
# $(TARGET): $(LOCAL_OBJS) $(LOCAL_CUDA_OBJS) $(CUDA_OBJS) $(SHARED_OBJS)
# 	$(CXX) $(LDFLAGS) $^ -o $@

# 链接
# $(TARGET): $(LOCAL_OBJS) $(SHARED_OBJS)
# 	$(CXX) $(LDFLAGS) $^ -o $@

# 清理规则
clean:
	rm -rf obj $(TARGET)