# Poiseuille/Makefile

# 加载公共配置
ROOT_DIR := $(shell until [ -d "include" -o "$$PWD" = "/" ]; do cd ..; done; echo $$PWD)
include $(ROOT_DIR)/common.mk

# 当前目录源文件
LOCAL_SRCS := $(wildcard *.cpp)
LOCAL_OBJS := $(addprefix obj/, $(notdir $(LOCAL_SRCS:.cpp=.o)))

# 最终目标
TARGET := Riemann1D

all: $(TARGET)

# 本地编译规则
obj/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) -c $< -o $@

# 链接最终程序
$(TARGET): $(LOCAL_OBJS) $(SHARED_OBJS)
	$(CXX) $(LDFLAGS) $^ -o $@

# 清理规则
clean:
	rm -rf obj $(TARGET)