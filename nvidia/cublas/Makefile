TARGET := gemm
CUDA_PATH := ${CUDA_PATH}
NVCC := $(CUDA_PATH)/bin/nvcc
GPU_CODE := sm_86
GPU_ARCH := compute_86
CXXFLAGS := -std=c++11
bin_dir := bin

all: bin_

bin_: $(TARGET).cu bin_dir
	$(NVCC) $(CXXFLAGS) $(TARGET).cu --gpu-architecture=$(GPU_ARCH) --gpu-code=$(GPU_CODE) -o $(bin_dir)/$(TARGET) -lcublas

bin_dir:
	mkdir -p $(bin_dir) 

.phony: clean

clean:
	rm -rf $(bin_dir)

