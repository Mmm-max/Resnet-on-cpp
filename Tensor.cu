#include "Tensor.hpp"
#include <iostream>
#include <cstdio>

#define CHECK_CUDA(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "Cuda error: %s at line %d in file %s\n", cudaGetErrorString(err), __LINE__, __FILE__); \
        exit(1); \
    } \
}

Tensor::Tensor(const size_t size, DeviceType device) {
    size_ = size;
    device_ = device;
    data_ptr = nullptr;
    if (device_ == DeviceType::CPU) {
        std::cout << "Allocating " << size << " float on CPU" << std::endl;
        data_ptr = new float[size_];
    } else if (device_ == DeviceType::GPU) {
        std::cout << "Allocating " << size_ << " float on GPU" << std::endl;
        CHECK_CUDA(cudaMalloc((void**)&data_ptr, size_ * sizeof(float)));
    }
}

Tensor::~Tensor() {
    if (data_ptr) {
        if (device_ == DeviceType::CPU) {
            std::cout << "Freeing CPU memory" << std::endl;
            delete[] data_ptr;
        } else if (device_ == DeviceType::GPU) {
            CHECK_CUDA(cudaFree(data_ptr));
        }
        data_ptr = nullptr;
    }
}


size_t Tensor::getSize() const {
    return size_;
}

DeviceType Tensor::getDeviceType() const {
    return deviceType;
}

float* Tensor::data() {
    return data_ptr;
}