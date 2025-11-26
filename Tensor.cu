#include "Tensor.hpp"
#include <iostream>
#include <cstdio>
#include <algorithm>

#define CHECK_CUDA(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "Cuda error: %s at line %d in file %s\n", cudaGetErrorString(err), __LINE__, __FILE__); \
        exit(1); \
    } \
}

__global__ void relu_kernel(float* data, int size) {
    // blockIdx.x - номер блока
    // blockDim.x - количество потоков
    // threadIdx.x - номер потока внутри блока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float val = data[idx];
        if (val < 0.0f) {
            data[idx] = 0.0f;
        }
    }
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
            std::cout << "Freeing GPU memory" << std::endl;
            CHECK_CUDA(cudaFree(data_ptr));
        }
        data_ptr = nullptr;
    }
}


size_t Tensor::getSize() const {
    return size_;
}

DeviceType Tensor::getDeviceType() const {
    return device_;
}

float* Tensor::data() {
    return data_ptr;
}

void Tensor::toDevice(const float* src) {
    if (device_ == DeviceType::CPU) {
        std::copy(src, src + size_, data_ptr);
    } else if (device_ == DeviceType::GPU) {
        CHECK_CUDA(cudaMemcpy(data_ptr, src, size_ * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void Tensor::toHost(float* dst) const {
    if (device_ == DeviceType::CPU) {
        std::copy(data_ptr, data_ptr + size_, dst);
    } else if (device_ == DeviceType::GPU) {
        CHECK_CUDA(cudaMemcpy(dst, data_ptr, size_ * sizeof(float), cudaMemcpyDeviceToHost));
    }
}

void Tensor::relu() {
    if (device_ == DeviceType::CPU){
        for (size_t i = 0; i < size_; ++i) {
            // std::cout << "relu, ind: " << i << " val: " << data_ptr[i] << std::endl;
            if (data_ptr[i] < 0.0f) {
                // std::cout << "relu lower val!! " << data_ptr[i] << std::endl;
                data_ptr[i] = 0.0f;
            }
        }
    } else if (device_ == DeviceType::GPU) {
        std::cout << "GPU relu launching..." << std::endl;
        int threadsPerBlock = 256;
        int blocksPerGrid = (size_ + threadsPerBlock - 1) / threadsPerBlock;

        relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(data_ptr, size_);
        CHECK_CUDA(cudaDeviceSynchronize()); // ожидание всех ядер
        CHECK_CUDA(cudaGetLastError()); // проверка, не упало ли ядро
    }
}

