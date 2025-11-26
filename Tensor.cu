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

__global__ void matmul_kernel(float* A, float* B, float* C, int M, int N, int K) {
    // вычисляем произведение матриц MxK and KxN
    // результирующая матрица MxN
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;

        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

Tensor::Tensor(size_t rows, size_t cols, DeviceType device) {
    rows_ = rows;
    cols_ = cols;
    size_ = rows_ * cols_; 
    device_ = device;
    data_ptr = nullptr;
    if (device_ == DeviceType::CPU) {
        std::cout << "Allocating " << size_ << " float on CPU" << std::endl;
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


size_t Tensor::getCols() const {
    return cols_;
}

size_t Tensor::getRows() const {
    return rows_;
}

size_t Tensor::getSize() const {
    return size_;
}

DeviceType Tensor::getDeviceType() const {
    return device_;
}

float* Tensor::getDataPtr() const {
    return data_ptr;
}

void Tensor::toDevice(const float* src) {
    if (device_ == DeviceType::CPU) {
        std::copy(src, src + size_, data_ptr);
    } else if (device_ == DeviceType::GPU) {
        CHECK_CUDA(cudaMemcpy(data_ptr, src, size_ * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void Tensor::toHost(float* dst) {
    if (device_ == DeviceType::CPU) {
        std::copy(data_ptr, data_ptr + size_, dst);
    } else if (device_ == DeviceType::GPU) {
        CHECK_CUDA(cudaMemcpy(dst, data_ptr, size_ * sizeof(float), cudaMemcpyDeviceToHost));
    }
}

// matmul

Tensor* Tensor::matmul(const Tensor& other) {
    if (cols_ != other.getRows()) {
        throw std::runtime_error("Matmul shape mismatch!");
    }
    if (device_ != DeviceType::GPU || other.getDeviceType() != DeviceType::GPU) {
        throw std::runtime_error("Matmul currently supports only on GPU!");
    }
    int M = rows_;
    int K = cols_;
    int N = other.getCols();

    Tensor* C = new Tensor(M, N, DeviceType::GPU);

    dim3 threadsPerBlock(16, 16);

    dim3 blocksPerGrid(
        (N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (M + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    std::cout << "launching matmul: " << M << "x" << K <<
        "times " << K << "x" << N << std::endl;
    
    matmul_kernel<<<blocksPerGrid, threadsPerBlock>>> (
        data_ptr,
        other.getDataPtr(),
        C->getDataPtr(),
        M, N, K
    );

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());

    return C;
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

