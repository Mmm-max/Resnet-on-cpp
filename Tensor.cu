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


__global__ void add_kernel(float* A, float* B, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] + B[idx];
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

__global__ void conv2d_kernel_3x3(float* I, float* W, float* O,
                                  int C_in, int C_out, int H, int Width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int out_ch = blockIdx.z;

    if (row < H &&  col < Width && out_ch < C_out) {
        float sum = 0.0f;
        // проходим по каждому каналу
        for (int in_ch = 0; in_ch < C_in; ++in_ch) {
            // перебераем окноо 3x3 от -1 до 1
            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    int current_row = row + i;
                    int current_col = col + j;

                    // padding = 1
                    if (current_row >= 0 && current_row < H &&
                        current_col >= 0 && current_col < Width) {
                            int input_idx = (in_ch * H + current_row) * Width + current_col;
                            int weight_idx = ((out_ch * C_in + in_ch) * 3 + (i + 1)) * 3  + (j + 1);
                            sum += I[input_idx] * W[weight_idx];
                        }
                }
            }
        }
        int output_idx = (out_ch * H + row) * Width + col;
        O[output_idx] = sum;
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


size_t Tensor::getCols() const  {
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

// add

Tensor* Tensor::add(const Tensor& other) {
    if (cols_ != other.getCols() && rows_ != other.getRows()) {
        throw std::runtime_error("addition shape mismatch!");
    }
    if (device_ != DeviceType::GPU || other.getDeviceType() != DeviceType::GPU) {
        throw std::runtime_error("addition currently supprosts only on GPU!");
    }

    Tensor* C = new Tensor(rows_, cols_, DeviceType::GPU);
    int ThreadsPerBlock = 256;
    int blocksPerGrid = (size_ + ThreadsPerBlock - 1) / ThreadsPerBlock;


    add_kernel<<<blocksPerGrid, ThreadsPerBlock>>>(
        data_ptr,
        other.getDataPtr(),
        C->getDataPtr(),
        size_
    );
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());

    return C;
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


Tensor* Tensor::conv2d(const Tensor& weights, int in_channels, int out_channels,
                    int height, int width) {
    size_t out_size = out_channels * height * width;
    // NOTE: заглушка для трехмерного вектора
    Tensor* Output = new Tensor(out_size, 1, DeviceType::GPU);

    dim3 threadsPerBlock(16, 16); // пикселей в блоке
    dim3 blocksPerGrid(
        (width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y,
        out_channels
    );

    std::cout << "Launching Conv2D 3x3..." << std::endl;

    conv2d_kernel_3x3<<<blocksPerGrid, threadsPerBlock>>>(
        data_ptr,
        weights.getDataPtr(),
        Output->getDataPtr(),
        in_channels, out_channels, height, width
    );

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());

    return Output;
}