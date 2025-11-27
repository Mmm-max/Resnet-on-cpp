#ifndef TENOSR_HPP
#define TENOSR_HPP

#include <vector>
#include <string>
#include <cuda_runtime.h>

enum DeviceType {
    CPU,
    GPU
};

class Tensor {
    private:
        float* data_ptr;
        size_t rows_;
        size_t cols_;
        size_t size_;
        DeviceType device_;
    public:
        Tensor(size_t rows, size_t cols,  DeviceType device = DeviceType::CPU);
        ~Tensor();
        size_t getCols() const;
        size_t getRows() const;
        size_t getSize() const;
        DeviceType getDeviceType() const;
        float* getDataPtr() const;

        void toDevice(const float* src);

        void toHost(float* dst);

        Tensor* add(const Tensor& other);
        // matmul
        Tensor* matmul(const Tensor& other);
        // relu
        void relu();
        // conv2d
        Tensor* conv2d(const Tensor& weights, int in_channels, int out_channels, int height, int width);
};

#endif