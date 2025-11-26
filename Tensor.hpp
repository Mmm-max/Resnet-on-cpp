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
        size_t size_;
        DeviceType device_;
    public:
        Tensor(const size_t tensor_size, DeviceType device=DeviceType::CPU);
        ~Tensor();
        size_t getSize() const;
        DeviceType getDeviceType() const;
        float* data();

        void toDevice(const float* src);

        void toHost(float* dst) const;

        // relu
        void relu();
};

#endif