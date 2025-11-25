#include <iostream>
#include "Tensor.hpp"

int main() {
    size_t N = 1024;

    std::cout << "=== test 1: CPU Tensor === " << std::endl;
    {
        Tensor t_cpu(N, DeviceType:CPU);
    }
    std::cout << "=== test 2: GPU Tensor ====" << std::endl;
    {
        Tensor t_gpu(N, DeviceType::GPU);
        std::cout << "GPU tensor created successfully!" << std::endl;
    }
    return 0;
}