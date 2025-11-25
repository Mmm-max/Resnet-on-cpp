#include <iostream>
#include "Tensor.hpp"

int main() {
    size_t N = 5;
    std::cout << "1. Prepare data..." << std::endl;
    std::vector<float> host_data(N);
    for(size_t i = 0; i < N; ++i) host_data[i] = i * 1.1f;
    std::cout << "=== test 2: GPU Tensor ====" << std::endl;
    {
        Tensor t_gpu(N, DeviceType::GPU);
        std::cout << "GPU tensor created successfully!" << std::endl;
        t_gpu.toDevice(host_data.data());
        std::cout << "coping data to GPU...." << std::endl;

        std::cout << "4. coping data to host from device..." << std::endl;
        std::vector<float> result_data(N);
        t_gpu.toHost(result_data.data());

        std::cout << "5. check result..." << std::endl;
        for (size_t i = 0; i < N; ++i) {
            std::cout << result_data[i] << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}