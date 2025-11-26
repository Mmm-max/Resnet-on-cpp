#include <iostream>
#include "Tensor.hpp"


int main() {
    size_t N = 10;
    std::vector<float> host_data = {-5, 2, -1, 10, 0, -3, 5, -100, 4, 8};

    Tensor t(N, DeviceType::GPU);
    Tensor t_cpu(N, DeviceType::CPU);
    t.toDevice(host_data.data());
    // t_cpu.toDevice(host_data.data());
    std::cout << "launcing relu...." << std::endl;
    t.relu();

    std::vector<float> result_data(N);
    t.toHost(result_data.data());
    // t_cpu.toHost(result_data.data());

    std::cout << "result: ";
    for (float x: result_data) {
        std::cout << x << " ";
    }
    std::cout << std::endl;

    return 0;
}