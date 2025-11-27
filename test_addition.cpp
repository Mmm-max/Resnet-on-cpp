#include <iostream>
#include "Tensor.hpp"


int main() {
    std::cout << "testing add funct..." << std::endl;

    std::vector<float> a_data = {1, 2, 3, 4, 5, 6};
    std::vector<float> b_data = {10, 11, 12, 13, 14, 15};

    Tensor a(2, 3, DeviceType::GPU);
    a.toDevice(a_data.data());

    Tensor b(2, 3, DeviceType::GPU);
    b.toDevice(b_data.data());

    Tensor* c = a.add(b);
    std::vector<float> res(6);
    c->toHost(res.data());

    std::cout << "input res\n";
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << res[i * 3 + j] << " ";
        }
        std::cout << "\n";
    }
    delete c;
    return 0;
}