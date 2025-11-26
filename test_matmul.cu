#include <iostream>
#include "Tensor.hpp"


int main() {
    std::vector<float> h_A = {1, 2, 3, 4, 5, 6};
    std::vector<float> h_B = {1, 0, 0, 1, 1, 1};

    Tensor A(2, 3, DeviceType::GPU);
    A.toDevice(h_A.data());
    Tensor B(3, 2, DeviceType::GPU);
    B.toDevice(h_B.data());

    Tensor* C = A.matmul(B);

    std::vector<float> h_C(C->getSize());
    C->toHost(h_C.data());

    std::cout << "Result:\n ";
    size_t rows = C->getRows();
    size_t cols = C->getCols();
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            std::cout << h_C[i * rows + j] << " ";
        }
        std::cout << "\n";
    }
    delete C;
    return 0;
}