#include <iostream>
#include "Tensor.hpp"


int main() {
    std::cout << "1. Conv2d test..." << std::endl;

    int H = 3; int W = 3;
    int C_in = 1; int C_out = 1;

    // Картинка 3x3 (просто цифры)
    std::vector<float> h_img = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    // Фильтр 3x3 (Единичный - только центр = 1, остальное 0)
    std::vector<float> h_filter = {
        0, 0, 0,
        0, 1, 0,
        0, 0, 0
    };

    Tensor img(H * W, 1, DeviceType::GPU);
    img.toDevice(h_img.data());
    
    Tensor filter(9, 1, DeviceType::GPU);
    filter.toDevice(h_filter.data());

    Tensor* res = img.conv2d(filter, C_in, C_out, H, W);
    
    std::vector<float> h_res(H * W);
    res->toHost(h_res.data());

    std::cout << "Input:\n1 2 3\n4 5 6\n7 8 9\n\nResult (Should be same):\n";
    for (int i = 0; i < H; ++i) {
        for (int j=0; j < W; ++j) {
            std::cout << h_res[i*W + j] << " ";
        }
        std::cout << "\n";
    }
    delete res;
    return 0;
}
