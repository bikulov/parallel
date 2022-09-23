// Adapted from https://stackoverflow.com/a/11229853
// clang++ --std=c++14 -lpthread thread.cpp -o thread && ./thread

#include <string>
#include <vector>
#include <iostream>
#include <thread>


void long_task(std::vector<double>& a) {
    for (auto& aa : a) {
        aa = 3 * 3;
    }
}


int main(int, char**) {
    std::vector<double> a;
    a.resize(100);

    std::thread th(long_task, std::ref(a));
    th.join();
    for (auto i = 0; i < 100; ++i) {
        std::cout << a[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}