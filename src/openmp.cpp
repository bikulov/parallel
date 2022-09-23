// adapted from https://ru.wikipedia.org/wiki/OpenMP
// sudo apt update && sudo apt install libomp-dev clang
// clang++ -std=c++14 -fopenmp=libomp openmp.cpp -o openmp && ./openmp

#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>

#include <omp.h>

int main(int /*argc*/, char ** /*argv*/) {
    std::vector<double> a, b, c;
    const auto N = 1'000'000;

    a.resize(N);
    b.resize(N);
    c.resize(N);

    omp_set_dynamic(0);      // запретить библиотеке openmp менять число потоков во время исполнения
    omp_set_num_threads(10); // установить число потоков в 10

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    #pragma omp parallel for
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < 1'000; ++j) {
            a[i] += std::log(j + 1 + i * 1.0);
            b[i] += std::sqrt(std::exp(i * 2.0)) + j;
            c[i] += std::sqrt(std::pow(a[i] + b[i], 3)) + j;
        }
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
    return 0;
}
