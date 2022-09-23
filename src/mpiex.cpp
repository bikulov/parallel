// sudo apt install mpich
// mpic++ mpiex.cpp && mpirun -n 11 ./a.out

#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>

#include <mpi.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int processRank = 0;
    int clusterSize = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
    MPI_Comm_size(MPI_COMM_WORLD, &clusterSize);

    std::cout << "rank " << processRank << " size " << std::endl;

    std::vector<double> a, b, c;
    auto N = 1'000'000;

    if (processRank != 0) {
        N /= (clusterSize - 1);
    }

    a.resize(N);
    b.resize(N);
    c.resize(N);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    if (processRank != 0) {
        for (size_t i = 0; i < N / clusterSize; ++i) {
            for (size_t j = 0; j < 1'000; ++j) {
                a[i] += std::log(j + 1 + i * 1.0);
                b[i] += std::sqrt(std::exp(i * 2.0)) + j;
                c[i] += std::sqrt(std::pow(a[i] + b[i], 3)) + j;
            }
        }

        MPI_Send((const void*)&a[0], N, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        MPI_Send((const void*)&b[0], N, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
        MPI_Send((const void*)&c[0], N, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
    } else {

        const auto chunkSize = N / (clusterSize - 1);

        MPI_Status mpiStatus;
        for (int i = 1; i < clusterSize; ++i) {
            MPI_Recv(&a[(i-1)*chunkSize], chunkSize, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &mpiStatus);
            MPI_Recv(&b[(i-1)*chunkSize], chunkSize, MPI_DOUBLE, i, 2, MPI_COMM_WORLD, &mpiStatus);
            MPI_Recv(&c[(i-1)*chunkSize], chunkSize, MPI_DOUBLE, i, 3, MPI_COMM_WORLD, &mpiStatus);
        }

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
