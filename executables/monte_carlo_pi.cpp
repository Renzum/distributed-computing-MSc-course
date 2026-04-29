#include <iostream>
#include <ostream>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

int main(int argc, char *argv[]) {
    int a = 12;
    Kokkos::initialize(argc, argv);

    {
        Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
        const long n_samples = 100000000000;
        long hits = 0;

        Kokkos::parallel_reduce(
            "MonteCarloPi", n_samples,
            KOKKOS_LAMBDA(const int i, long &local_hits) {
                auto gen = random_pool.get_state();
                double x{gen.drand(0, 1)};
                double y{gen.drand(0, 1)};

                random_pool.free_state(gen);

                if (x * x + y * y <= 1.0) {
                    ++local_hits;
                }
            },
            hits);

        std::cout << "PI = " << 4.0 * static_cast<double>(hits) / n_samples
                  << std::endl;
    }

    Kokkos::finalize();
    return 0;
}