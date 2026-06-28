#include <iostream>
#include <tuple>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <lattice-boltzman-class.hpp>

#include "main.hpp"

#define GRID_WIDTH 15
#define GRID_HEIGHT 10

#define GENERATIONS 10

int main(int argc, char *argv[]) {
    Kokkos::initialize(argc, argv);

    auto lb = LatticeBoltzman(GRID_WIDTH, GRID_HEIGHT);

    lb.lbm_step();
    lb.current_distribution_to_file();

    // {
    //     auto density_function = init_density_function();

    //     for (int i = 0; i < GENERATIONS; i++) {
    //         streaming_step(density_function);

    //         print_distribution(density_function, i);

    //         std::cout
    //             << std::endl; // Flush the buffer after each iteration
    //             print
    //     }
    // }

    Kokkos::finalize();
}
