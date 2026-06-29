#include <iostream>
#include <tuple>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <lattice_boltzmann_class.hpp>

#include "main.hpp"

#define GRID_WIDTH 15
#define GRID_HEIGHT 10

#define GENERATIONS 10

int main(int argc, char *argv[]) {
    Kokkos::initialize(argc, argv);

    {
        auto lb = LatticeBoltzmann(GRID_WIDTH, GRID_HEIGHT, 0.8);

        for (int i = 0; i < GENERATIONS; i++) {
            lb.streaming_step();
            lb.current_distribution_to_file();
        }
    }

    Kokkos::finalize();
}
