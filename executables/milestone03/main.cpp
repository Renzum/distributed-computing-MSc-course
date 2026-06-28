#include <iostream>
#include <tuple>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <lattice_boltzman_class.hpp>

#define GRID_WIDTH 15
#define GRID_HEIGHT 10

#define GENERATIONS 10

int main(int argc, char *argv[]) {
    Kokkos::initialize(argc, argv);

    {
        auto lb = LatticeBoltzman(GRID_WIDTH, GRID_HEIGHT, 1.0);

        lb.lbm_step();
        lb.current_distribution_to_file();
    }

    Kokkos::finalize();
}
