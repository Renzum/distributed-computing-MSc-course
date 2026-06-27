#pragma once

#include <Kokkos_Core.hpp>
#include <fstream>

class LatticeBoltzman {

  private:
    int grid_width, grid_height;
    int iteration_count = 0;

    std::ofstream distribution_output_file;
    std::ofstream density_output_file;

    Kokkos::View<double ***> current_distribution;
    Kokkos::View<double ***> previous_distribution;

    Kokkos::View<double **> current_density;

    void calculate_current_density();
    void current_density_to_file();

  public:
    LatticeBoltzman(int, int);

    ~LatticeBoltzman();

    void calculate_next_step();
    void current_distribution_to_file();
};