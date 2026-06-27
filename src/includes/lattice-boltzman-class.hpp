#pragma once

#include <Kokkos_Core.hpp>
#include <fstream>

class LatticeBoltzman {

  private:
    int grid_width, grid_height;
    int iteration_count = 0;

    double viscocity;

    std::ofstream distribution_output_file;
    std::ofstream density_output_file;
    std::ofstream local_average_velocity_output_file;

    Kokkos::View<double ***> distribution_function;
    Kokkos::View<double ***> buffer_distribution_view;

    Kokkos::View<double **> density_function;
    Kokkos::View<double ***> local_average_velocity;

    void calculate_density_function();
    void current_density_to_file();
    void calculate_local_average_velocity_function();
    void calculate_equilibrium_distribution_function();

  public:
    LatticeBoltzman(int, int, double);

    ~LatticeBoltzman();

    void calculate_next_step();

    void current_distribution_to_file();
    void current_density_to_file();
    void current_local_average_velocity_to_file();
};