#pragma once

#include <fstream>

#include <Kokkos_Core.hpp>

class LatticeBoltzman {

  private:
    int grid_width, grid_height;
    int iteration_count = 0;

    double viscocity;

    std::ofstream distribution_output_file;
    std::ofstream density_output_file;
    std::ofstream local_average_velocity_output_file;

    // We use a buffer view for storing the equilibrium and performing the
    // streaming step
    Kokkos::View<double ***> buffer_distribution_view;

    Kokkos::View<double ***> distribution_function;
    Kokkos::View<double **> density_function;

    // ACHTUNG: The local average velocity is three dimensional but it's
    // essentially a 2D matrix that points to a vector represented as an array
    // of size 2
    Kokkos::View<double ***> local_average_velocity;

    // Wrapper functions that pass the member variables into the static
    // implementation functions
    void calculate_density_function();
    void calculate_local_average_velocity_function();
    void calculate_equilibrium_distribution_function();
    void calculate_relaxed_distribution_function();

  public:
    LatticeBoltzman(int, int, double);

    ~LatticeBoltzman();

    // Performs a full Lattice-Boltzman step
    // Calculates density, local average velocity, equilibrium, performs
    // relaxation and then does streaming
    void lbm_step();

    // Output all different functions into corresponding files
    void current_distribution_to_file();
    void current_density_to_file();
    void current_local_average_velocity_to_file();
};