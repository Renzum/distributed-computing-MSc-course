#pragma once

#include <fstream>

#include <Kokkos_Core.hpp>

#include <output_functions.hpp>

class LatticeBoltzmann {
  public:
    /**
     * Creates a Lattice-Boltzmann simulation with a random initial distribution
     *
     * @param grid_width Width of the lattice
     * @param grid_height Height of the lattice
     * @param viscocity Viscocity of the fluid
     */
    LatticeBoltzmann(const int grid_width, const int grid_height,
                     const double viscocity);
    /**
     * Creates a Lattice-Boltzmann simulation with a provided distribution
     *
     * @param distribution_function
     * @param viscocity Viscocity of the fluid
     */
    LatticeBoltzmann(Kokkos::View<double ***> distribution_function,
                     const double viscocity);

    // Performs a full Lattice-Boltzmann step
    // Calculates density, local average velocity, equilibrium, performs
    // relaxation and then does streaming
    void lbm_step();

    // Performs the Lattice-Boltzmann streaming step
    void streaming_step();

    // Output all different functions into corresponding files
    void current_distribution_to_file();
    void current_density_to_file();
    void current_local_average_velocity_to_file();

  private:
    int grid_width, grid_height;
    int iteration_count = 0;

    double viscocity;

    DistributionFunctionOutput distribution_output_file{};
    DensityFunctionOutput density_output_file{};
    LocalAverageVelocityFunctionOutput local_average_velocity_output_file{};

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
};