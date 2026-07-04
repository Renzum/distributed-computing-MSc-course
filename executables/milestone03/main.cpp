#include <iostream>
#include <tuple>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <direction_definitions.hpp>
#include <distribution_initializers.hpp>
#include <lattice_boltzmann.hpp>
#include <output_functions.hpp>

#define GRID_WIDTH 15
#define GRID_HEIGHT 10

#define GENERATIONS 10

void uniformWithHigherDensity() {

    // Set a low relaxation rate to observe the density change over a long
    // number of step iterations
    const double omega = 0.5;

    auto distribution_output = DistributionFunctionOutput{};
    auto density_output = DensityFunctionOutput{
        "milestone03_uniform_density_with_higher_center.csv"};

    auto lbm_functions = LatticeBoltzmann::Functions{GRID_WIDTH, GRID_HEIGHT};

    LatticeBoltzmann::DistributionInitializers::
        uniform_density_with_higher_center(lbm_functions, 1.0, 1.1);

    for (int i = 0; i < 100; i++) {
        distribution_output.output(lbm_functions.distribution_function, i);

        LatticeBoltzmann::calculate_density(lbm_functions);

        density_output.output(lbm_functions.density_function, i);

        LatticeBoltzmann::calculate_local_average_velocity(lbm_functions);
        LatticeBoltzmann::calculate_equilibrium_distribution(lbm_functions);
        LatticeBoltzmann::relax_distribution(lbm_functions, omega);
        LatticeBoltzmann::streaming_step(lbm_functions);
    }
}

void randomLongRun() {
    // Set a low relaxation rate to observe the density change over a long
    // number of step iterations
    const double omega = 0.5;

    auto distribution_output = DistributionFunctionOutput{
        "milestone03_random_density_long_run-distribution.csv"};
    auto density_output = DensityFunctionOutput{
        "milestone03_random_density_long_run-density.csv"};
    auto local_average_velocity_output = LocalAverageVelocityFunctionOutput{
        "milestone03_random_density_long_run-local_average_velocity.csv"};

    auto lbm_functions = LatticeBoltzmann::Functions{GRID_WIDTH, GRID_HEIGHT};

    LatticeBoltzmann::DistributionInitializers::random_density(lbm_functions);

    for (int i = 0; i < 1000; i++) {
        distribution_output.output(lbm_functions.distribution_function, i);

        LatticeBoltzmann::calculate_density(lbm_functions);

        density_output.output(lbm_functions.density_function, i);

        LatticeBoltzmann::calculate_local_average_velocity(lbm_functions);
        local_average_velocity_output.output(
            lbm_functions.local_average_velocity, i);

        LatticeBoltzmann::calculate_equilibrium_distribution(lbm_functions);
        LatticeBoltzmann::relax_distribution(lbm_functions, omega);
        LatticeBoltzmann::streaming_step(lbm_functions);
    }
}

int main(int argc, char *argv[]) {
    Kokkos::initialize(argc, argv);

    uniformWithHigherDensity();
    randomLongRun();

    Kokkos::finalize();
}
