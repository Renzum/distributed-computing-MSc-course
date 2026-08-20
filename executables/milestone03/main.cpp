#include <iostream>
#include <tuple>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <direction_definitions.hpp>
#include <distribution_initializers.hpp>
#include <lattice_boltzmann.hpp>
#include <lattice_boltzmann_types.hpp>
#include <output_functions.hpp>

#define GRID_WIDTH 15
#define GRID_HEIGHT 10

#define GENERATIONS 10

void run(LatticeBoltzmann::DistributionFunction &distribution_function,
         LatticeBoltzmann::DensityFunction &density_function,
         LatticeBoltzmann::VelocityProfile &velocity_profile,
         DensityFunctionOutput &density_output,
         DistributionFunctionOutput &distribution_output) {
    // Set a low relaxation rate to observe the density change over a long
    // number of step iterations
    const double omega = 0.5;

    LatticeBoltzmann::DistributionFunction buffer_distribution_function(
        "Buffer Distribution Function", GRID_WIDTH, GRID_HEIGHT);

    LatticeBoltzmann::DistributionFunction::HostMirror
        host_distribution_mirror =
            Kokkos::create_mirror_view(distribution_function);
    LatticeBoltzmann::DensityFunction::HostMirror host_density_mirror =
        Kokkos::create_mirror_view(density_function);

    for (int i = 0; i < 100; i++) {
        Kokkos::deep_copy(host_distribution_mirror, distribution_function);
        distribution_output.output(host_distribution_mirror, i);

        LatticeBoltzmann::calculate_density(density_function,
                                            distribution_function);

        Kokkos::deep_copy(host_density_mirror, density_function);
        density_output.output(host_density_mirror, i);

        LatticeBoltzmann::calculate_local_average_velocity(
            velocity_profile, distribution_function, density_function);
        LatticeBoltzmann::calculate_equilibrium_distribution(
            buffer_distribution_function, density_function, velocity_profile);
        LatticeBoltzmann::relax_distribution(
            distribution_function, buffer_distribution_function, omega);
        LatticeBoltzmann::streaming_step_with_periodic_bounds(
            buffer_distribution_function, distribution_function);
    }
}

void uniformWithHigherDensity() {

    auto distribution_output = DistributionFunctionOutput{};
    auto density_output = DensityFunctionOutput{
        "milestone03_uniform_density_with_higher_center.csv"};

    LatticeBoltzmann::DistributionFunction distribution_function(
        "Distribution Function", GRID_WIDTH, GRID_HEIGHT);
    LatticeBoltzmann::DensityFunction density_function("Density Function",
                                                       GRID_WIDTH, GRID_HEIGHT);
    LatticeBoltzmann::VelocityProfile velocity_profile("Velocity Profile",
                                                       GRID_WIDTH, GRID_HEIGHT);

    LatticeBoltzmann::DistributionInitializers::
        uniform_density_with_higher_center(distribution_function, 1.0, 1.1);

    run(distribution_function, density_function, velocity_profile,
        density_output, distribution_output);
}

void randomLongRun() {
    // Set a low relaxation rate to observe the density change over a long
    // number of step iterations
    const double omega = 0.5;

    auto distribution_output = DistributionFunctionOutput{
        "milestone03_random_density_long_run-distribution.csv"};
    auto density_output = DensityFunctionOutput{
        "milestone03_random_density_long_run-density.csv"};

    LatticeBoltzmann::DistributionFunction distribution_function(
        "Distribution Function", GRID_WIDTH, GRID_HEIGHT);
    LatticeBoltzmann::DensityFunction density_function("Density Function",
                                                       GRID_WIDTH, GRID_HEIGHT);
    LatticeBoltzmann::VelocityProfile velocity_profile("Velocity Profile",
                                                       GRID_WIDTH, GRID_HEIGHT);

    LatticeBoltzmann::DistributionInitializers::random_density(
        distribution_function);

    run(distribution_function, density_function, velocity_profile,
        density_output, distribution_output);
}

int main(int argc, char *argv[]) {
    Kokkos::initialize(argc, argv);

    uniformWithHigherDensity();
    randomLongRun();

    Kokkos::finalize();
}
