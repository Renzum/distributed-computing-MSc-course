#include <cmath>
#include <iostream>
#include <iterator>

#include <Kokkos_Core.hpp>

#include <fmt/format.h>

#include <amplitude.hpp>
#include <lattice_boltzmann.hpp>
#include <lattice_boltzmann_types.hpp>
#include <output_functions.hpp>

#include "main.hpp"

int main(int argc, char *argv[]) {
    Kokkos::initialize(argc, argv);

    ms4();

    Kokkos::finalize();
}

void ms4() {
    const int lattice_width = 50;
    const int lattice_height = 50;

    std::vector<double> omegas{0.1, 0.2,  0.3, 0.4, 0.5, 0.8, 1.0,
                               1.1, 1.25, 1.4, 1.6, 1.8, 1.99};

    AmplitudeOutput amp_output("milestone4-amplitudes.yaml");

    LatticeBoltzmann::DistributionFunction origin_distribution_function(
        "Original Distribution Function", lattice_width, lattice_height);
    LatticeBoltzmann::DensityFunction origin_density_function(
        "Original Density Function", lattice_width, lattice_height);
    LatticeBoltzmann::VelocityProfile origin_velocity_profile(
        "Original Velocity Profile Function", lattice_width, lattice_height);

    LatticeBoltzmann::DistributionFunction buffer_distribution_function(
        "Buffer Distribution Function", lattice_width, lattice_height);

    init_starting_distribution(origin_density_function, origin_velocity_profile,
                               lattice_width, lattice_height);

    for (const double &omega : omegas) {
        LatticeBoltzmann::DistributionFunction distribution_function(
            "Distribution Function", lattice_width, lattice_height);
        Kokkos::deep_copy(distribution_function, origin_distribution_function);

        LatticeBoltzmann::DensityFunction density_function(
            "Density Function", lattice_width, lattice_height);
        Kokkos::deep_copy(density_function, origin_density_function);

        LatticeBoltzmann::VelocityProfile velocity_profile(
            "Velocity Profile Function", lattice_width, lattice_height);
        Kokkos::deep_copy(velocity_profile, velocity_profile);

        std::cout << fmt::format(
                         "Simulating a {:d}x{:d} lattice with relaxation "
                         "parameter omega set to {:f}",
                         lattice_width, lattice_height, omega)
                  << std::endl;

        simulate_and_calculate(distribution_function,
                               buffer_distribution_function, density_function,
                               velocity_profile, omega, amp_output);
    }
}

void init_starting_distribution(
    LatticeBoltzmann::DensityFunction &density_function,
    LatticeBoltzmann::VelocityProfile &velocity_profile, int lattice_width,
    int lattice_height) {

    // According to my research and Claude, 0.01 is a common choice for epsilon
    constexpr double epsilon = 0.01;
    constexpr double two_pi = 2.0 * M_PI;

    Kokkos::View<double *> u_x("u_x Cache", lattice_height);

    // Precalculate u_x
    Kokkos::parallel_for(
        "Calculate u_x", Kokkos::RangePolicy(0, lattice_height),
        KOKKOS_LAMBDA(const int &y) {
            u_x(y) = epsilon * Kokkos::sin(two_pi * static_cast<double>(y) /
                                           static_cast<double>(lattice_height));
        });

    Kokkos::parallel_for(
        "Initialize Density to 0.5",
        Kokkos::MDRangePolicy({0, 0}, {lattice_width, lattice_height}),
        KOKKOS_LAMBDA(const int &x, const int &y) {
            // Set density to 1 in all cells of the lattice
            density_function(x, y) = 0.5;

            // Set the y component of the local average velocity to 0
            velocity_profile(x, y, 1) = 0;
            velocity_profile(x, y, 0) = u_x(y);
        });
}

void simulate_and_calculate(
    LatticeBoltzmann::DistributionFunction &distribution_function,
    LatticeBoltzmann::DistributionFunction &buffer_distribution_function,
    LatticeBoltzmann::DensityFunction &density_function,
    LatticeBoltzmann::VelocityProfile &velocity_profile, const double &omega,
    AmplitudeOutput &amp_output) {
    const int lattice_width = distribution_function.extent_int(0);
    const int lattice_height = distribution_function.extent_int(1);

    LatticeBoltzmann::Walls walls{
        LatticeBoltzmann::Wall{LatticeBoltzmann::BoundaryType::Periodic},
        LatticeBoltzmann::Wall{LatticeBoltzmann::BoundaryType::Periodic},
        LatticeBoltzmann::Wall{LatticeBoltzmann::BoundaryType::Periodic},
        LatticeBoltzmann::Wall{LatticeBoltzmann::BoundaryType::Periodic},
    };

    DistributionFunctionOutput distribution_output(
        "milestone4-distribution.csv");
    DensityFunctionOutput density_output(fmt::format(
        "milestone4-density-omega={:f}-max_y={:d}.csv", omega, lattice_height));
    LocalAverageVelocityFunctionOutput velocity_output(
        fmt::format("milestone4-velocity-field-omega={:f}-y={:d}.yaml", omega,
                    lattice_height),
        lattice_width, lattice_height, walls);

    amp_output.new_data_set(omega, lattice_height);

    // Calculate the f_eq and set f to it
    LatticeBoltzmann::calculate_equilibrium_distribution(
        distribution_function, density_function, velocity_profile);

    LatticeBoltzmann::DistributionFunction::HostMirror
        host_distribution_mirror =
            Kokkos::create_mirror_view(distribution_function);
    LatticeBoltzmann::DensityFunction::HostMirror host_density_mirror =
        Kokkos::create_mirror_view(density_function);
    LatticeBoltzmann::VelocityProfile::HostMirror host_velocity_profile_mirror =
        Kokkos::create_mirror_view(velocity_profile);

    const int iterations = 600;
    for (int i = 0; i < iterations; i++) {
        if (i % 25 == 0) {
            std::cout << fmt::format("Heartbeat {:d}/{:d}", i, iterations)
                      << std::endl;
        }

        Kokkos::deep_copy(host_distribution_mirror, distribution_function);
        Kokkos::deep_copy(host_density_mirror, density_function);

        if (i % 25 == 0) {
            std::cout << "Saving " << i << std::endl;
            Kokkos::deep_copy(host_velocity_profile_mirror, velocity_profile);
            velocity_output.add_timestep(host_velocity_profile_mirror, i);
        }

        distribution_output.output(host_distribution_mirror, i);
        density_output.output(host_density_mirror, i);

#ifndef USE_SAMPLE_AT_MAX
        const long double amp =
            calculate_amplitude_via_projection(velocity_profile);
#else
        const long double amp = sample_amp_at_max(velocity_profile);
#endif

        amp_output.append_data_set(amp, i);

        LatticeBoltzmann::calculate_density(density_function,
                                            distribution_function);
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
