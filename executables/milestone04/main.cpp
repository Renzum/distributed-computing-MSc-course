#include <cmath>
#include <iterator>

#include <Kokkos_Core.hpp>

#include <fmt/format.h>

#include <amplitude.hpp>
#include <lattice_boltzmann.hpp>
#include <output_functions.hpp>

#include "main.hpp"

int main(int argc, char *argv[]) {
    Kokkos::initialize(argc, argv);

    ms4();

    Kokkos::finalize();
}

void ms4() {
    const int lattice_width = 20;
    const int lattice_height = 20;

    std::vector<double> omegas{0.1, 0.2,  0.3, 0.4, 0.5, 0.8, 1.0,
                               1.1, 1.25, 1.4, 1.6, 1.8, 1.99};

    AmplitudeOutput amp_output("milestone4-amplitudes.yaml");

    LatticeBoltzmann::Functions lbm_functions(lattice_width, lattice_height);

    init_starting_distribution(lbm_functions);

    for (auto it = omegas.begin(); it != omegas.end(); it++) {
        auto lbm_copy = duplicate_functions(lbm_functions);

        simulate_and_calculate(lbm_copy, *it, amp_output);
    }
}

void init_starting_distribution(LatticeBoltzmann::Functions &lbm_functions) {
    const int lattice_width = lbm_functions.distribution_function.extent_int(0);
    const int lattice_height =
        lbm_functions.distribution_function.extent_int(1);

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
        "Initialize Density to 1",
        Kokkos::MDRangePolicy({0, 0}, {lattice_width, lattice_height}),
        KOKKOS_LAMBDA(const int &x, const int &y) {
            // Set density to 1 in all cells of the lattice
            lbm_functions.density_function(x, y) = 1;

            // Set the y component of the local average velocity to 0
            lbm_functions.local_average_velocity(x, y, 1) = 0;

            lbm_functions.local_average_velocity(x, y, 0) = u_x(y);
        });
}

// I originally wanted to do it via a copy constructor, but there were pointer
// bugs that I could not figure out and the amplitudes became a NaN
LatticeBoltzmann::Functions
duplicate_functions(const LatticeBoltzmann::Functions &source) {

    LatticeBoltzmann::Functions lbm_copy(
        source.distribution_function.extent_int(0),
        source.distribution_function.extent_int(1));

    Kokkos::deep_copy(lbm_copy.distribution_function,
                      source.distribution_function);
    Kokkos::deep_copy(lbm_copy.density_function, source.density_function);
    Kokkos::deep_copy(lbm_copy.local_average_velocity,
                      source.local_average_velocity);
    return lbm_copy;
}

void simulate_and_calculate(LatticeBoltzmann::Functions &lbm_functions,
                            const double &omega, AmplitudeOutput &amp_output) {
    const int lattice_height =
        lbm_functions.distribution_function.extent_int(1);
    DistributionFunctionOutput distribution_output(
        "milestone4-distribution.csv");
    DensityFunctionOutput density_output(fmt::format(
        "milestone4-density-omega={:f}-max_y={:d}.csv", omega, lattice_height));

    amp_output.new_data_set(omega, lattice_height);

    // Calculate the f_eq and set f to it
    LatticeBoltzmann::calculate_equilibrium_distribution(
        lbm_functions.distribution_function, lbm_functions.density_function,
        lbm_functions.local_average_velocity);

    const int iterations = 600;
    for (int i = 0; i < iterations; i++) {
        distribution_output.output(lbm_functions.distribution_function, i);
        density_output.output(lbm_functions.density_function, i);

#ifndef USE_SAMPLE_AT_MAX
        const long double amp = calculate_amplitude_via_projection(
            lbm_functions.local_average_velocity);
#else
        const long double amp =
            sample_amp_at_max(lbm_functions.local_average_velocity);
#endif

        amp_output.append_data_set(amp, i);

        LatticeBoltzmann::calculate_density(lbm_functions);
        LatticeBoltzmann::calculate_local_average_velocity(lbm_functions);
        LatticeBoltzmann::calculate_equilibrium_distribution(lbm_functions);
        LatticeBoltzmann::relax_distribution(lbm_functions, omega);
        LatticeBoltzmann::streaming_step(lbm_functions);
    }
}
