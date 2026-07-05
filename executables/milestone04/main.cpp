#include <cmath>
#include <iostream>
#include <iterator>

#include <Kokkos_Core.hpp>
#include <amplitude.hpp>
#include <output_functions.hpp>

#include <fmt/format.h>

#include <lattice_boltzmann.hpp>

LatticeBoltzmann::Functions
initStartingDistribution(const int &lattice_width, const int &lattice_height) {
    // According to my research and Claude, 0.01 is a common choice for epsilon
    constexpr double epsilon = 0.01;
    constexpr double two_pi = 2.0 * M_PI;

    LatticeBoltzmann::Functions lbm_functions(lattice_width, lattice_height);

    // TODO: OPTIMIZE (All x y cells share the same velocity with x' y cells)
    Kokkos::parallel_for(
        "Initialize Density to 1",
        Kokkos::MDRangePolicy({0, 0}, {lattice_width, lattice_height}),
        KOKKOS_LAMBDA(const int &x, const int &y) {
            // Set density to 1 in all cells of the lattice
            lbm_functions.density_function(x, y) = 1;

            // Set the y component of the local average velocity to 0
            lbm_functions.local_average_velocity(x, y, 1) = 0;

            lbm_functions.local_average_velocity(x, y, 0) =
                epsilon * Kokkos::sin(two_pi * static_cast<double>(y) /
                                      static_cast<double>(lattice_height));
        });

    return lbm_functions;
}

void ms4(const double &omega, AmplitudeOutput &amp_output) {
    const int lattice_width = 20;
    const int lattice_height = 20;

    const int iterations = 600;
    DistributionFunctionOutput distribution_output(
        "milestone4-distribution.csv");
    DensityFunctionOutput density_output(fmt::format(
        "milestone4-density-omega={:f}-max_y={:d}.csv", omega, lattice_height));
    amp_output.new_data_set(omega, lattice_width);
    auto lbm_functions =
        initStartingDistribution(lattice_width, lattice_height);

    // Calculate the f_eq and set f to it
    LatticeBoltzmann::calculate_equilibrium_distribution(
        lbm_functions.distribution_function, lbm_functions.density_function,
        lbm_functions.local_average_velocity);

    for (int i = 0; i < iterations; i++) {
        distribution_output.output(lbm_functions.distribution_function, i);
        density_output.output(lbm_functions.density_function, i);
        const long double amp = calculate_amplitude_via_project(
            lbm_functions.local_average_velocity);

        // double_sample_at_max(lbm_functions.local_average_velocity);

        amp_output.append_data_set(amp, i);

        LatticeBoltzmann::calculate_density(lbm_functions);
        LatticeBoltzmann::calculate_local_average_velocity(lbm_functions);
        LatticeBoltzmann::calculate_equilibrium_distribution(lbm_functions);
        LatticeBoltzmann::relax_distribution(lbm_functions, omega);
        LatticeBoltzmann::streaming_step(lbm_functions);
    }
}

int main(int argc, char *argv[]) {
    Kokkos::initialize(argc, argv);

    std::vector<double> omegas{0.1, 0.2,  0.3, 0.4, 0.5, 0.8, 1.0,
                               1.1, 1.25, 1.4, 1.6, 1.8, 1.99};

    AmplitudeOutput amp_output("milestone4-amplitudes.yaml");
    // if (argc < 2) {
    //     std::cerr << "Please provide an omega command line argument."
    //               << std::endl;
    //     std::exit(1);
    // }

    for (auto it = omegas.begin(); it != omegas.end(); it++) {
        ms4(*it, amp_output);
    }

    Kokkos::finalize();
}
