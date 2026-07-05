#include <cmath>

#include <Kokkos_Core.hpp>
#include <amplitude.hpp>
#include <output_functions.hpp>

#include <lattice_boltzmann.hpp>

void ms4() {
    const int lattice_width = 20;
    const int lattice_height = 20;

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

            if (x == 0) {
                lbm_functions.local_average_velocity(x, y, 0) =
                    epsilon * Kokkos::sin(two_pi * static_cast<double>(y) /
                                          static_cast<double>(lattice_height));
            } else {
                lbm_functions.local_average_velocity(x, y, 0) =
                    lbm_functions.local_average_velocity(0, y, 0);
            }
        });

    const double omega = 1.9;

    // Calculate the f_eq and set f to it
    LatticeBoltzmann::calculate_equilibrium_distribution(
        lbm_functions.distribution_function, lbm_functions.density_function,
        lbm_functions.local_average_velocity);

    DistributionFunctionOutput distribution_output(
        "milestone4-distribution.csv");
    DensityFunctionOutput density_output(
        "milestone4-density-omega=1.9-max_y=20.csv");
    AmplitudeOutput amplitude_output("milestone4-amplitude.yaml", omega,
                                     lattice_width);

    const int iterations = 600;
    for (int i = 0; i < iterations; i++) {
        distribution_output.output(lbm_functions.distribution_function, i);
        density_output.output(lbm_functions.density_function, i);
        amplitude_output.append(calculate_amplitude_via_project(
                                    lbm_functions.local_average_velocity),
                                i);
        // amplitude_output.output(
        //     double_sample_at_max(lbm_functions.local_average_velocity),
        //     omega, lattice_height, i);

        LatticeBoltzmann::calculate_density(lbm_functions);
        LatticeBoltzmann::calculate_local_average_velocity(lbm_functions);
        LatticeBoltzmann::calculate_equilibrium_distribution(lbm_functions);
        LatticeBoltzmann::relax_distribution(lbm_functions, omega);
        LatticeBoltzmann::streaming_step(lbm_functions);
    }
}

int main(int argc, char *argv[]) {
    Kokkos::initialize(argc, argv);

    ms4();

    Kokkos::finalize();
}
