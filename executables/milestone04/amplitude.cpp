#include <fstream>
#include <iomanip>
#include <string>

#include <Kokkos_Core.hpp>

#include <lattice_boltzmann_types.hpp>

#include "amplitude.hpp"

AmplitudeOutput::AmplitudeOutput(std::string file_name)
    : output_file(file_name, std::ios::out) {};

AmplitudeOutput::~AmplitudeOutput() {
    output_file.close();
}

void AmplitudeOutput::append_data_set(const long double &amplitude,
                                      const int &time_step) {
    output_file << "    - time_step: " << time_step << "\n";
    output_file << std::setprecision(17) << "      amplitude: " << amplitude
                << std::endl;
}

void AmplitudeOutput::new_data_set(const double &omega, const int &max_y) {

    output_file << "- omega: " << omega << "\n";
    output_file << "  max_y: " << max_y << "\n";
    output_file << "  amplitudes: " << std::endl;
}

long double
sample_amp_at_max(const LatticeBoltzmann::VelocityProfile &velocity_profile) {
    const int max_y =
        velocity_profile.extent_int(1); // Get the y size of the lattice
    const long double zeta = 2.0 * M_PI / max_y; // 2pi / L_y

    long double amplitude =
        velocity_profile(0, static_cast<int>(max_y / 4.0), 0);
    return amplitude;
}

// Calculate the Fourier amplitude of the u_x
long double calculate_amplitude_via_projection(
    const LatticeBoltzmann::VelocityProfile &velocity_profile) {

    const int max_y =
        velocity_profile.extent_int(1); // Get the y size of the lattice

    const long double zeta = 2.0 * M_PI / max_y; // 2pi / L_y

    long double amplitude = 0;

    // We know that for a given y, u_x(y) will be the same for all x
    // Thus we only go through x = 0 and reduce the values to a sum
    Kokkos::parallel_reduce(
        "Sine Mode Projection", Kokkos::RangePolicy(0, max_y),
        KOKKOS_LAMBDA(const int &y_j, long double &local_vel_sum) {
            const long double sin_component = std::sin(zeta * y_j);
            local_vel_sum +=
                static_cast<long double>(velocity_profile(0, y_j, 0)) *
                sin_component;
        },
        amplitude);

    const long double C1 = 2.0 / static_cast<double>(max_y); // 2 / L_y
    amplitude *= C1;

    return amplitude;
}