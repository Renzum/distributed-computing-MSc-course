#include <fstream>
#include <iomanip>
#include <string>

#include <Kokkos_Core.hpp>

#include "amplitude.hpp"

AmplitudeOutput::AmplitudeOutput(std::string file_name)
    : output_file(file_name, std::ios::out) {

    // Write the csv spec to the file
    output_file << "time_step,amplitude,omega,max_y" << std::endl;
};

AmplitudeOutput::~AmplitudeOutput() {
    output_file.close();
}

void AmplitudeOutput::output(const long double &amplitude, const double &omega,
                             const int &max_y, const int &time_step) {
    output_file << time_step << "," << amplitude << "," << omega << "," << max_y
                << std::endl;
}

// Calculate the Fourier amplitude of the u_x
long double
calculate_amplitude_via_project(const Kokkos::View<double ***> &local_avg_vel) {

    const int max_y =
        local_avg_vel.extent_int(1); // Get the y size of the lattice

    const long double zeta = 2.0 * M_PI / max_y; // 2pi / L_y

    long double amplitude = 0;

    // We know that for a given y, u_x(y) will be the same for all x
    // Thus we only go through x = 0 and reduce the values to a sum
    Kokkos::parallel_reduce(
        "Sine Mode Projection", Kokkos::RangePolicy(0, max_y),
        KOKKOS_LAMBDA(const int &y_j, long double &local_vel_sum) {
            const long double sin_component = std::sin(zeta * y_j);
            local_vel_sum +=
                static_cast<long double>(local_avg_vel(0, y_j, 0)) *
                sin_component;
        },
        amplitude);

    const long double C1 = 2.0 / static_cast<double>(max_y); // 2 / L_y
    amplitude *= C1;

    return amplitude;
}