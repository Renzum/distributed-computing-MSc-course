#pragma once

#include <fstream>
#include <string>

#include <Kokkos_Core.hpp>

#include <lattice_boltzmann.hpp>

long double calculate_amplitude_via_project(const Kokkos::View<double ***> &);
long double sample_amp_at_max(const Kokkos::View<double ***> &);

class AmplitudeOutput {
  private:
    std::fstream output_file;

  public:
    AmplitudeOutput(std::string file_name);
    ~AmplitudeOutput();

    void append_data_set(const long double &amplitude, const int &iteration);
    void new_data_set(const double &omega, const int &max_y);
};