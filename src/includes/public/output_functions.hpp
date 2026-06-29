#pragma once

#include <fstream>
#include <string>

#include <Kokkos_Core.hpp>

class IOutput {
  protected:
    std::ofstream file;

  public:
    IOutput(std::string);
    ~IOutput();
};

class DistributionFunctionOutput : IOutput {
  public:
    DistributionFunctionOutput(std::string file_name);
    DistributionFunctionOutput()
        : DistributionFunctionOutput("distribution_function_data.csv") {};
    void output(const Kokkos::View<double ***> &distribution,
                const int &iteration);
};

class DensityFunctionOutput : IOutput {
  public:
    DensityFunctionOutput(std::string file_name);
    DensityFunctionOutput()
        : DensityFunctionOutput("density_function_data.csv") {};
    void output(const Kokkos::View<double **> &density, const int &iteration);
};

class LocalAverageVelocityFunctionOutput : IOutput {
  public:
    LocalAverageVelocityFunctionOutput(std::string file_name);
    LocalAverageVelocityFunctionOutput()
        : LocalAverageVelocityFunctionOutput(
              "local_average_velocity_function_data.csv") {};
    void output(const Kokkos::View<double ***> &local_average_velocity,
                const int &iteration);
};