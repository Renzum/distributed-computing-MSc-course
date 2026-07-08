#pragma once

#include <fstream>
#include <string>

#include <lattice_boltzmann.hpp>

class IOutput {
  protected:
    std::ofstream file;

  public:
    IOutput(std::string);
    ~IOutput();
};

class DistributionFunctionOutput : IOutput {
  public:
    inline DistributionFunctionOutput(std::string file_name)
        : IOutput(file_name) {
        file << "iteration,x,y,direction,value" << std::endl;
    };
    inline DistributionFunctionOutput()
        : DistributionFunctionOutput("distribution_function_data.csv") {};
    void output(const LatticeBoltzmann::HostDistributionMirror distribution,
                const int &iteration);
};

class DensityFunctionOutput : IOutput {
  public:
    inline DensityFunctionOutput(std::string file_name) : IOutput(file_name) {
        file << "iteration,x,y,density" << std::endl;
    }
    inline DensityFunctionOutput()
        : DensityFunctionOutput("density_function_data.csv") {};
    void output(const Kokkos::View<double **> &density, const int &iteration);
};

class LocalAverageVelocityFunctionOutput : IOutput {
  public:
    inline LocalAverageVelocityFunctionOutput(std::string file_name)
        : IOutput(file_name) {
        file << "iteration,x,y,velocity_x,velocity_y" << std::endl;
    }
    inline LocalAverageVelocityFunctionOutput()
        : LocalAverageVelocityFunctionOutput(
              "local_average_velocity_function_data.csv") {};
    void output(const Kokkos::View<double ***> &local_average_velocity,
                const int &iteration);
};