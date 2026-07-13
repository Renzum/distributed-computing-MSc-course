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
    void output(
        const LatticeBoltzmann::DistributionFunction::HostMirror distribution,
        const int &iteration);
};

class DensityFunctionOutput : IOutput {
  public:
    inline DensityFunctionOutput(std::string file_name) : IOutput(file_name) {
        file << "iteration,x,y,density" << std::endl;
    }
    inline DensityFunctionOutput()
        : DensityFunctionOutput("density_function_data.csv") {};
    void output(const LatticeBoltzmann::DensityFunction::HostMirror density,
                const int &iteration);
};

class LocalAverageVelocityFunctionOutput {
  private:
    int lattice_width;
    int lattice_height;

    std::ofstream file;

  public:
    LocalAverageVelocityFunctionOutput(std::string file_name,
                                       const int lattice_width,
                                       const int lattice_height,
                                       const LatticeBoltzmann::Walls &walls);

    void add_timestep(const LatticeBoltzmann::VelocityProfile::HostMirror
                          local_average_velocity,
                      const int &iteration);
};