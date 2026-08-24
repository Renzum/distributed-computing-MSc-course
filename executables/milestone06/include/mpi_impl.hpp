#pragma once

#include <iostream>
#include <memory>

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <mpi-ext.h>

#include <lattice_boltzmann_types.hpp>
#include <mpi_data.hpp>

class MPIImplementation {
  protected:
    using GPUBuffer = Kokkos::View<double *[TOTAL_DIRECTIONS]>;

  public:
    virtual void
    communicate_left(const LatticeBoltzmann::DistributionFunction &) = 0;
    virtual void
    communicate_right(const LatticeBoltzmann::DistributionFunction &) = 0;
    virtual void
    communicate_down(const LatticeBoltzmann::DistributionFunction &) = 0;
    virtual void
    communicate_up(const LatticeBoltzmann::DistributionFunction &) = 0;
};

namespace MPIImplementationFactory {

std::unique_ptr<MPIImplementation>
get_implementation(const int total_lattice_width,
                   const int total_lattice_height,
                   std::shared_ptr<MPIData> mpi_data, bool gpu_aware);

} // namespace MPIImplementationFactory
