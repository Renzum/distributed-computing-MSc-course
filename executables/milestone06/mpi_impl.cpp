#include <mpi.h>

#include <Kokkos_Core.hpp>

#include <lattice_boltzmann_types.hpp>
#include <mpi_data.hpp>

#include <mpi_impl_gpu_aware.hpp>
#include <mpi_impl_not_gpu_aware.hpp>

#include "mpi_impl.hpp"

namespace MPIImplementationFactory {

std::unique_ptr<MPIImplementation>
get_implementation(const int total_lattice_width,
                   const int total_lattice_height,
                   std::shared_ptr<MPIData> mpi_data, bool gpu_aware) {

    if (gpu_aware) {
        std::cout << "Using GPU aware MPI implementation." << std::endl;
        return std::unique_ptr<MPIImplementation>(new MPIGPUAware(
            total_lattice_width, total_lattice_height, mpi_data));
    }

    std::cout << "Using Non-GPU aware MPI implementation." << std::endl;
    return std::unique_ptr<MPIImplementation>(new MPINotGPUAware(
        total_lattice_width, total_lattice_height, mpi_data));
}

} // namespace MPIImplementationFactory
