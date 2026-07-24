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
                   std::shared_ptr<MPIData> mpi_data) {

#ifdef MPIX_CUDA_AWARE_SUPPORT
    if (MPIX_Query_cuda_support() == 1) {
        std::cout << "CUDA-aware MPI is available. Using the GPU Aware "
                     "implementation."
                  << std::endl;
        return std::unique_ptr<MPIImplementation>(new MPIGPUAware(
            total_lattice_width, total_lattice_height, mpi_data));
    }
#endif

    std::cout << "CUDA-aware MPI is not available. Using the Non - GPU Aware "
                 "implementation."
              << std::endl;
    return std::unique_ptr<MPIImplementation>(new MPINotGPUAware(
        total_lattice_width, total_lattice_height, mpi_data));
}

} // namespace MPIImplementationFactory
