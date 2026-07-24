#pragma once

#include <iostream>
#include <memory>

#include <mpi.h>

#include <mpi-ext.h>

#include <mpi_data.hpp>

class MPIImplementation {
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

class MPINotGPUAware : public MPIImplementation {
  public:
    int lattice_width, lattice_height;
    std::shared_ptr<MPIData> mpi_data;

    // Buffers for moving slices from GPU to CPU and vice versa
    using GPUBuffer = Kokkos::View<double *[TOTAL_DIRECTIONS]>;
    GPUBuffer left_right_gpu_buffer;
    int left_right_buffer_len;

    GPUBuffer down_up_gpu_buffer;
    int down_up_buffer_len;

    // Host Space buffers for sending without GPU Aware MPI
    using GPUBufferMirror = GPUBuffer::HostMirror;
    GPUBufferMirror left_right_send_buffer, left_right_recv_buffer;
    GPUBufferMirror down_up_send_buffer, down_up_recv_buffer;

    MPINotGPUAware(const int lattice_width, const int lattice_height,
                   std::shared_ptr<MPIData> mpi_data);

    void
    communicate_left(const LatticeBoltzmann::DistributionFunction &) override;
    void communicate_right(const LatticeBoltzmann::DistributionFunction &

                           ) override;
    void
    communicate_down(const LatticeBoltzmann::DistributionFunction &) override;
    void
    communicate_up(const LatticeBoltzmann::DistributionFunction &) override;
};

// class MPIGPUAware : public MPIImplementation {
//   public:
//     MPIGPUAware(const int lattice_width, const int lattice_height);

//     void communicate_left(const LatticeBoltzmann::DistributionFunction &,
//                           const int left_neighbor,
//                           const int right_neighbor) override;
//     void communicate_right(const LatticeBoltzmann::DistributionFunction &,
//                            const int left_neighbor,
//                            const int right_neighbor) override;
//     void communicate_down(const LatticeBoltzmann::DistributionFunction &,
//                           const int left_neighbor,
//                           const int right_neighbor) override;
//     void communicate_up(const LatticeBoltzmann::DistributionFunction &,
//                         const int left_neighbor,
//                         const int right_neighbor) override;
// };

// class MPIImplementationFactory {
//   public:
//     inline static std::unique_ptr<MPIImplementation>
//     get_implementation(const int lattice_width, const int lattice_height) {
// #ifdef MPIX_CUDA_AWARE_SUPPORT
//         if (MPIX_Query_cuda_support() == 1) {
//             std::cout << "CUDA-aware MPI is available. Using the GPU Aware "
//                          "implementation."
//                       << std::endl;
//             return std::unique_ptr<MPIImplementation>(
//                 new MPIGPUAware(lattice_width, lattice_height));
//         }
// #endif

//         std::cout << "CUDA-aware MPI is not available. Using the Non-GPU
//         Aware "
//                      "implementation."
//                   << std::endl;
//         return std::unique_ptr<MPIImplementation>(
//             new MPINotGPUAware(lattice_width, lattice_height));
//     }
// };

namespace MPIImplementationFactory {

inline std::unique_ptr<MPIImplementation>
get_implementation(const int total_lattice_width,
                   const int total_lattice_height,
                   std::shared_ptr<MPIData> mpi_data) {

    // #ifdef MPIX_CUDA_AWARE_SUPPORT
    //     if (MPIX_Query_cuda_support() == 1) {
    //         std::cout << "CUDA-aware MPI is available. Using the GPU Aware "
    //                      "implementation."
    //                   << std::endl;
    //         return std::unique_ptr<MPIImplementation>(
    //             new MPIGPUAware(total_lattice_width, total_lattice_height));
    //     }
    // #endif

    std::cout << "CUDA-aware MPI is not available. Using the Non - GPU Aware "
                 "implementation."
              << std::endl;
    return std::unique_ptr<MPIImplementation>(new MPINotGPUAware(
        total_lattice_width, total_lattice_height, mpi_data));
}

} // namespace MPIImplementationFactory