#pragma once

#include <iostream>
#include <mpi.h>

#include <mpi-ext.h>

class IMPIImplementation {
    // virtual void communicate_left();
    // virtual void communicate_right();
    // virtual void communicate_down();
    // virtual void communicate_up();
};

class MPINotGPUAware : public IMPIImplementation {};

class MPIGPUAware : public IMPIImplementation {};

class MPIImplementationFactory {
  public:
    inline static IMPIImplementation get_implementation() {
        // #ifdef MPIX_CUDA_AWARE_SUPPORT
        //         if (MPIX_Query_cuda_support() == 1) {
        //             std::cout << "CUDA-aware MPI is available. Using the GPU
        //             Aware "
        //                          "implementation."
        //                       << std::endl;
        //             return MPIGPUAware{};
        //         }
        // #endif

        std::cout << "CUDA-aware MPI is not available. Using the Non-GPU Aware "
                     "implementation."
                  << std::endl;
        return MPINotGPUAware{};
    }
};