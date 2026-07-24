#include <memory>

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <direction_definitions.hpp>

#include "mpi_impl_gpu_aware.hpp"

MPIGPUAware::MPIGPUAware(const int lattice_width, const int lattice_height,
                         std::shared_ptr<MPIData> mpi_data) {
    this->lattice_width = lattice_width;
    this->lattice_height = lattice_height;

    this->mpi_data = mpi_data;

    left_right_gpu_send_buffer =
        GPUBuffer("Left Right GPU Send Buffer", lattice_height);
    left_right_gpu_recv_buffer =
        GPUBuffer("Left Right GPU Recv Buffer", lattice_height);
    left_right_buffer_len = lattice_height * TOTAL_DIRECTIONS;

    down_up_gpu_send_buffer =
        GPUBuffer("Down Up GPU Send Buffer", lattice_width);
    down_up_gpu_recv_buffer =
        GPUBuffer("Down Up GPU Recv Buffer", lattice_width);
    down_up_buffer_len = lattice_width * TOTAL_DIRECTIONS;
}

void MPIGPUAware::communicate_left(
    const LatticeBoltzmann::DistributionFunction &distribution_function) {

    MPI_Status status;
    if (mpi_data->left_neighbor != MPI_PROC_NULL) {
        auto left_send_subview =
            Kokkos::subview(distribution_function, 1, Kokkos::ALL, Kokkos::ALL);
        Kokkos::deep_copy(left_right_gpu_send_buffer, left_send_subview);
    }

    MPI_Sendrecv(left_right_gpu_send_buffer.data(), left_right_buffer_len,
                 MPI_DOUBLE, mpi_data->left_neighbor, 0,
                 left_right_gpu_recv_buffer.data(), left_right_buffer_len,
                 MPI_DOUBLE, mpi_data->right_neighbor, 0, mpi_data->cart,
                 &status);

    if (mpi_data->right_neighbor != MPI_PROC_NULL) {

        auto right_recv_subview = Kokkos::subview(
            distribution_function, lattice_width - 1, Kokkos::ALL, Kokkos::ALL);

        Kokkos::deep_copy(right_recv_subview, left_right_gpu_recv_buffer);
    }
}

void MPIGPUAware::communicate_right(
    const LatticeBoltzmann::DistributionFunction &distribution_function) {

    MPI_Status status;
    if (mpi_data->right_neighbor != MPI_PROC_NULL) {
        auto right_send_subview = Kokkos::subview(
            distribution_function, lattice_width - 2, Kokkos::ALL, Kokkos::ALL);
        Kokkos::deep_copy(left_right_gpu_send_buffer, right_send_subview);
    }

    MPI_Sendrecv(left_right_gpu_send_buffer.data(), left_right_buffer_len,
                 MPI_DOUBLE, mpi_data->right_neighbor, 0,
                 left_right_gpu_recv_buffer.data(), left_right_buffer_len,
                 MPI_DOUBLE, mpi_data->left_neighbor, 0, mpi_data->cart,
                 &status);

    if (mpi_data->left_neighbor != MPI_PROC_NULL) {

        auto left_recv_subview =
            Kokkos::subview(distribution_function, 0, Kokkos::ALL, Kokkos::ALL);

        Kokkos::deep_copy(left_recv_subview, left_right_gpu_recv_buffer);
    }
}

void MPIGPUAware::communicate_down(
    const LatticeBoltzmann::DistributionFunction &distribution_function) {
    MPI_Status status;
    if (mpi_data->bottom_neighbor != MPI_PROC_NULL) {
        auto bottom_send_subview =
            Kokkos::subview(distribution_function, Kokkos::ALL, 1, Kokkos::ALL);
        Kokkos::deep_copy(down_up_gpu_send_buffer, bottom_send_subview);
    }

    MPI_Sendrecv(down_up_gpu_send_buffer.data(), down_up_buffer_len, MPI_DOUBLE,
                 mpi_data->bottom_neighbor, 0, down_up_gpu_recv_buffer.data(),
                 down_up_buffer_len, MPI_DOUBLE, mpi_data->top_neighbor, 0,
                 mpi_data->cart, &status);

    if (mpi_data->top_neighbor != MPI_PROC_NULL) {

        auto top_recv_subview =
            Kokkos::subview(distribution_function, Kokkos::ALL,
                            lattice_height - 1, Kokkos::ALL);

        Kokkos::deep_copy(top_recv_subview, down_up_gpu_recv_buffer);
    }
}

void MPIGPUAware::communicate_up(
    const LatticeBoltzmann::DistributionFunction &distribution_function) {
    MPI_Status status;
    if (mpi_data->top_neighbor != MPI_PROC_NULL) {
        auto top_send_subview =
            Kokkos::subview(distribution_function, Kokkos::ALL,
                            lattice_height - 2, Kokkos::ALL);
        Kokkos::deep_copy(down_up_gpu_send_buffer, top_send_subview);
    }

    MPI_Sendrecv(down_up_gpu_send_buffer.data(), down_up_buffer_len, MPI_DOUBLE,
                 mpi_data->top_neighbor, 0, down_up_gpu_recv_buffer.data(),
                 down_up_buffer_len, MPI_DOUBLE, mpi_data->bottom_neighbor, 0,
                 mpi_data->cart, &status);

    if (mpi_data->bottom_neighbor != MPI_PROC_NULL) {

        auto bottom_recv_subview =
            Kokkos::subview(distribution_function, Kokkos::ALL, 0, Kokkos::ALL);

        Kokkos::deep_copy(bottom_recv_subview, down_up_gpu_recv_buffer);
    }
}