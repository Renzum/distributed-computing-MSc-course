#include <mpi_impl.hpp>

#include "mpi_impl_not_gpu_aware.hpp"

MPINotGPUAware::MPINotGPUAware(const int total_lattice_width,
                               const int total_lattice_height,
                               std::shared_ptr<MPIData> mpi_data) {
    lattice_width = total_lattice_width;
    lattice_height = total_lattice_height;

    left_right_gpu_buffer = GPUBuffer("Left Right GPU Buffer", lattice_height);
    left_right_buffer_len = lattice_height * TOTAL_DIRECTIONS;

    left_right_send_buffer = Kokkos::create_mirror_view(left_right_gpu_buffer);
    left_right_recv_buffer = Kokkos::create_mirror_view(left_right_gpu_buffer);

    down_up_gpu_buffer = GPUBuffer("Down Up GPU Buffer", lattice_width);
    down_up_buffer_len = lattice_width * TOTAL_DIRECTIONS;

    down_up_send_buffer = Kokkos::create_mirror_view(down_up_gpu_buffer);
    down_up_recv_buffer = Kokkos::create_mirror_view(down_up_gpu_buffer);

    this->mpi_data = mpi_data;
}

void MPINotGPUAware::communicate_left(
    const LatticeBoltzmann::DistributionFunction &distribution_function) {

    MPI_Status status;
    if (mpi_data->left_neighbor != MPI_PROC_NULL) {
        auto left_send_subview =
            Kokkos::subview(distribution_function, 1, Kokkos::ALL, Kokkos::ALL);
        Kokkos::deep_copy(left_right_gpu_buffer, left_send_subview);
        Kokkos::deep_copy(left_right_send_buffer, left_right_gpu_buffer);
    }

    MPI_Sendrecv(left_right_send_buffer.data(), left_right_buffer_len,
                 MPI_DOUBLE, mpi_data->left_neighbor, 0,
                 left_right_recv_buffer.data(), left_right_buffer_len,
                 MPI_DOUBLE, mpi_data->right_neighbor, 0, mpi_data->cart,
                 &status);

    if (mpi_data->right_neighbor != MPI_PROC_NULL) {

        auto right_recv_subview = Kokkos::subview(
            distribution_function, lattice_width - 1, Kokkos::ALL, Kokkos::ALL);

        Kokkos::deep_copy(left_right_gpu_buffer, left_right_recv_buffer);
        Kokkos::deep_copy(right_recv_subview, left_right_gpu_buffer);
    }
}

void MPINotGPUAware::communicate_right(
    const LatticeBoltzmann::DistributionFunction &distribution_function) {
    MPI_Status status;

    if (mpi_data->right_neighbor != MPI_PROC_NULL) {
        auto right_send_subview = Kokkos::subview(
            distribution_function, lattice_width - 2, Kokkos::ALL, Kokkos::ALL);
        Kokkos::deep_copy(left_right_gpu_buffer, right_send_subview);
        Kokkos::deep_copy(left_right_send_buffer, left_right_gpu_buffer);
    }

    MPI_Sendrecv(left_right_send_buffer.data(), left_right_buffer_len,
                 MPI_DOUBLE, mpi_data->right_neighbor, 0,
                 left_right_recv_buffer.data(), left_right_buffer_len,
                 MPI_DOUBLE, mpi_data->left_neighbor, 0, mpi_data->cart,
                 &status);

    if (mpi_data->left_neighbor != MPI_PROC_NULL) {

        auto left_recv_subview =
            Kokkos::subview(distribution_function, 0, Kokkos::ALL, Kokkos::ALL);

        Kokkos::deep_copy(left_right_gpu_buffer, left_right_recv_buffer);
        Kokkos::deep_copy(left_recv_subview, left_right_gpu_buffer);
    }
}
void MPINotGPUAware::communicate_down(
    const LatticeBoltzmann::DistributionFunction &distribution_function) {
    MPI_Status status;

    if (mpi_data->bottom_neighbor != MPI_PROC_NULL) {
        auto bottom_send_subview =
            Kokkos::subview(distribution_function, Kokkos::ALL, 1, Kokkos::ALL);
        Kokkos::deep_copy(down_up_gpu_buffer, bottom_send_subview);
        Kokkos::deep_copy(down_up_send_buffer, down_up_gpu_buffer);
    }

    MPI_Sendrecv(down_up_send_buffer.data(), down_up_buffer_len, MPI_DOUBLE,
                 mpi_data->bottom_neighbor, 0, down_up_recv_buffer.data(),
                 down_up_buffer_len, MPI_DOUBLE, mpi_data->top_neighbor, 0,
                 mpi_data->cart, &status);

    if (mpi_data->top_neighbor != MPI_PROC_NULL) {

        auto top_recv_subview =
            Kokkos::subview(distribution_function, Kokkos::ALL,
                            lattice_height - 1, Kokkos::ALL);

        Kokkos::deep_copy(down_up_gpu_buffer, down_up_recv_buffer);
        Kokkos::deep_copy(top_recv_subview, down_up_gpu_buffer);
    }
}

void MPINotGPUAware::communicate_up(
    const LatticeBoltzmann::DistributionFunction &distribution_function) {
    MPI_Status status;

    if (mpi_data->top_neighbor != MPI_PROC_NULL) {
        auto top_send_subview =
            Kokkos::subview(distribution_function, Kokkos::ALL,
                            lattice_height - 2, Kokkos::ALL);
        Kokkos::deep_copy(down_up_gpu_buffer, top_send_subview);
        Kokkos::deep_copy(down_up_send_buffer, down_up_gpu_buffer);
    }

    MPI_Sendrecv(down_up_send_buffer.data(), down_up_buffer_len, MPI_DOUBLE,
                 mpi_data->top_neighbor, 0, down_up_recv_buffer.data(),
                 down_up_buffer_len, MPI_DOUBLE, mpi_data->bottom_neighbor, 0,
                 mpi_data->cart, &status);

    if (mpi_data->bottom_neighbor != MPI_PROC_NULL) {

        auto bottom_recv_subview =
            Kokkos::subview(distribution_function, Kokkos::ALL, 1, Kokkos::ALL);

        Kokkos::deep_copy(down_up_gpu_buffer, down_up_recv_buffer);
        Kokkos::deep_copy(bottom_recv_subview, down_up_gpu_buffer);
    }
}