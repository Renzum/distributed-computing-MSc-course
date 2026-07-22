#include <Kokkos_Core.hpp>

#include <fmt/format.h>

#include <lattice_boltzmann.hpp>

#include "mpi_comm.hpp"

namespace GPUAware {

Node::Node(int global_domain_size) {
    // The following code snippets were taken from the Domain Decomposition
    // lecture notes
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_domain_size);

    // Balanced 2D process grid, e.g. 9 ranks -> dims = {3, 3}
    MPI_Dims_create(mpi_domain_size, 2, dims);

    // Periodic Cartesian communicator (periodic in both dimensions)
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, /* reorder */ 1, &cart);

    // Rank and coordinates within the new communicator
    MPI_Comm_rank(cart, &rank);
    MPI_Cart_coords(cart, rank, 2, coords);

    std::cout << fmt::format("{:d}: Dimensions: [{:d}, {:d}]", rank, dims[0],
                             dims[1])
              << std::endl;
    std::cout << fmt::format("{:d}: Coordinates: [{:d}, {:d}]", rank, coords[0],
                             coords[1])
              << std::endl;

    // The four neighbors; with non-periodict boundaries every rank has all four
    // Edge nodes set their non-existing neighbors to MPI_PROC_NULL
    MPI_Cart_shift(cart, /* dim */ 0, /* displacement */ 1, &up, &down);
    MPI_Cart_shift(cart, /* dim */ 1, /* displacement */ 1, &left, &right);

    // MPI is row dominant, so the first dimension is the vertical position
    // and the second dimension is the horizontal position
    // Positive y is down, so we treat node at [0, 0] as the top left node

    std::cout << fmt::format("{:d}: My up is {:d}", rank, up) << std::endl;
    std::cout << fmt::format("{:d}: My down is {:d}", rank, down) << std::endl;
    std::cout << fmt::format("{:d}: My left is {:d}", rank, left) << std::endl;
    std::cout << fmt::format("{:d}: My right is {:d}", rank, right)
              << std::endl;

    // We only want ghost layers on the sides that have a neighbor which the
    // node shalle communicate with
    if (up != MPI_PROC_NULL) {
        ghost_layers.up = 1;
    }

    if (down != MPI_PROC_NULL) {
        ghost_layers.down = 1;
    }

    if (left != MPI_PROC_NULL) {
        ghost_layers.left = 1;
    }

    if (right != MPI_PROC_NULL) {
        ghost_layers.right = 1;
    }

    lattice_width =
        global_domain_size / dims[0] + ghost_layers.left + ghost_layers.right;

    lattice_height =
        global_domain_size / dims[1] + ghost_layers.down + ghost_layers.up;

    left_right_send_gpu_buffer =
        GPUBuffer("Horzintal Send GPU Buffer", lattice_height);
    left_right_recv_gpu_buffer =
        GPUBuffer("Horzintal Receive GPU Buffer", lattice_height);

    left_right_buffer_len = lattice_height * TOTAL_DIRECTIONS;

    up_down_send_gpu_buffer =
        GPUBuffer("Vertical Send GPU Buffer", lattice_width);
    up_down_recv_gpu_buffer =
        GPUBuffer("Vertical Receive GPU Buffer", lattice_width);

    up_down_buffer_len = lattice_width * TOTAL_DIRECTIONS;
}

void Node::communicate_left(
    LatticeBoltzmann::DistributionFunction &distribution_function) {
    MPI_Status status;
    if (left != MPI_PROC_NULL) {
        auto left_subview =
            Kokkos::subview(distribution_function, 1, Kokkos::ALL, Kokkos::ALL);
        Kokkos::deep_copy(left_right_send_gpu_buffer, left_subview);
    }

    MPI_Sendrecv(left_right_send_gpu_buffer.data(), left_right_buffer_len,
                 MPI_DOUBLE, left, 0, left_right_recv_gpu_buffer.data(),
                 left_right_buffer_len, MPI_DOUBLE, right, 0, cart, &status);

    if (right != MPI_PROC_NULL) {

        auto right_subview = Kokkos::subview(
            distribution_function, lattice_width - 1, Kokkos::ALL, Kokkos::ALL);

        Kokkos::deep_copy(right_subview, left_right_recv_gpu_buffer);
    }
}

void Node::communicate_right(
    LatticeBoltzmann::DistributionFunction &distribution_function) {
    MPI_Status status;
    if (right != MPI_PROC_NULL) {
        auto right_subview = Kokkos::subview(
            distribution_function, lattice_width - 2, Kokkos::ALL, Kokkos::ALL);
        Kokkos::deep_copy(left_right_send_gpu_buffer, right_subview);
    }

    MPI_Sendrecv(left_right_send_gpu_buffer.data(), left_right_buffer_len,
                 MPI_DOUBLE, right, 0, left_right_recv_gpu_buffer.data(),
                 left_right_buffer_len, MPI_DOUBLE, left, 0, cart, &status);

    if (left != MPI_PROC_NULL) {
        auto left_subview =
            Kokkos::subview(distribution_function, 0, Kokkos::ALL, Kokkos::ALL);

        Kokkos::deep_copy(left_subview, left_right_recv_gpu_buffer);
    }
}

void Node::communicate_down(
    LatticeBoltzmann::DistributionFunction &distribution_function) {
    MPI_Status status;

    if (down != MPI_PROC_NULL) {
        auto down_subview =
            Kokkos::subview(distribution_function, Kokkos::ALL, 1, Kokkos::ALL);
        Kokkos::deep_copy(up_down_send_gpu_buffer, down_subview);
    }

    MPI_Sendrecv(up_down_send_gpu_buffer.data(), up_down_buffer_len, MPI_DOUBLE,
                 down, 0, up_down_recv_gpu_buffer.data(), up_down_buffer_len,
                 MPI_DOUBLE, up, 0, cart, &status);

    if (up != MPI_PROC_NULL) {
        auto up_subview = Kokkos::subview(distribution_function, Kokkos ::ALL,
                                          lattice_height - 1, Kokkos::ALL);

        Kokkos::deep_copy(up_subview, up_down_recv_gpu_buffer);
    }
}
void Node::communicate_up(
    LatticeBoltzmann::DistributionFunction &distribution_function) {
    MPI_Status status;

    if (up != MPI_PROC_NULL) {
        auto up_subview = Kokkos::subview(distribution_function, Kokkos::ALL,
                                          lattice_height - 2, Kokkos::ALL);
        Kokkos::deep_copy(up_down_send_gpu_buffer, up_subview);
    }

    MPI_Sendrecv(up_down_send_gpu_buffer.data(), up_down_buffer_len, MPI_DOUBLE,
                 up, 0, up_down_recv_gpu_buffer.data(), up_down_buffer_len,
                 MPI_DOUBLE, down, 0, cart, &status);

    if (down != MPI_PROC_NULL) {
        auto down_subview = Kokkos::subview(distribution_function, Kokkos ::ALL,
                                            0, Kokkos::ALL);

        Kokkos::deep_copy(down_subview, up_down_recv_gpu_buffer);
    }
}
void Node::communicate_all(
    LatticeBoltzmann::DistributionFunction &distribution_function) {
    // Skip horizontal communication if no left or right neighbors exist
    if (left != MPI_PROC_NULL || right != MPI_PROC_NULL) {
        communicate_left(distribution_function);
        communicate_right(distribution_function);
    }

    // Skip vertical communication if no up or down neighbors exist
    if (down != MPI_PROC_NULL || up != MPI_PROC_NULL) {
        communicate_down(distribution_function);
        communicate_up(distribution_function);
    }
};

} // namespace GPUAware