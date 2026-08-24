#include <mpi.h>

#include <fmt/format.h>

#include "mpi_layer.hpp"

MPILayer::MPILayer(int domain_width, int domain_height) {
    mpi_data = std::shared_ptr<MPIData>(new MPIData{});

    // We only want ghost layers on the sides that have a neighbor which the
    // node shall communicate with
    if (mpi_data->top_neighbor != MPI_PROC_NULL) {
        ghost_layers.up = 1;
    }

    if (mpi_data->bottom_neighbor != MPI_PROC_NULL) {
        ghost_layers.down = 1;
    }

    if (mpi_data->left_neighbor != MPI_PROC_NULL) {
        ghost_layers.left = 1;
    }

    if (mpi_data->right_neighbor != MPI_PROC_NULL) {
        ghost_layers.right = 1;
    }

    // MPI is row dominant, so the first dimension is the vertical position
    // and the second dimension is the horizontal position
    // Positive y is down, so we treat node at [0, 0] as the top left node
    local_lattice_width = static_cast<int>(domain_width / mpi_data->dims[1]) +
                          ghost_layers.left + ghost_layers.right;

    // We add the remainder of the domain width / MPI column dimension size to
    // the left most nodes on each row which allows us to use any domain size
    // even if it is not divisible by the width of the MPI dimensions
    if (mpi_data->coords[1] == 0) {
        local_lattice_width += domain_width % mpi_data->dims[1];
    }

    local_lattice_height = static_cast<int>(domain_height / mpi_data->dims[0]) +
                           ghost_layers.down + ghost_layers.up;

    // We add the remainder of the domain height / MPI row dimension size to
    // the top most nodes of each column which allows us to use any domain size
    // even if it is not divisible by the width of the MPI dimensions
    if (mpi_data->coords[0] == 0) {
        local_lattice_height += domain_height % mpi_data->dims[0];
    }

    if (mpi_data->dims[1] > 1) {
        global_lattice_offset_x =
            static_cast<int>(domain_width / mpi_data->dims[1]) *
            mpi_data->coords[1];
        // Since the leftmost nodes of each row in the MPI decomposition will
        // include the remainder of the domain_width / mpi_x_dimension, we need
        // to add it to the offset if the node isn't the leftmost of it's row
        if (mpi_data->coords[1] > 0) {
            global_lattice_offset_x += domain_width % mpi_data->dims[1];
        }
    } else {
        global_lattice_offset_x = 0;
    }

    if (mpi_data->dims[0] > 1) {
        global_lattice_offset_y =
            static_cast<int>(domain_height / mpi_data->dims[0]) *
            (mpi_data->dims[0] - mpi_data->coords[0] - 1);
        // We don't have a corner case for the remainder here since it is only
        // present in the top most node of each column, so it doesn't affect any
        // of the other nodes in the column
    } else {
        global_lattice_offset_y = 0;
    }

    std::cout << fmt::format("{:d}: Global offsets [y, x] are [{:d}, {:d}]",
                             mpi_data->rank, global_lattice_offset_y,
                             global_lattice_offset_x)
              << std::endl;

    mpi_communication_layer = MPIImplementationFactory::get_implementation(
        local_lattice_width, local_lattice_height, mpi_data);
}

void MPILayer::communicate(
    const LatticeBoltzmann::DistributionFunction &distribution_function) {
    mpi_communication_layer->communicate_left(distribution_function);
    mpi_communication_layer->communicate_right(distribution_function);
    mpi_communication_layer->communicate_down(distribution_function);
    mpi_communication_layer->communicate_up(distribution_function);
}
