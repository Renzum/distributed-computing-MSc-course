#include <mpi.h>

#include "mpi_layer.hpp"

MPILayer::MPILayer(int domain_width, int domain_height) {
    mpi_data = std::shared_ptr<MPIData>(new MPIData{});

    // We only want ghost layers on the sides that have a neighbor which the
    // node shalle communicate with
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
    lattice_width = domain_width / mpi_data->dims[1] + ghost_layers.left +
                    ghost_layers.right;

    lattice_height =
        domain_height / mpi_data->dims[0] + ghost_layers.down + ghost_layers.up;

    mpi_communication_layer = MPIImplementationFactory::get_implementation(
        lattice_width, lattice_height, mpi_data);
}
void MPILayer::communicate(
    const LatticeBoltzmann::DistributionFunction &distribution_function) {
    mpi_communication_layer->communicate_left(distribution_function);
    mpi_communication_layer->communicate_right(distribution_function);
    mpi_communication_layer->communicate_down(distribution_function);
    mpi_communication_layer->communicate_up(distribution_function);
}
