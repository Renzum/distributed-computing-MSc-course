#include <mpi.h>

#include "mpi_data.hpp"

MPIData::MPIData() {
    // The following code snippets were taken from the Domain Decomposition
    // lecture notes
    MPI_Comm_size(MPI_COMM_WORLD, &domain_size);

    // Balanced 2D process grid, e.g. 9 ranks -> mpi_dims = {3, 3}
    MPI_Dims_create(domain_size, 2, dims);

    // Non-periodic Cartesian communicator
    int periods[2] = {0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, /* reorder */ 1, &cart);

    // Rank and coordinates within the new communicator
    MPI_Comm_rank(cart, &rank);
    MPI_Cart_coords(cart, rank, 2, coords);

    // The four neighbors; with non-periodic boundaries every rank has all four
    // Edge nodes set their non-existing neighbors to MPI_PROC_NULL
    MPI_Cart_shift(cart, /* dim */ 0, /* displacement */ 1, &top_neighbor,
                   &bottom_neighbor);
    MPI_Cart_shift(cart, /* dim */ 1, /* displacement */ 1, &left_neighbor,
                   &right_neighbor);
}