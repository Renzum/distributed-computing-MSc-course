#pragma once

#include <mpi.h>

struct MPIData {
    int domain_size;

    int dims[2] = {0, 0};

    MPI_Comm cart{};

    int rank;

    int coords[2];

    int left_neighbor, right_neighbor, bottom_neighbor, top_neighbor;

    MPIData();
};