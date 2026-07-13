#include <utility>

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <fmt/format.h>

#include <lattice_boltzmann.hpp>

#define GLOBAL_DOMAIN_SIZE 300

struct GhostLayers {
    int left = 0, right = 0, down = 0, up = 0;
};

struct Node {
    GhostLayers ghost_layers{};

    MPI_Comm cart{};

    int mpi_domain_size;
    int dims[2] = {0, 0};
    int periods[2] = {1, 1};

    int rank, coords[2];

    int up, down, left, right;

    int lattice_width, lattice_height;

    Node() {
        // The following code snippets were taken from the Domain Decomposition
        // lecture notes
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_domain_size);

        // Balanced 2D process grid, e.g. 9 ranks -> dims = {3, 3}
        MPI_Dims_create(mpi_domain_size, 2, dims);

        // Periodic Cartesian communicator (periodic in both dimensions)
        MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, /* reorder */ 1,
                        &cart);

        // Rank and coordinates within the new communicator
        MPI_Comm_rank(cart, &rank);
        MPI_Cart_coords(cart, rank, 2, coords);

        // The four neighbors; with periodic boundaries every rank has all four
        MPI_Cart_shift(cart, /* dim */ 0, /* displacement */ 1, &up, &down);
        MPI_Cart_shift(cart, /* dim */ 1, /* displacement */ 1, &left, &right);

        // Check if the neighbors wrap around
        // If yes, set the neighbor rank to MPI_PROC_NULL
        // If no, remember that this node has a ghost layer on that side

        if (left < rank)
            ghost_layers.left = 1;
        else
            left = MPI_PROC_NULL;

        if (right > rank)
            ghost_layers.right = 1;
        else
            right = MPI_PROC_NULL;

        if (down < rank)
            ghost_layers.down = 1;
        else
            down = MPI_PROC_NULL;

        if (up > rank)
            ghost_layers.up = 1;
        else
            up = MPI_PROC_NULL;

        lattice_width = GLOBAL_DOMAIN_SIZE / mpi_domain_size +
                        ghost_layers.left + ghost_layers.right;
        lattice_height = GLOBAL_DOMAIN_SIZE / mpi_domain_size +
                         ghost_layers.down + ghost_layers.up;
    }
};

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);

    Node node{};

    std::cout << "I'm node with rank " << node.rank << std::endl;

    {
        LatticeBoltzmann::DistributionFunction distribution_function(
            "Distribution Function", node.lattice_width, node.lattice_height);

        LatticeBoltzmann::DensityFunction distribution_function(
            "Distribution Function", node.lattice_width, node.lattice_height);
        // LatticeBoltzmann::DistributionFunction test("Test Function", 100,
        // 200); LatticeBoltzmann::HostDistributionMirror mirror =
        //     Kokkos::create_mirror_view(test);

        // auto subview = Kokkos::subview(mirror, 0, Kokkos::ALL, Kokkos::ALL);

        // Kokkos::View<double *[9], Kokkos::HostSpace> packed("buffer",
        //                                                     subview.extent(0));
        // std::cout << packed.span_is_contiguous() << std::endl;

        // MPI_Sendrecv(packed.data(), packed.extent(0) * 9)
    }

    Kokkos::finalize();
    MPI_Finalize();
}