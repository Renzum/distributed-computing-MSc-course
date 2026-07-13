#include <iostream>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <fmt/format.h>

#include <direction_definitions.hpp>
#include <distribution_initializers.hpp>
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

    const double omega = 1.2;
    const int iterations = 10;

    {
        LatticeBoltzmann::DistributionFunction distribution_function(
            "Distribution Function", node.lattice_width, node.lattice_height);
        LatticeBoltzmann::DistributionInitializers::random_density(
            distribution_function);
        LatticeBoltzmann::DistributionFunction buffer_distribution(
            "Buffer Distribution Function", node.lattice_width,
            node.lattice_height);
        LatticeBoltzmann::DensityFunction density_function(
            "Distribution Function", node.lattice_width, node.lattice_height);
        LatticeBoltzmann::VelocityProfile velocity_profile(
            "Distribution Function", node.lattice_width, node.lattice_height);

        Kokkos::View<double *[TOTAL_DIRECTIONS]> vertical_buffer(
            "Vertical GPU Buffer", node.lattice_width);

        auto v_send_buffer = Kokkos::create_mirror_view(vertical_buffer);
        auto v_recv_buffer = Kokkos::create_mirror(vertical_buffer);

        Kokkos::View<double *[TOTAL_DIRECTIONS]> horizontal_buffer(
            "Horizontal GPU Buffer", node.lattice_height);

        auto h_send_buffer = Kokkos::create_mirror_view(horizontal_buffer);
        auto h_recv_buffer = Kokkos::create_mirror(horizontal_buffer);

        MPI_Status status;

        std::cout << "starting iterations" << std::endl;
        for (int i = 0; i < iterations; i++) {
            LatticeBoltzmann::calculate_density(density_function,
                                                distribution_function);
            LatticeBoltzmann::calculate_local_average_velocity(
                velocity_profile, distribution_function, density_function);

            LatticeBoltzmann::calculate_equilibrium_distribution(
                buffer_distribution, density_function, velocity_profile);

            LatticeBoltzmann::relax_distribution(distribution_function,
                                                 buffer_distribution, omega);

            std::cout << "waiting for GPU" << std::endl;
            Kokkos::fence();

            std::cout << "Beginning communication phase" << std::endl;

            int buffer_len;
            buffer_len = node.lattice_height * TOTAL_DIRECTIONS;

            if (node.left != MPI_PROC_NULL && node.right != MPI_PROC_NULL) {

                auto left_subview = Kokkos::subview(distribution_function, 1,
                                                    Kokkos::ALL, Kokkos::ALL);

                Kokkos::deep_copy(h_send_buffer, left_subview);

                std::cout << "Sending to " << node.left
                          << " and receiving from " << node.right << std::endl;
                MPI_Sendrecv(h_send_buffer.data(), buffer_len, MPI_DOUBLE,
                             node.left, 0, h_recv_buffer.data(), buffer_len,
                             MPI_DOUBLE, node.right, 0, MPI_COMM_WORLD,
                             &status);
                std::cout << "Received from " << node.right << std::endl;

                auto right_subview = Kokkos::subview(distribution_function,
                                                     node.lattice_width - 2,
                                                     Kokkos::ALL, Kokkos::ALL);

                Kokkos::deep_copy(h_send_buffer, right_subview);

                std::cout << "Sending to " << node.right
                          << " and receiving from " << node.left << std::endl;

                MPI_Sendrecv(h_send_buffer.data(), buffer_len, MPI_DOUBLE,
                             node.right, 0, h_recv_buffer.data(), buffer_len,
                             MPI_DOUBLE, node.left, 0, MPI_COMM_WORLD, &status);

                std::cout << "Received from " << node.left << std::endl;
            } else if (node.left != MPI_PROC_NULL) {
                auto left_subview = Kokkos::subview(distribution_function, 1,
                                                    Kokkos::ALL, Kokkos::ALL);

                Kokkos::deep_copy(h_send_buffer, left_subview);

                std::cout << "Sending to " << node.left;
                MPI_Send(h_send_buffer.data(), buffer_len, MPI_DOUBLE,
                         node.left, 0, MPI_COMM_WORLD);

                MPI_Recv(h_recv_buffer.data(), buffer_len, MPI_DOUBLE,
                         node.left, 0, MPI_COMM_WORLD, &status);
                std::cout << "Received from " << node.left << std::endl;
            } else if (node.right != MPI_PROC_NULL) {
                auto right_subview = Kokkos::subview(distribution_function,
                                                     node.lattice_width - 2,
                                                     Kokkos::ALL, Kokkos::ALL);

                Kokkos::deep_copy(h_send_buffer, right_subview);

                MPI_Recv(h_recv_buffer.data(), buffer_len, MPI_DOUBLE,
                         node.right, 0, MPI_COMM_WORLD, &status);
                std::cout << "Received from " << node.right << std::endl;

                std::cout << "Sending to " << node.right;
                MPI_Send(h_send_buffer.data(), buffer_len, MPI_DOUBLE,
                         node.right, 0, MPI_COMM_WORLD);

            } else {
                std::cout << "There are no neighbors to send." << std::endl;
            }
        }
    }

    Kokkos::finalize();
    MPI_Finalize();
}