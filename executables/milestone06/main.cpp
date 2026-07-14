#include <iostream>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>
#include <fmt/format.h>
#include <mpi.h>

#include <direction_definitions.hpp>
#include <distribution_initializers.hpp>
#include <lattice_boltzmann.hpp>

#define GLOBAL_DOMAIN_SIZE 300

class Node {
  private:
    struct GhostLayers {
        int left = 0, right = 0, down = 0, up = 0;
    };

    GhostLayers ghost_layers{};

    MPI_Comm cart{};

    int dims[2] = {0, 0};
    int periods[2] = {1, 1};

    int coords[2];

    int up, down, left, right;

    // Buffers for moving slices from GPU to CPU and vice versa
    using GPUBuffer = Kokkos::View<double *[TOTAL_DIRECTIONS]>;
    GPUBuffer left_right_gpu_buffer;
    GPUBuffer up_down_gpu_buffer;

    // CPU Space Buffers for contagious data sending via MPI
    using SendRecvBuffer = GPUBuffer::HostMirror;
    SendRecvBuffer left_right_send_buffer;
    SendRecvBuffer left_right_recv_buffer;
    int left_right_buffer_len;

    SendRecvBuffer up_down_send_buffer;
    SendRecvBuffer up_down_recv_buffer;
    int up_down_buffer_len;

  public:
    int lattice_width, lattice_height;

    int rank, mpi_domain_size;

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

        left_right_gpu_buffer =
            GPUBuffer("Horzintal GPU Buffer", lattice_height);

        left_right_send_buffer =
            Kokkos::create_mirror_view(left_right_gpu_buffer);
        left_right_recv_buffer = Kokkos::create_mirror(left_right_gpu_buffer);
        left_right_buffer_len = lattice_height * TOTAL_DIRECTIONS;

        up_down_gpu_buffer = GPUBuffer("Vertical GPU Buffer", lattice_width);

        up_down_send_buffer = Kokkos::create_mirror_view(up_down_gpu_buffer);
        up_down_recv_buffer = Kokkos::create_mirror(up_down_gpu_buffer);
        up_down_buffer_len = lattice_width * TOTAL_DIRECTIONS;
    }

    void communicate_left(
        LatticeBoltzmann::DistributionFunction &distribution_function) {
        MPI_Status status;
        if (left != MPI_PROC_NULL) {
            std::cout << fmt::format("{:d}: Sending left to {:d}", rank, left)
                      << std::endl;
            auto left_subview = Kokkos::subview(distribution_function, 1,
                                                Kokkos::ALL, Kokkos::ALL);
            Kokkos::deep_copy(left_right_gpu_buffer, left_subview);
            Kokkos::deep_copy(left_right_send_buffer, left_right_gpu_buffer);
        }

        MPI_Sendrecv(left_right_send_buffer.data(), left_right_buffer_len,
                     MPI_DOUBLE, left, 0, left_right_recv_buffer.data(),
                     left_right_buffer_len, MPI_DOUBLE, right, 0, cart,
                     &status);

        if (right != MPI_PROC_NULL) {
            std::cout << fmt::format("{:d}: Receiving right from {:d}", rank,
                                     right)
                      << std::endl;

            auto right_subview =
                Kokkos::subview(distribution_function, lattice_width - 1,
                                Kokkos::ALL, Kokkos::ALL);

            Kokkos::deep_copy(left_right_gpu_buffer, left_right_recv_buffer);
            Kokkos::deep_copy(right_subview, left_right_gpu_buffer);
        }
    }

    void communicate_right(
        LatticeBoltzmann::DistributionFunction &distribution_function) {
        MPI_Status status;
        if (right != MPI_PROC_NULL) {
            std::cout << fmt::format("{:d}: Sending right to {:d}", rank, right)
                      << std::endl;
            auto right_subview =
                Kokkos::subview(distribution_function, lattice_width - 2,
                                Kokkos::ALL, Kokkos::ALL);
            Kokkos::deep_copy(left_right_gpu_buffer, right_subview);
            Kokkos::deep_copy(left_right_send_buffer, left_right_gpu_buffer);
        }

        MPI_Sendrecv(left_right_send_buffer.data(), left_right_buffer_len,
                     MPI_DOUBLE, right, 0, left_right_recv_buffer.data(),
                     left_right_buffer_len, MPI_DOUBLE, left, 0, cart, &status);

        if (left != MPI_PROC_NULL) {
            std::cout << fmt::format("{:d}: Receiving left from {:d}", rank,
                                     left)
                      << std::endl;

            auto left_subview = Kokkos::subview(distribution_function, 0,
                                                Kokkos::ALL, Kokkos::ALL);

            Kokkos::deep_copy(left_right_gpu_buffer, left_right_recv_buffer);
            Kokkos::deep_copy(left_subview, left_right_gpu_buffer);
        }
    }

    void communicate_down(
        LatticeBoltzmann::DistributionFunction &distribution_function) {
        MPI_Status status;

        if (down != MPI_PROC_NULL) {
            std::cout << fmt::format("{:d}: Sending down to {:d}", rank, down)
                      << std::endl;
            auto down_subview = Kokkos::subview(distribution_function,
                                                Kokkos::ALL, 1, Kokkos::ALL);
            Kokkos::deep_copy(up_down_gpu_buffer, down_subview);
            Kokkos::deep_copy(up_down_send_buffer, left_right_gpu_buffer);
        }

        MPI_Sendrecv(up_down_send_buffer.data(), up_down_buffer_len, MPI_DOUBLE,
                     down, 0, up_down_recv_buffer.data(), up_down_buffer_len,
                     MPI_DOUBLE, up, 0, cart, &status);

        if (up != MPI_PROC_NULL) {
            std::cout << fmt::format("{:d}: Receiving up from {:d}", rank, up)
                      << std::endl;

            auto up_subview =
                Kokkos::subview(distribution_function, Kokkos ::ALL,
                                lattice_height - 1, Kokkos::ALL);

            Kokkos::deep_copy(up_down_gpu_buffer, up_down_recv_buffer);
            Kokkos::deep_copy(up_subview, left_right_gpu_buffer);
        }
    }

    void communicate_up(
        LatticeBoltzmann::DistributionFunction &distribution_function) {
        MPI_Status status;

        if (up != MPI_PROC_NULL) {
            std::cout << fmt::format("{:d}: Sending up to {:d}", rank, down)
                      << std::endl;
            auto up_subview =
                Kokkos::subview(distribution_function, Kokkos::ALL,
                                lattice_height - 2, Kokkos::ALL);
            Kokkos::deep_copy(up_down_gpu_buffer, up_subview);
            Kokkos::deep_copy(up_down_send_buffer, left_right_gpu_buffer);
        }

        MPI_Sendrecv(up_down_send_buffer.data(), up_down_buffer_len, MPI_DOUBLE,
                     up, 0, up_down_recv_buffer.data(), up_down_buffer_len,
                     MPI_DOUBLE, down, 0, cart, &status);

        if (down != MPI_PROC_NULL) {
            std::cout << fmt::format("{:d}: Receiving down from {:d}", rank,
                                     down)
                      << std::endl;

            auto down_subview = Kokkos::subview(distribution_function,
                                                Kokkos ::ALL, 0, Kokkos::ALL);

            Kokkos::deep_copy(up_down_gpu_buffer, up_down_recv_buffer);
            Kokkos::deep_copy(down_subview, left_right_gpu_buffer);
        }
    }

    void communicate_all(
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
};

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);

    {
        Node node{};

        std::cout << "I'm node with rank " << node.rank << std::endl;

        const double omega = 1.2;
        const int iterations = 1;

        {
            LatticeBoltzmann::DistributionFunction distribution_function(
                "Distribution Function", node.lattice_width,
                node.lattice_height);

            LatticeBoltzmann::DistributionInitializers::random_density(
                distribution_function);
            LatticeBoltzmann::DistributionFunction buffer_distribution(
                "Buffer Distribution Function", node.lattice_width,
                node.lattice_height);
            LatticeBoltzmann::DensityFunction density_function(
                "Distribution Function", node.lattice_width,
                node.lattice_height);
            LatticeBoltzmann::VelocityProfile velocity_profile(
                "Distribution Function", node.lattice_width,
                node.lattice_height);

            // Kokkos::View<double *[TOTAL_DIRECTIONS]> vertical_buffer(
            //     "Vertical GPU Buffer", node.lattice_width);

            // Kokkos::View<double *[TOTAL_DIRECTIONS]> horizontal_buffer(
            //     "Horizontal GPU Buffer", node.lattice_height);

            // MPI_Status status;

            std::cout << "starting iterations" << std::endl;
            for (int i = 0; i < iterations; i++) {
                Kokkos::fence();
                node.communicate_all(distribution_function);
            }
        }
    }

    Kokkos::finalize();
    MPI_Finalize();
}
