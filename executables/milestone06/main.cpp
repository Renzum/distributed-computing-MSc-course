#include <iostream>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>
#include <fmt/format.h>
#include <mpi.h>

#include <direction_definitions.hpp>
#include <distribution_initializers.hpp>
#include <lattice_boltzmann.hpp>
#include <output_functions.hpp>

#define GLOBAL_DOMAIN_SIZE 32000

class Node {
  private:
  public:
    struct GhostLayers {
        int left = 0, right = 0, down = 0, up = 0;
    };

    GhostLayers ghost_layers{};

    MPI_Comm cart{};

    int dims[2] = {0, 0};
    int periods[2] = {0, 0};

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

        std::cout << fmt::format("{:d}: Dimensions: [{:d}, {:d}]", rank,
                                 dims[0], dims[1])
                  << std::endl;
        std::cout << fmt::format("{:d}: Coordinates: [{:d}, {:d}]", rank,
                                 coords[0], coords[1])
                  << std::endl;

        // The four neighbors; with periodic boundaries every rank has all four
        MPI_Cart_shift(cart, /* dim */ 0, /* displacement */ 1, &up, &down);
        MPI_Cart_shift(cart, /* dim */ 1, /* displacement */ 1, &left, &right);

        // Check if the neighbors wrap around
        // If yes, set the neighbor rank to MPI_PROC_NULL
        // If no, remember that this node has a ghost layer on that side

        // MPI is row dominant, so the first dimension is the vertical position
        // and the second dimension is the horizontal position
        // Positive y is down, so we treat node at [0, 0] as the top left node

        std::cout << fmt::format("{:d}: My up is {:d}", rank, up) << std::endl;
        std::cout << fmt::format("{:d}: My down is {:d}", rank, down)
                  << std::endl;
        std::cout << fmt::format("{:d}: My left is {:d}", rank, left)
                  << std::endl;
        std::cout << fmt::format("{:d}: My right is {:d}", rank, right)
                  << std::endl;

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

        lattice_width = GLOBAL_DOMAIN_SIZE / dims[0] + ghost_layers.left +
                        ghost_layers.right;

        lattice_height =
            GLOBAL_DOMAIN_SIZE / dims[1] + ghost_layers.down + ghost_layers.up;

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
            auto down_subview = Kokkos::subview(distribution_function,
                                                Kokkos::ALL, 1, Kokkos::ALL);
            Kokkos::deep_copy(up_down_gpu_buffer, down_subview);
            Kokkos::deep_copy(up_down_send_buffer, up_down_gpu_buffer);
        }

        MPI_Sendrecv(up_down_send_buffer.data(), up_down_buffer_len, MPI_DOUBLE,
                     down, 0, up_down_recv_buffer.data(), up_down_buffer_len,
                     MPI_DOUBLE, up, 0, cart, &status);

        if (up != MPI_PROC_NULL) {
            auto up_subview =
                Kokkos::subview(distribution_function, Kokkos ::ALL,
                                lattice_height - 1, Kokkos::ALL);

            Kokkos::deep_copy(up_down_gpu_buffer, up_down_recv_buffer);
            Kokkos::deep_copy(up_subview, up_down_gpu_buffer);
        }
    }

    void communicate_up(
        LatticeBoltzmann::DistributionFunction &distribution_function) {
        MPI_Status status;

        if (up != MPI_PROC_NULL) {
            auto up_subview =
                Kokkos::subview(distribution_function, Kokkos::ALL,
                                lattice_height - 2, Kokkos::ALL);
            Kokkos::deep_copy(up_down_gpu_buffer, up_subview);
            Kokkos::deep_copy(up_down_send_buffer, up_down_gpu_buffer);
        }

        MPI_Sendrecv(up_down_send_buffer.data(), up_down_buffer_len, MPI_DOUBLE,
                     up, 0, up_down_recv_buffer.data(), up_down_buffer_len,
                     MPI_DOUBLE, down, 0, cart, &status);

        if (down != MPI_PROC_NULL) {
            auto down_subview = Kokkos::subview(distribution_function,
                                                Kokkos ::ALL, 0, Kokkos::ALL);

            Kokkos::deep_copy(up_down_gpu_buffer, up_down_recv_buffer);
            Kokkos::deep_copy(down_subview, up_down_gpu_buffer);
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

        std::cout << fmt::format(
                         "I'm node with rank {:d} and dimensions {:d}x{:d}",
                         node.rank, node.lattice_width, node.lattice_height)
                  << std::endl;

        const double omega = 1.2;
        const int iterations = 200;

        const double lid_vel_x = 0.1, lid_vel_y = 0.0;

        LatticeBoltzmann::Walls walls{
                // If there is no right neighbor, the right wall is a
                // bounce-back
                (node.right != MPI_PROC_NULL)
                    ? LatticeBoltzmann::Wall{ 
                          LatticeBoltzmann::BoundaryType::Periodic, }
                    : LatticeBoltzmann::Wall{ 
                          LatticeBoltzmann::BoundaryType::BounceBack, 0, 0, },
                // If there is no down neighbor, the bottom wall is a
                // bounce-back
                (node.down != MPI_PROC_NULL) ? LatticeBoltzmann::Wall{LatticeBoltzmann::BoundaryType::Periodic,}
                                             : LatticeBoltzmann::Wall{ LatticeBoltzmann::BoundaryType::BounceBack, 0, 0 ,},
                // If there is no left neighbor, the left wall is a
                // bounce-back (otherwise periodic)
                (node.left != MPI_PROC_NULL) ? LatticeBoltzmann::Wall{LatticeBoltzmann::BoundaryType::Periodic}
                                             : LatticeBoltzmann::Wall{LatticeBoltzmann::BoundaryType::BounceBack, 0, 0 ,},
                // If there is no up neighbor, the top wall is a
                // moving lid (otherwise periodic)
                (node.up != MPI_PROC_NULL)
                    ? LatticeBoltzmann::Wall{LatticeBoltzmann::BoundaryType::Periodic}
                    : LatticeBoltzmann::Wall{LatticeBoltzmann::BoundaryType::BounceBack, lid_vel_x, lid_vel_y ,},
            };

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

        std::cout << "starting iterations" << std::endl;
        auto timer = Kokkos::Timer();
        for (int i = 0; i < iterations; i++) {
            LatticeBoltzmann::calculate_density(density_function,
                                                distribution_function);
            LatticeBoltzmann::calculate_local_average_velocity(
                velocity_profile, distribution_function, density_function);
            LatticeBoltzmann::calculate_equilibrium_distribution(
                buffer_distribution, density_function, velocity_profile);
            LatticeBoltzmann::relax_distribution(distribution_function,
                                                 buffer_distribution, omega);

            Kokkos::fence();

            node.communicate_all(distribution_function);

            LatticeBoltzmann::streaming_step_with_bounce_back_and_lid(
                buffer_distribution, distribution_function, density_function,
                walls);
        }

        Kokkos::fence();

        auto elapsed = timer.seconds();
        auto actual_width = node.lattice_width - node.left - node.right;
        auto actual_height = node.lattice_height - node.down - node.up;
        auto mlups = (static_cast<double>(GLOBAL_DOMAIN_SIZE) *
                      GLOBAL_DOMAIN_SIZE * iterations) /
                     (elapsed * 1000000);

        std::cout << fmt::format("{:d}: {:d}x{:d} - {:d} iterations took "
                                 "{:f} seconds = {:f} MLUPS",
                                 node.rank, actual_width, actual_height,
                                 iterations, elapsed, mlups)
                  << std::endl;

        // // Make a view which excludes the ghost layers of the velocity
        // profile
        // // view
        // // TODO: Optimize. Only distribution function view needs ghost
        // layers. auto effective_velocity_profile = Kokkos::subview(
        //     velocity_profile,
        //     std::make_pair(node.left, node.lattice_width - node.right),
        //     std::make_pair(node.down, node.lattice_height - node.up),
        //     Kokkos::ALL);
        // auto velocity_profile_mirror =
        //     Kokkos::create_mirror_view(effective_velocity_profile);

        // Kokkos::View<double *[2], Kokkos::HostSpace> receive_buffer;

        // Kokkos::deep_copy(velocity_profile_mirror,
        // effective_velocity_profile);

        // if (node.rank == 0) {
        //     receive = LatticeBoltzmann::VelocityProfile("Receive Buffer")
        //         MPI_Gather(velocity_profile_mirror.data(), );
        // }
    }

    Kokkos::finalize();
    MPI_Finalize();
}
