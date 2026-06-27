#include <tuple>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <density_calculation.hpp>
#include <direction_definitions.hpp>
#include <equilibrium_distribution_calculation.hpp>
#include <local_velocity_calculation.hpp>
#include <streaming_step.hpp>

#include "lattice-boltzman-class.hpp"

LatticeBoltzman::LatticeBoltzman(int grid_width, int grid_height,
                                 double viscocity) {
    this->grid_width = grid_width;
    this->grid_height = grid_height;

    this->viscocity = viscocity;

    distribution_output_file =
        std::ofstream("distribution_data.txt", std::ios::out);
    density_output_file = std::ofstream("density_data.txt", std::ios::out);
    local_average_velocity_output_file =
        std::ofstream("local_average_velocity_data.txt", std::ios::out);

    buffer_distribution_view =
        Kokkos::View<double ***>("Previous Distribution Function", grid_width,
                                 grid_height, TOTAL_DIRECTIONS);
    distribution_function =
        Kokkos::View<double ***>("Current Distribution Function", grid_width,
                                 grid_height, TOTAL_DIRECTIONS);

    density_function = Kokkos::View<double **>("Current Density Function",
                                               grid_width, grid_height);
    local_average_velocity = Kokkos::View<double ***>(
        "Current Local Average Velocity", grid_width, grid_height,
        2); // Each point is a two dimensional vector

    auto rand = Kokkos::Random_XorShift64_Pool<>(/* seed = */ 12345);

    Kokkos::parallel_for(
        "Initialize Density",
        Kokkos::MDRangePolicy({0, 0, 0},
                              {grid_width, grid_height, TOTAL_DIRECTIONS}),
        KOKKOS_LAMBDA(const int &x, const int &y, const int &dir) {
            auto gen = rand.get_state();

            distribution_function(x, y, dir) = x + y + dir + gen.rand(0, 100);

            rand.free_state(gen);
        });

    calculate_density_function();
    calculate_local_average_velocity_function();
    calculate_equilibrium_distribution_function();

    current_distribution_to_file(); // Output initial distribution on creation

    // Calculate starting density and output to file
    calculate_density_function();
    current_density_to_file();

    calculate_local_average_velocity_function();
}

void LatticeBoltzman::calculate_next_step() {
    iteration_count++;

    streaming_step(buffer_distribution_view, distribution_function, grid_width,
                   grid_height);
}

void LatticeBoltzman::calculate_density_function() {
    calculate_density(density_function, distribution_function, grid_width,
                      grid_height);
}

void LatticeBoltzman::calculate_local_average_velocity_function() {
    calculate_local_average_velocity(local_average_velocity,
                                     distribution_function, density_function,
                                     grid_width, grid_height);
}

void LatticeBoltzman::calculate_equilibrium_distribution_function() {
    calculate_equilibrium_distribution(buffer_distribution_view,
                                       density_function, local_average_velocity,
                                       grid_width, grid_height);

    Kokkos::parallel_for(
        "Relaxation",
        Kokkos::MDRangePolicy({0, 0, 0},
                              {grid_width, grid_height, TOTAL_DIRECTIONS}),
        KOKKOS_LAMBDA(const int &x, const int &y, const int &dir) {
            const double distribution_value = distribution_function(x, y, dir);
            const double eq_distribution_value =
                buffer_distribution_view(x, y, dir);

            buffer_distribution_view(x, y, dir) =
                distribution_value +
                viscocity * (eq_distribution_value - distribution_value);
        });

    Kokkos::kokkos_swap(buffer_distribution_view, distribution_function);
}

void LatticeBoltzman::current_distribution_to_file() {
    for (int x = 0; x < grid_width; x++) {
        for (int y = 0; y < grid_height; y++) {
            for (int dir = 0; dir < TOTAL_DIRECTIONS; dir++) {
                distribution_output_file << "{it: " << iteration_count << ", ";
                distribution_output_file << "x: " << x << ", ";
                distribution_output_file << "y: " << y << ", ";
                distribution_output_file << "dir: " << dir << ", ";
                distribution_output_file
                    << "val: " << distribution_function(x, y, dir) << "}"
                    << std::endl;
            }
        }
    }
}

void LatticeBoltzman::current_density_to_file() {
    for (int x = 0; x < grid_width; x++) {
        for (int y = 0; y < grid_height; y++) {
            density_output_file << "{it: " << iteration_count << ", ";
            density_output_file << "x: " << x << ", ";
            density_output_file << "y: " << y << ", ";
            density_output_file << "val: " << density_function(x, y) << "}"
                                << std::endl;
        }
    }
}

void LatticeBoltzman::current_density_to_file() {
    for (int x = 0; x < grid_width; x++) {
        for (int y = 0; y < grid_height; y++) {
            local_average_velocity_output_file << "{it: " << iteration_count
                                               << ", ";
            local_average_velocity_output_file << "x: " << x << ", ";
            local_average_velocity_output_file << "y: " << y << ", ";
            local_average_velocity_output_file
                << "vec_x: " << local_average_velocity(x, y, 0) << ", ";
            local_average_velocity_output_file
                << "vec_y: " << local_average_velocity(x, y, 1) << "}"
                << std::endl;
        }
    }
}

LatticeBoltzman::~LatticeBoltzman() {
    distribution_output_file.close();
    density_output_file.close();
    local_average_velocity_output_file.close();
}
