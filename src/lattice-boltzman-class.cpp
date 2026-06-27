#include <tuple>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <density_calculation.hpp>
#include <direction_definitions.hpp>
#include <streaming_step.hpp>

#include "lattice-boltzman-class.hpp"

LatticeBoltzman::LatticeBoltzman(int grid_width, int grid_height) {
    this->grid_width = grid_width;
    this->grid_height = grid_height;

    previous_distribution =
        Kokkos::View<double ***>("Previous Distribution Function", grid_width,
                                 grid_height, TOTAL_DIRECTIONS);
    current_distribution =
        Kokkos::View<double ***>("Current Distribution Function", grid_width,
                                 grid_height, TOTAL_DIRECTIONS);

    current_density = Kokkos::View<double **>("Current Density Function",
                                              grid_width, grid_height);

    auto rand = Kokkos::Random_XorShift64_Pool<>(/* seed = */ 12345);

    Kokkos::parallel_for(
        "Initialize Density",
        Kokkos::MDRangePolicy({0, 0, 0},
                              {grid_width, grid_height, TOTAL_DIRECTIONS}),
        KOKKOS_LAMBDA(const int &x, const int &y, const int &dir) {
            auto gen = rand.get_state();

            current_distribution(x, y, dir) = x + y + dir + gen.rand(0, 100);

            rand.free_state(gen);
        });

    distribution_output_file = std::ofstream("test_output.txt", std::ios::out);
    density_output_file = std::ofstream("test_output.txt", std::ios::out);

    current_distribution_to_file(); // Output initial distribution on creation

    // Calculate starting density and output to file
    calculate_current_density();
    current_density_to_file();
}

void LatticeBoltzman::calculate_next_step() {
    iteration_count++;

    streaming_step(previous_distribution, current_distribution, grid_width,
                   grid_height);
}

void LatticeBoltzman::calculate_current_density() {
    calculate_density(current_density, current_distribution, grid_width,
                      grid_height);
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
                    << "val: " << current_distribution(x, y, dir) << "}"
                    << std::endl;
            }
        }
    }
}

void LatticeBoltzman::current_density_to_file() {
    for (int x = 0; x < grid_width; x++) {
        for (int y = 0; y < grid_height; y++) {
            for (int dir = 0; dir < TOTAL_DIRECTIONS; dir++) {
                distribution_output_file << "{it: " << iteration_count << ", ";
                distribution_output_file << "x: " << x << ", ";
                distribution_output_file << "y: " << y << ", ";
                distribution_output_file << "dir: " << dir << ", ";
                distribution_output_file
                    << "val: " << current_distribution(x, y, dir) << "}"
                    << std::endl;
            }
        }
    }
}

LatticeBoltzman::~LatticeBoltzman() {
    distribution_output_file.close();
}
