#include <tuple>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

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

std::tuple<int, int> inline calculate_new_position(const int &x, const int &y,
                                                   const int &direction,
                                                   int grid_width,
                                                   int grid_height) {
    int new_x, new_y;

    switch (direction) {
    case Direction::Left:
    case Direction::UpLeft:
    case Direction::DownLeft:
        // x wraps around to prevent segfault
        new_x = (x == 0) ? (grid_width - 1) : (x - 1);
        break;
    case Direction::Right:
    case Direction::UpRight:
    case Direction::DownRight:
        // x wraps around to prevent segfault
        new_x = (x == grid_width - 1) ? 0 : (x + 1);
        break;
    default:
        new_x = x; // x doesn't change, no need to check
        break;
    }

    switch (direction) {
    case Direction::Down:
    case Direction::DownRight:
    case Direction::DownLeft:
        // y wraps around to prevent segfaul
        new_y = (y == 0) ? (grid_height - 1) : (y - 1);
        break;
    case Direction::Up:
    case Direction::UpRight:
    case Direction::UpLeft:
        // y wraps around to prevent segfaul
        new_y = (y == grid_height - 1) ? 0 : (y + 1);
        break;
    default:
        new_y = y; // y doesn't change, no need to check
        break;
    }

    return std::tuple<int, int>(new_x, new_y);
}

void LatticeBoltzman::calculate_next_step() {
    Kokkos::parallel_for(
        "Streaming Step",
        Kokkos::MDRangePolicy(
            {0, 0, 0}, {this->grid_width, this->grid_height, TOTAL_DIRECTIONS}),
        KOKKOS_LAMBDA(const int &x_idx, const int &y_idx, const int &dir) {
            auto [x, y] = calculate_new_position(x_idx, y_idx, dir, grid_width,
                                                 grid_height);

            previous_distribution(x, y, dir) =
                current_distribution(x_idx, y_idx, dir);
        });

    Kokkos::kokkos_swap(current_distribution, previous_distribution);
    iteration_count++;
}

void LatticeBoltzman::calculate_current_density() {
    Kokkos::parallel_for(
        "Density Calculation",
        Kokkos::MDRangePolicy({0, 0}, {grid_width, grid_height}),
        KOKKOS_LAMBDA(const int &x, const int &y) {
            double local_density = 0;
            for (int dir = 0; dir < TOTAL_DIRECTIONS; dir++) {
                local_density += current_distribution(x, y, dir);
            }

            current_density(x, y) = local_density;
        });
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
