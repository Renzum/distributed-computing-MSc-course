#include <tuple>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <lattice-boltzman-class.hpp>

LatticeBoltzman::LatticeBoltzman(int grid_width, int grid_height) {
    this->grid_width = grid_width;
    this->grid_height = grid_height;

    buffer_distribution_view = Kokkos::View<double ***>(
        "Previous Density Function", grid_width, grid_height, TOTAL_DIRECTIONS);
    distribution_function = Kokkos::View<double ***>(
        "Current Density Function", grid_width, grid_height, TOTAL_DIRECTIONS);

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

    distribution_output_file = std::ofstream("test_output.txt", std::ios::out);

    current_distribution_to_file(); // Output initial distribution on creation
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

            buffer_distribution_view(x, y, dir) =
                distribution_function(x_idx, y_idx, dir);
        });

    Kokkos::kokkos_swap(distribution_function, buffer_distribution_view);
    iteration_count++;
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

LatticeBoltzman::~LatticeBoltzman() {
    distribution_output_file.close();
}
