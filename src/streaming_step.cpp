#include <tuple>

#include <Kokkos_Core.hpp>

#include <direction_definitions.hpp>

#include "streaming_step.hpp"

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

void streaming_step(Kokkos::View<double ***> &previous_distribution,
                    Kokkos::View<double ***> &current_distribution,
                    int grid_width, int grid_height) {
    Kokkos::parallel_for(
        "Streaming Step",
        Kokkos::MDRangePolicy({0, 0, 0},
                              {grid_width, grid_height, TOTAL_DIRECTIONS}),
        KOKKOS_LAMBDA(const int &x_idx, const int &y_idx, const int &dir) {
            auto [x, y] = calculate_new_position(x_idx, y_idx, dir, grid_width,
                                                 grid_height);

            previous_distribution(x, y, dir) =
                current_distribution(x_idx, y_idx, dir);
        });

    Kokkos::kokkos_swap(current_distribution, previous_distribution);
}