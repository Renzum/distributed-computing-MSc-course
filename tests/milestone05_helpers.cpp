#include <gtest/gtest.h>

#include <lattice_boltzmann_types.hpp>

#include "milestone05_helpers.hpp"

// Correction Functions calculate the expected correction term based on the wall
// velocity and the corresponding function's direction

// Check bounce back functions simply assert that the current value in the cell
// is the correct bounce back with the corresponding correction based on the
// moving wall velocity

double correction_left(const int &x, const int &y, const double &density,
                       const double &wall_vel_x, const double &wall_vel_y) {
    // Left
    constexpr int vec_x = -1, vec_y = 0;                             // Vector
    constexpr double weight = 1.0 / 9.0;                             // Weight
    const double dot_prod = wall_vel_x * vec_x + wall_vel_y * vec_y; // c * u

    return 6 * weight * density * dot_prod;
};

void check_bounce_back_left(
    const LatticeBoltzmann::DistributionFunction::HostMirror
        &distribution_function,
    const LatticeBoltzmann::DistributionFunction::HostMirror
        &previous_distribution_function,
    const int &x, const int &y, const double &correction_left) {

    const double current_value = distribution_function(x, y, Direction::Right);
    const double previous_value =
        previous_distribution_function(x, y, Direction::Left);

    ASSERT_DOUBLE_EQ(current_value, previous_value - correction_left);
}

double correction_up_left(const int &x, const int &y, const double &density,
                          const double &wall_vel_x, const double &wall_vel_y) {
    // Up Left
    constexpr int vec_x = -1, vec_y = 1;                             // Vector
    constexpr double weight = 1.0 / 36.0;                            // Weight
    const double dot_prod = wall_vel_x * vec_x + wall_vel_y * vec_y; // c * u

    return 6 * weight * density * dot_prod;
};

void check_bounce_back_up_left(
    const LatticeBoltzmann::DistributionFunction::HostMirror
        &distribution_function,
    const LatticeBoltzmann::DistributionFunction::HostMirror
        &previous_distribution_function,
    const int &x, const int &y, const double &correction_up_left) {

    const double current_value =
        distribution_function(x, y, Direction::DownRight);
    const double previous_value =
        previous_distribution_function(x, y, Direction::UpLeft);

    ASSERT_DOUBLE_EQ(current_value, previous_value - correction_up_left);
}

double correction_up(const int &x, const int &y, const double &density,
                     const double &wall_vel_x, const double &wall_vel_y) {
    // Up
    constexpr int vec_x = 0, vec_y = 1;                              // Vector
    constexpr double weight = 1.0 / 9.0;                             // Weight
    const double dot_prod = wall_vel_x * vec_x + wall_vel_y * vec_y; // c * u

    return 6 * weight * density * dot_prod;
};

void check_bounce_back_up(
    const LatticeBoltzmann::DistributionFunction::HostMirror
        &distribution_function,
    const LatticeBoltzmann::DistributionFunction::HostMirror
        &previous_distribution_function,
    const int &x, const int &y, const double &correction_up) {

    const double current_value = distribution_function(x, y, Direction::Down);
    const double previous_value =
        previous_distribution_function(x, y, Direction::Up);

    ASSERT_DOUBLE_EQ(current_value, previous_value - correction_up);
}

double correction_up_right(const int &x, const int &y, const double &density,
                           const double &wall_vel_x, const double &wall_vel_y) {
    // Up Right
    constexpr int vec_x = 1, vec_y = 1;                              // Vector
    constexpr double w5 = 1.0 / 36.0;                                // Weight
    const double dot_prod = wall_vel_x * vec_x + wall_vel_y * vec_y; // c * u

    return 6 * w5 * density * dot_prod;
};

void check_bounce_back_up_right(
    const LatticeBoltzmann::DistributionFunction::HostMirror
        &distribution_function,
    const LatticeBoltzmann::DistributionFunction::HostMirror
        &previous_distribution_function,
    const int &x, const int &y, const double &correction_up_right) {

    const double current_value =
        distribution_function(x, y, Direction::DownLeft);
    const double previous_value =
        previous_distribution_function(x, y, Direction::UpRight);

    ASSERT_DOUBLE_EQ(current_value, previous_value - correction_up_right);
}

double correction_right(const int &x, const int &y, const double &density,
                        const double &wall_vel_x, const double &wall_vel_y) {
    // Right
    constexpr int vec_x = 1, vec_y = 0;                              // Vector
    constexpr double weight = 1.0 / 9.0;                             // Weight
    const double dot_prod = wall_vel_x * vec_x + wall_vel_y * vec_y; // c * u

    return 6 * weight * density * dot_prod;
};

void check_bounce_back_right(
    const LatticeBoltzmann::DistributionFunction::HostMirror
        &distribution_function,
    const LatticeBoltzmann::DistributionFunction::HostMirror
        &previous_distribution_function,
    const int &x, const int &y, const double &correction_right) {

    const double current_value = distribution_function(x, y, Direction::Left);
    const double previous_value =
        previous_distribution_function(x, y, Direction::Right);

    ASSERT_DOUBLE_EQ(current_value, previous_value - correction_right);
}

double correction_down_right(const int &x, const int &y, const double &density,
                             const double &wall_vel_x,
                             const double &wall_vel_y) {
    // Down Right
    constexpr int vec_x = 1, vec_y = -1;                             // Vector
    constexpr double weight = 1.0 / 36.0;                            // Weight
    const double dot_prod = wall_vel_x * vec_x + wall_vel_y * vec_y; // c * u

    return 6 * weight * density * dot_prod;
};

void check_bounce_back_down_right(
    const LatticeBoltzmann::DistributionFunction::HostMirror
        &distribution_function,
    const LatticeBoltzmann::DistributionFunction::HostMirror
        &previous_distribution_function,
    const int &x, const int &y, const double &correction_down_right) {

    const double current_value = distribution_function(x, y, Direction::UpLeft);
    const double previous_value =
        previous_distribution_function(x, y, Direction::DownRight);

    ASSERT_DOUBLE_EQ(current_value, previous_value - correction_down_right);
}

double correction_down(const int &x, const int &y, const double &density,
                       const double &wall_vel_x, const double &wall_vel_y) {
    // Down
    constexpr int vec_x = 0, vec_y = -1;                             // Vector
    constexpr double weight = 1.0 / 9.0;                             // Weight
    const double dot_prod = wall_vel_x * vec_x + wall_vel_y * vec_y; // c * u

    return 6 * weight * density * dot_prod;
};

void check_bounce_back_down(
    const LatticeBoltzmann::DistributionFunction::HostMirror
        &distribution_function,
    const LatticeBoltzmann::DistributionFunction::HostMirror
        &previous_distribution_function,
    const int &x, const int &y, const double &correction_down) {

    const double current_value = distribution_function(x, y, Direction::Up);
    const double previous_value =
        previous_distribution_function(x, y, Direction::Down);

    ASSERT_DOUBLE_EQ(current_value, previous_value - correction_down);
}

double correction_down_left(const int &x, const int &y, const double &density,
                            const double &wall_vel_x,
                            const double &wall_vel_y) {
    // Down Left
    constexpr int vec_x = -1, vec_y = -1;                            // Vector
    constexpr double weight = 1.0 / 36.0;                            // Weight
    const double dot_prod = wall_vel_x * vec_x + wall_vel_y * vec_y; // c * u

    return 6 * weight * density * dot_prod;
};

void check_bounce_back_down_left(
    const LatticeBoltzmann::DistributionFunction::HostMirror
        &distribution_function,
    const LatticeBoltzmann::DistributionFunction::HostMirror
        &previous_distribution_function,
    const int &x, const int &y, const double &correction_down_left) {

    const double current_value =
        distribution_function(x, y, Direction::UpRight);
    const double previous_value =
        previous_distribution_function(x, y, Direction::DownLeft);

    ASSERT_DOUBLE_EQ(current_value, previous_value - correction_down_left);
}
