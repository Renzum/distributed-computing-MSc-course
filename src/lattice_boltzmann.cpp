#include <iostream>

#include <Kokkos_Core.hpp>

#include <direction_definitions.hpp>

#include "lattice_boltzmann.hpp"

namespace LatticeBoltzmann {

namespace {

KOKKOS_INLINE_FUNCTION
void calculate_new_position(const int &old_x, const int &old_y,
                            const int &direction, const int grid_width,
                            const int grid_height, int &new_x, int &new_y) {
    switch (direction) {
    case Direction::Left:
    case Direction::UpLeft:
    case Direction::DownLeft:
        // x wraps around to prevent segfault
        new_x = (old_x == 0) ? (grid_width - 1) : (old_x - 1);
        break;
    case Direction::Right:
    case Direction::UpRight:
    case Direction::DownRight:
        // x wraps around to prevent segfault
        new_x = (old_x == grid_width - 1) ? 0 : (old_x + 1);
        break;
    default:
        new_x = old_x; // x doesn't change, no need to check
        break;
    }

    switch (direction) {
    case Direction::Down:
    case Direction::DownRight:
    case Direction::DownLeft:
        // y wraps around to prevent segfault
        new_y = (old_y == 0) ? (grid_height - 1) : (old_y - 1);
        break;
    case Direction::Up:
    case Direction::UpRight:
    case Direction::UpLeft:
        // y wraps around to prevent segfault
        new_y = (old_y == grid_height - 1) ? 0 : (old_y + 1);
        break;
    default:
        new_y = old_y; // y doesn't change, no need to check
        break;
    }
}

} // namespace

Functions::Functions(const int grid_width, const int grid_height,
                     const int ghost_layers)
    : ghost_layers(ghost_layers) {
    distribution_function =
        DistributionFunction("Distribution Function", grid_width + ghost_layers,
                             grid_height + ghost_layers);
    host_distribution_function =
        Kokkos::create_mirror_view(distribution_function);

    buffer_distribution_function = DistributionFunction(
        "Buffer Distribution Function", grid_width + ghost_layers,
        grid_height + ghost_layers);

    density_function =
        DensityFunction("Density Function", grid_width + ghost_layers,
                        grid_height + ghost_layers);
    host_density_function = Kokkos::create_mirror_view(density_function);

    local_average_velocity = LocalAverageVelocity("Local Average Velocity",
                                                  grid_width + ghost_layers,
                                                  grid_height + ghost_layers);
    host_local_average_velocity =
        Kokkos::create_mirror_view(local_average_velocity);
};

void streaming_step_with_periodic_bounds(
    DistributionFunction &buffer_distribution_view,
    DistributionFunction &distribution_function,
    const GhostLayers &ghost_layers) {
    const int grid_width = distribution_function.extent_int(0);
    const int grid_height = distribution_function.extent_int(1);

    // By default, ghost layers is 0, so the streaming step will perform a
    // periodic boundary operation and wrap the values across the lattice
    // If we provide positive ghost_layers integers, it will loop through the
    // inner lattice skipping the ghost layers
    Kokkos::parallel_for(
        "Streaming Step",
        Kokkos::MDRangePolicy({ghost_layers.left, ghost_layers.bottom, 0},
                              {grid_width - ghost_layers.right,
                               grid_height - ghost_layers.top,
                               TOTAL_DIRECTIONS}),
        KOKKOS_LAMBDA(const int &current_x, const int &current_y,
                      const int &dir) {
            int new_x, new_y;
            calculate_new_position(current_x, current_y, dir, grid_width,
                                   grid_height, new_x, new_y);

            // We move the values to the buffer according to their velocity
            // And then simply swap the buffer with the distribution function
            // view
            buffer_distribution_view(new_x, new_y, dir) =
                distribution_function(current_x, current_y, dir);
        });

    Kokkos::kokkos_swap(distribution_function, buffer_distribution_view);
}

void calculate_density(DensityFunction &density_function,
                       const DistributionFunction &distribution_function,
                       const GhostLayers &ghost_layers) {
    const int grid_width = distribution_function.extent_int(0);
    const int grid_height = distribution_function.extent_int(1);

    Kokkos::parallel_for(
        "Density Calculation",
        Kokkos::MDRangePolicy(
            {ghost_layers.left, ghost_layers.bottom},
            {grid_width - ghost_layers.right, grid_height - ghost_layers.top}),
        KOKKOS_LAMBDA(const int &x, const int &y) {
            double local_density = 0;
            for (int dir = 0; dir < TOTAL_DIRECTIONS; dir++) {
                local_density += distribution_function(x, y, dir);
            }

            density_function(x, y) = local_density;
        });
}

void calculate_local_average_velocity(
    LocalAverageVelocity &local_velocty_function,
    const DistributionFunction &distribution_function,
    const DensityFunction &density_function, const GhostLayers &ghost_layers) {
    const int grid_width = distribution_function.extent_int(0);
    const int grid_height = distribution_function.extent_int(1);

    Kokkos::parallel_for(
        "Average Local Velocity Calculation",
        Kokkos::MDRangePolicy(
            {ghost_layers.left, ghost_layers.bottom},
            {grid_width - ghost_layers.right, grid_height - ghost_layers.top}),
        KOKKOS_LAMBDA(const int &x, const int &y) {
            int velocity_vector_x, velocity_vector_y;

            const double inverse_density = 1.0 / density_function(x, y);

            double vec_x = 0;
            double vec_y = 0;

            for (int dir = 0; dir < TOTAL_DIRECTIONS; dir++) {

                get_velocity_vector(static_cast<Direction>(dir),
                                    velocity_vector_x, velocity_vector_y);

                vec_x += velocity_vector_x * distribution_function(x, y, dir);
                vec_y += velocity_vector_y * distribution_function(x, y, dir);
            }

            vec_x *= inverse_density;
            vec_y *= inverse_density;

            local_velocty_function(x, y, 0) = vec_x;
            local_velocty_function(x, y, 1) = vec_y;
        });
}

void calculate_equilibrium_distribution(
    DistributionFunction &equilibrium_distribution,
    const DensityFunction &density_function,
    const LocalAverageVelocity &local_average_velocity_function,
    const GhostLayers &ghost_layers) {
    const int grid_width = equilibrium_distribution.extent_int(0);
    const int grid_height = equilibrium_distribution.extent_int(1);

    // Calculate constants at compile time
    constexpr double C1 = 3.0;
    constexpr double C2 = 1.0;
    constexpr double C3 = 9.0 / 2.0;
    constexpr double C4 = -3.0 / 2.0;

    Kokkos::parallel_for(
        "Equilibrium Distribution Calculation",
        Kokkos::MDRangePolicy({ghost_layers.left, ghost_layers.bottom, 0},
                              {grid_width - ghost_layers.right,
                               grid_height - ghost_layers.top,
                               TOTAL_DIRECTIONS}),
        KOKKOS_LAMBDA(const int &x, const int &y, const int &dir) {
            // This is where the fun begins

            double velocity_fraction;
            get_velocity_fraction(static_cast<Direction>(dir),
                                  velocity_fraction);

            // Precompute the w_i * rho coefficient
            const double coefficient =
                velocity_fraction * density_function(x, y);

            int velocity_vec_x, velocity_vec_y;
            get_velocity_vector(static_cast<Direction>(dir), velocity_vec_x,
                                velocity_vec_y);

            const double avg_velocity_x =
                local_average_velocity_function(x, y, 0);
            const double avg_velocity_y =
                local_average_velocity_function(x, y, 1);

            // We use FMA to reduce floating point rounding errors as much
            // as possible

            // c_ix * ux + c_iy * uy
            const double dot_product =
                Kokkos::fma(velocity_vec_x, avg_velocity_x,
                            velocity_vec_y * avg_velocity_y);

            // ux * ux + uy * uy
            const double avg_velocity_vec_len_sqr =
                Kokkos::fma(avg_velocity_x, avg_velocity_x,
                            avg_velocity_y * avg_velocity_y);

            // A1 = 3.0 * (c_i * u)  + 1.0
            const double A1 = Kokkos::fma(C1, dot_product, C2);

            // A2 = (9.0 / 2.0) * (c_i * u)(c_i * u) + A1
            const double A2 = Kokkos::fma(C3, dot_product * dot_product, A1);

            // A3 = (-3.0 / 2.0) * (|u| * |u|) + A2
            const double A3 = Kokkos::fma(C4, avg_velocity_vec_len_sqr, A2);

            // Result = w_i * rho * A3
            equilibrium_distribution(x, y, dir) = coefficient * A3;
        });
}

void relax_distribution(
    DistributionFunction &distribution_function,
    const DistributionFunction &equilibrium_distribution_function,
    const double omega, const GhostLayers &ghost_layers) {
    const int grid_width = distribution_function.extent_int(0);
    const int grid_height = distribution_function.extent_int(1);

    Kokkos::parallel_for(
        "Relaxation",
        Kokkos::MDRangePolicy({ghost_layers.left, ghost_layers.bottom, 0},
                              {grid_width - ghost_layers.right,
                               grid_height - ghost_layers.top,
                               TOTAL_DIRECTIONS}),
        KOKKOS_LAMBDA(const int &x, const int &y, const int &dir) {
            const double distribution_value = distribution_function(x, y, dir);
            const double eq_distribution_value =
                equilibrium_distribution_function(x, y, dir);

            distribution_function(x, y, dir) =
                Kokkos::fma(omega, eq_distribution_value - distribution_value,
                            distribution_value);
        });
}

void streaming_step_with_bounce_back_and_lid(
    DistributionFunction &buffer_distribution_view,
    DistributionFunction &distribution_function,
    const DensityFunction &density_function, const double lid_vel_x,
    const double lid_vel_y, const GhostLayers &ghost_layers) {

    streaming_step_with_periodic_bounds(buffer_distribution_view,
                                        distribution_function, ghost_layers);

    const int lattice_width = distribution_function.extent_int(0);
    const int lattice_height = distribution_function.extent_int(1);

    Kokkos::parallel_for(
        "Bounce Back Vertical Walls",
        Kokkos::RangePolicy(ghost_layers.bottom,
                            lattice_height - ghost_layers.top),
        KOKKOS_LAMBDA(const int &y) {
            // Left Wall
            distribution_function(ghost_layers.left, y, Direction::UpRight) =
                buffer_distribution_view(ghost_layers.left, y,
                                         Direction::DownLeft);

            distribution_function(ghost_layers.left, y, Direction::Right) =
                buffer_distribution_view(ghost_layers.left, y, Direction::Left);

            distribution_function(ghost_layers.left, y, Direction::DownRight) =
                buffer_distribution_view(ghost_layers.left, y,
                                         Direction::UpLeft);

            // Right Wall
            distribution_function(lattice_width - ghost_layers.right - 1, y,
                                  Direction::UpLeft) =
                buffer_distribution_view(lattice_width - ghost_layers.right - 1,
                                         y, Direction::DownRight);

            distribution_function(lattice_width - ghost_layers.right - 1, y,
                                  Direction::Left) =
                buffer_distribution_view(lattice_width - ghost_layers.right - 1,
                                         y, Direction::Right);

            distribution_function(lattice_width - ghost_layers.right - 1, y,
                                  Direction::DownLeft) =
                buffer_distribution_view(lattice_width - ghost_layers.right - 1,
                                         y, Direction::UpRight);
        });

    Kokkos::parallel_for(
        "Bounce Back Horizontal Walls",
        Kokkos::RangePolicy(ghost_layers.left,
                            lattice_width - ghost_layers.right),
        KOKKOS_LAMBDA(const int &x) {
            // Bottom Wall
            distribution_function(x, ghost_layers.bottom, Direction::UpRight) =
                buffer_distribution_view(x, ghost_layers.bottom,
                                         Direction::DownLeft);

            distribution_function(x, ghost_layers.bottom, Direction::Up) =
                buffer_distribution_view(x, ghost_layers.bottom,
                                         Direction::Down);

            distribution_function(x, ghost_layers.bottom, Direction::UpLeft) =
                buffer_distribution_view(x, ghost_layers.bottom,
                                         Direction::DownRight);

            // Top Wall
            distribution_function(x, lattice_height - ghost_layers.top - 1,
                                  Direction::DownRight) =
                buffer_distribution_view(x,
                                         lattice_height - ghost_layers.top - 1,
                                         Direction::UpLeft);

            distribution_function(x, lattice_height - ghost_layers.top - 1,
                                  Direction::Down) =
                buffer_distribution_view(
                    x, lattice_height - ghost_layers.top - 1, Direction::Up);

            distribution_function(x, lattice_height - ghost_layers.top - 1,
                                  Direction::DownLeft) =
                buffer_distribution_view(x,
                                         lattice_height - ghost_layers.top - 1,
                                         Direction::UpRight);
        });
}

} // namespace LatticeBoltzmann
