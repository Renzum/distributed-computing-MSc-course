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
        // y wraps around to prevent segfaul
        new_y = (old_y == 0) ? (grid_height - 1) : (old_y - 1);
        break;
    case Direction::Up:
    case Direction::UpRight:
    case Direction::UpLeft:
        // y wraps around to prevent segfaul
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
    DistributionFunction &distribution_function, const int ghost_layers) {
    const int grid_width = distribution_function.extent_int(0);
    const int grid_height = distribution_function.extent_int(1);

    // By default, ghost layers is 0, so the streaming step will perform a
    // periodic boundary operation and wrap the values across the lattice
    // If we provide a positive ghost_layers integer, it will loop through the
    // inner lattice skipping the ghost layers
    Kokkos::parallel_for(
        "Streaming Step",
        Kokkos::MDRangePolicy({ghost_layers, ghost_layers, 0},
                              {grid_width - ghost_layers,
                               grid_height - ghost_layers, TOTAL_DIRECTIONS}),
        KOKKOS_LAMBDA(const int &current_x, const int &current_y,
                      const int &dir) {
            // If ghost_layers > 0, new position will never wrap around since
            // any possible x will be greater than 0 and less than the lattice
            // width. Same for y.
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

void calculate_density(DensityFunction density_function,
                       DistributionFunction distribution_function,
                       const int ghost_layers) {
    const int grid_width = distribution_function.extent_int(0);
    const int grid_height = distribution_function.extent_int(1);

    Kokkos::parallel_for(
        "Density Calculation",
        Kokkos::MDRangePolicy(
            {ghost_layers, ghost_layers},
            {grid_width - ghost_layers, grid_height - ghost_layers}),
        KOKKOS_LAMBDA(const int &x, const int &y) {
            double local_density = 0;
            for (int dir = 0; dir < TOTAL_DIRECTIONS; dir++) {
                local_density += distribution_function(x, y, dir);
            }

            density_function(x, y) = local_density;
        });
}

void calculate_local_average_velocity(
    LocalAverageVelocity local_velocty_function,
    DistributionFunction distribution_function,
    DensityFunction density_function, const int ghost_layers) {
    const int grid_width = distribution_function.extent_int(0);
    const int grid_height = distribution_function.extent_int(1);

    Kokkos::parallel_for(
        "Average Local Velocity Calculation",
        Kokkos::MDRangePolicy(
            {ghost_layers, ghost_layers},
            {grid_width - ghost_layers, grid_height - ghost_layers}),
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
    DistributionFunction equilibrium_distribution,
    DensityFunction density_function,
    LocalAverageVelocity local_average_velocity_function,
    const int ghost_layers) {
    const int grid_width = equilibrium_distribution.extent_int(0);
    const int grid_height = equilibrium_distribution.extent_int(1);

    // Calculate constants at compile time
    constexpr double C1 = 3.0;
    constexpr double C2 = 1.0;
    constexpr double C3 = 9.0 / 2.0;
    constexpr double C4 = -3.0 / 2.0;

    Kokkos::parallel_for(
        "Equilibrium Distribution Calculation",
        Kokkos::MDRangePolicy({ghost_layers, ghost_layers, 0},
                              {grid_width - ghost_layers,
                               grid_height - ghost_layers, TOTAL_DIRECTIONS}),
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

void relax_distribution(DistributionFunction distribution_function,
                        DistributionFunction equilibrium_distribution_function,
                        const double omega, const int ghost_layers) {
    const int grid_width = distribution_function.extent_int(0);
    const int grid_height = distribution_function.extent_int(1);

    Kokkos::parallel_for(
        "Relaxation",
        Kokkos::MDRangePolicy({ghost_layers, ghost_layers, 0},
                              {grid_width - ghost_layers,
                               grid_height - ghost_layers, TOTAL_DIRECTIONS}),
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
    DistributionFunction buffer_distribution_view,
    DistributionFunction distribution_function,
    DensityFunction density_function, const double lid_vel_x,
    const double lid_vel_y, const int ghost_layers) {
    if (ghost_layers < 1) {
        std::cerr << "To perform a streaming with bounceback, there need to be "
                     "at least 1 ghost layer."
                  << std::endl;
        std::exit(1);
    }

    const int lattice_width = distribution_function.extent_int(0);
    const int lattice_height = distribution_function.extent_int(1);

    streaming_step_with_periodic_bounds(buffer_distribution_view,
                                        distribution_function, 1);

    // Since we moved all the values to the ghost nodes already, we make the
    // loop range only be the inner lattice points (i.e. excluding the outer
    // ghost layers)
    Kokkos::parallel_for(
        "Bounce Back Vertical", Kokkos::RangePolicy(1, lattice_height - 1),
        KOKKOS_LAMBDA(const int &y) {
            // Left Wall
            distribution_function(1, y, Direction::Right) =
                distribution_function(0, y, Direction::Left);
            // Diagonals
            distribution_function(1, y, Direction::UpRight) =
                distribution_function(0, y - 1, Direction::DownLeft);
            distribution_function(1, y, Direction::DownRight) =
                distribution_function(0, y + 1, Direction::UpLeft);

            // Right Wall
            distribution_function(lattice_width - 2, y, Direction::Left) =
                distribution_function(lattice_width - 1, y, Direction::Right);
            // Diagonals
            distribution_function(lattice_width - 2, y, Direction::UpLeft) =
                distribution_function(lattice_width - 1, y - 1,
                                      Direction::DownRight);
            distribution_function(lattice_width - 2, y, Direction::DownLeft) =
                distribution_function(lattice_width - 1, y + 1,
                                      Direction::UpRight);
        });

    Kokkos::parallel_for(
        "Bounce Back Horizontal", Kokkos::RangePolicy(1, lattice_width - 1),
        KOKKOS_LAMBDA(const int &x) {
            auto velocity_correction =
                [&distribution_function, &lid_vel_x,
                 &lid_vel_y](const double &local_density,
                             const Direction direction) -> double {
                if (lid_vel_x == 0 && lid_vel_y == 0) {
                    return 0;
                }

                double velocity_fraction;
                get_velocity_fraction(static_cast<Direction>(direction),
                                      velocity_fraction);

                int velocity_vector_x, velocity_vector_y;
                get_velocity_vector(static_cast<Direction>(direction),
                                    velocity_vector_x, velocity_vector_y);

                constexpr double c1 = 2.0 / (1.0 / 3.0);

                const double velocity_dot_product =
                    Kokkos::fma(velocity_vector_x, lid_vel_x,
                                velocity_vector_y * lid_vel_y);
                return c1 * velocity_fraction * local_density *
                       velocity_dot_product;
            };

            // Bottom Wall
            distribution_function(x, 1, Direction::Up) =
                distribution_function(x, 0, Direction::Down);
            // Diagonals
            distribution_function(x, 1, Direction::UpRight) =
                distribution_function(x - 1, 0, Direction::DownLeft);
            distribution_function(x, 1, Direction::UpLeft) =
                distribution_function(x + 1, 0, Direction::DownRight);

            const double local_density =
                density_function(x, lattice_height - 2);

            // Top Wall (Lid)
            distribution_function(x, lattice_height - 2, Direction::Down) =
                distribution_function(x, lattice_height - 1, Direction::Up) -
                velocity_correction(local_density, Direction::Up);
            // Diagonals
            distribution_function(x, lattice_height - 2, Direction::DownLeft) =
                distribution_function(x + 1, lattice_height - 1,
                                      Direction::UpRight) -
                velocity_correction(density_function(x, lattice_height - 2),
                                    Direction::UpRight);
            distribution_function(x, lattice_height - 2, Direction::DownRight) =
                distribution_function(x - 1, lattice_height - 1,
                                      Direction::UpLeft) -
                velocity_correction(local_density, Direction::UpLeft);
        });
}

} // namespace LatticeBoltzmann
