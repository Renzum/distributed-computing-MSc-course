#pragma once

#include <Kokkos_Core.hpp>

#include <direction_definitions.hpp>

namespace LatticeBoltzmann {

using DistributionFunction = Kokkos::View<double **[TOTAL_DIRECTIONS]>;
using HostDistributionFunction =
    Kokkos::View<double **[TOTAL_DIRECTIONS], Kokkos::HostSpace>;

using DensityFunction = Kokkos::View<double **>;
using HostDensityFunction = Kokkos::View<double **, Kokkos::HostSpace>;

using VelocityProfile = Kokkos::View<double **[2]>;
using HostVelocityProfile = Kokkos::View<double **[2], Kokkos::HostSpace>;

enum BoundaryType {
    Periodic = 0,
    BounceBack = 1,
};

/**
 * The configuration of a wall.
 * Defines whether it is a periodic boundary or a bounce-back.
 * And the velocity of the wall if it is bounce-back.
 */
struct Wall {
    BoundaryType boundary_type = BoundaryType::Periodic;

    double vel_x = 0;
    double vel_y = 0;

    /**
     * Default constructor returns a periodic-boundary wall.
     */
    inline Wall() {};

    /**
     * Constructor which returns a bounce-back wall.
     * Set both parameters to 0 to get a fixed wall.
     * @param vel_x x velocity component of the moving wall
     * @param vel_y y velocity component of the moving wall
     */
    inline Wall(double vel_x, double vel_y)
        : boundary_type(BoundaryType::BounceBack),
          vel_x(vel_x),
          vel_y(vel_y) {};
};

struct Walls {
    Wall right, bottom, left, top;

    Walls() = delete; // Delete the default constructor

    /**
     * Explicit constructor for defining the wall configuration.
     */
    inline Walls(Wall right, Wall down, Wall left, Wall top)
        : right(right),
          bottom(down),
          left(left),
          top(top) {
    }
};

} // namespace LatticeBoltzmann
