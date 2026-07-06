#pragma once

#include <Kokkos_Core.hpp>

#include <direction_definitions.hpp>

namespace LatticeBoltzmann {

using DistributionFunction = Kokkos::View<double **[TOTAL_DIRECTIONS]>;
using HostDistributionMirror = DistributionFunction::HostMirror;

using DensityFunction = Kokkos::View<double **>;
using HostDensityMirror = DensityFunction::HostMirror;

using LocalAverageVelocity = Kokkos::View<double **[2]>;
using HostLocalAverageVelocityMirror = LocalAverageVelocity::HostMirror;

enum WallType {
    Streaming = 0,
    BounceBack = 1,
};

struct Wall {
    WallType wall_type = WallType::Streaming;

    double vel_x = 0;
    double vel_y = 0;

    int ghost_layers = 0;

    inline Wall() {};
    inline Wall(int ghost_layers) : ghost_layers(ghost_layers) {};
    inline Wall(WallType wall_type, int ghost_layers = 0, double vel_x = 0,
                double vel_y = 0)
        : wall_type(wall_type),
          vel_x(vel_x),
          vel_y(vel_y),
          ghost_layers(ghost_layers) {};
};

struct Walls {
    Wall right, bottom, left, top;

    Walls() = delete; // Delete the default constructor

    inline Walls(Wall right, Wall down, Wall left, Wall top)
        : right(right),
          bottom(down),
          left(left),
          top(top) {
    }
};

// struct GhostLayers {
//     int right, bottom, left, top;

//     // Default constructor sets them all to 0
//     inline GhostLayers() {
//         right = bottom = left = top = 0;
//     };

//     // 4 Int Constructor
//     inline GhostLayers(int right, int bottom, int left, int top)
//         : right(right),
//           bottom(bottom),
//           left(left),
//           top(top) {};
// };

/**
 * Struct containing all the necessariy Kokkos::Views to represent the
 * different functions of Lattice-Boltzmann
 */
struct Functions {
    /**
     * 3D Lattice of x and y size where each cell is an array of 9 elements
     * representing all the directions for particles
     */
    DistributionFunction distribution_function;
    HostDistributionMirror host_distribution_function;

    /**
     * 3D Lattice of x and y size where each cell is an array of 9 elements
     * representing all the directions for particles.
     *
     * Used as a buffer for storing temporary results.
     */
    DistributionFunction buffer_distribution_function;

    /**
     * 2D Lattice of x and y size where each cell represents the density of
     * the distribution function
     */
    DensityFunction density_function;
    HostDensityMirror host_density_function;

    /**
     * 3D Lattice of x and y size where each cell holds an array of size 2
     * that represents a 2D velocity vector
     */
    LocalAverageVelocity local_average_velocity;
    HostLocalAverageVelocityMirror host_local_average_velocity;

    /**
     * Allocates the necessary views with the provided lattice width and
     * height and appends any required ghost layers to the internal width and
     * height of the lattice
     */
    Functions(const int grid_width, const int grid_height);
};

} // namespace LatticeBoltzmann
