#include <Kokkos_Core.hpp>

#include "lattice_boltzmann_types.hpp"

namespace LatticeBoltzmann {

Functions::Functions(const int grid_width, const int grid_height,
                     const GhostLayers ghost_layers)
    : ghost_layers(ghost_layers) {
    const int full_width = grid_width + ghost_layers.left + ghost_layers.right;
    const int full_height =
        grid_height + ghost_layers.bottom + ghost_layers.top;
    distribution_function =
        DistributionFunction("Distribution Function", full_width, full_height);

    host_distribution_function =
        Kokkos::create_mirror_view(distribution_function);

    buffer_distribution_function = DistributionFunction(
        "Buffer Distribution Function", full_width, full_height);

    density_function =
        DensityFunction("Density Function", full_width, full_height);
    host_density_function = Kokkos::create_mirror_view(density_function);

    local_average_velocity =
        LocalAverageVelocity("Local Average Velocity", full_width, full_height);
    host_local_average_velocity =
        Kokkos::create_mirror_view(local_average_velocity);
};

} // namespace LatticeBoltzmann