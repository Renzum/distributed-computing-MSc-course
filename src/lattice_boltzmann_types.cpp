#include <Kokkos_Core.hpp>

#include "lattice_boltzmann_types.hpp"

namespace LatticeBoltzmann {

Functions::Functions(const int grid_width, const int grid_height) {
    distribution_function =
        DistributionFunction("Distribution Function", grid_width, grid_height);

    host_distribution_function =
        Kokkos::create_mirror_view(distribution_function);

    buffer_distribution_function = DistributionFunction(
        "Buffer Distribution Function", grid_width, grid_height);

    density_function =
        DensityFunction("Density Function", grid_width, grid_height);
    host_density_function = Kokkos::create_mirror_view(density_function);

    local_average_velocity =
        LocalAverageVelocity("Local Average Velocity", grid_width, grid_height);
    host_local_average_velocity =
        Kokkos::create_mirror_view(local_average_velocity);
};

} // namespace LatticeBoltzmann