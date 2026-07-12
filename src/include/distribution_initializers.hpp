#pragma once

#include <Kokkos_Core.hpp>

#include <lattice_boltzmann.hpp>

namespace LatticeBoltzmann {

namespace DistributionInitializers {

void uniform_density(DistributionFunction &, const double = 1.0);

void uniform_density_with_higher_center(DistributionFunction &,
                                        double uniform_value = 1.0,
                                        double higher_value = 1.1);

void uniform_at_rest(DensityFunction &, VelocityProfile &,
                     const double &uniform_value = 1.0);

void random_density(DistributionFunction &);

} // namespace DistributionInitializers

} // namespace LatticeBoltzmann