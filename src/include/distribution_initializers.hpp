#pragma once

#include <Kokkos_Core.hpp>

#include <lattice_boltzmann.hpp>

namespace LatticeBoltzmann {

namespace DistributionInitializers {

void uniform_density(DistributionFunction, const double);
inline void uniform_density(const Functions &lbm_functions,
                            double uniform_value = 1.0) {
    uniform_density(lbm_functions.distribution_function, uniform_value);
}

void uniform_density_with_higher_center(DistributionFunction,
                                        double uniform_value,
                                        double higher_value);
inline void uniform_density_with_higher_center(
    const LatticeBoltzmann::Functions &lbm_functions,
    double uniform_value = 1.0, double higher_value = 1.1) {
    uniform_density_with_higher_center(lbm_functions.distribution_function,
                                       uniform_value, higher_value);
}

void random_density(DistributionFunction);
inline void random_density(const Functions &lbm_functions) {
    random_density(lbm_functions.distribution_function);
}

} // namespace DistributionInitializers

} // namespace LatticeBoltzmann