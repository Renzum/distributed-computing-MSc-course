#pragma once

#include <Kokkos_Core.hpp>

#include <lattice_boltzmann.hpp>

namespace LatticeBoltzmann {

namespace DistributionInitializers {

void uniformDensity(const Kokkos::View<double ***> &, const double);
inline void uniformDensity(const Functions &lbm_functions,
                           double uniform_value = 1.0) {
    uniformDensity(lbm_functions.distribution_function, uniform_value);
}

void uniformDensityWithHigherCenter(const Kokkos::View<double ***> &,
                                    double uniform_value, double higher_value);
inline void
uniformDensityWithHigherCenter(const LatticeBoltzmann::Functions &lbm_functions,
                               double uniform_value = 1.0,
                               double higher_value = 1.1) {
    uniformDensityWithHigherCenter(lbm_functions.distribution_function,
                                   uniform_value, higher_value);
}

void randomDensity(const Kokkos::View<double ***> &);
inline void randomDensity(const Functions &lbm_functions) {
    randomDensity(lbm_functions.distribution_function);
}

} // namespace DistributionInitializers

} // namespace LatticeBoltzmann