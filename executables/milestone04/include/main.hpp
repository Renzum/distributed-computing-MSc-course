#pragma once

#include <amplitude.hpp>
#include <lattice_boltzmann.hpp>
#include <lattice_boltzmann_types.hpp>

void ms4();

void init_starting_distribution(LatticeBoltzmann::DensityFunction &,
                                LatticeBoltzmann::VelocityProfile &, int, int);

void simulate_and_calculate(
    LatticeBoltzmann::DistributionFunction &distribution_function,
    LatticeBoltzmann::DistributionFunction &buffer_distribution_function,
    LatticeBoltzmann::DensityFunction &density_function,
    LatticeBoltzmann::VelocityProfile &velocity_profile, const double &omega,
    AmplitudeOutput &amp_output);