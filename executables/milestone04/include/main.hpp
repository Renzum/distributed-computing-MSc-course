#pragma once

#include <amplitude.hpp>
#include <lattice_boltzmann.hpp>

void ms4();

void init_starting_distribution(LatticeBoltzmann::Functions &);

LatticeBoltzmann::Functions
duplicate_functions(const LatticeBoltzmann::Functions &);

void simulate_and_calculate(LatticeBoltzmann::Functions &, const double &,
                            AmplitudeOutput &);