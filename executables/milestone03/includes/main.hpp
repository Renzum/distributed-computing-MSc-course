#pragma once

#include <Kokkos_Core.hpp>
#include <tuple>

typedef Kokkos::View<int ***> Distribution;

std::tuple<int, int> calculate_new_position(const int &x, const int &y,
                                            const int &direction);
void streaming_step(Distribution &);
Distribution init_density_function();
void print_distribution(const Distribution &, const int &iteration);