#include <Kokkos_Core.hpp>

#include "density_calculation.hpp"

void calculate_density(const Kokkos::View<double **> &density_function,
                       const Kokkos::View<double ***> distribution_function,
                       int grid_width, int grid_height) {
    Kokkos::parallel_for(
        "Density Calculation",
        Kokkos::MDRangePolicy({0, 0}, {grid_width, grid_height}),
        KOKKOS_LAMBDA(const int &x, const int &y) {
            double local_density = 0;
            for (int dir = 0; dir < TOTAL_DIRECTIONS; dir++) {
                local_density += distribution_function(x, y, dir);
            }

            density_function(x, y) = local_density;
        });
}