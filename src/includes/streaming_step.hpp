#pragma once

void streaming_step(Kokkos::View<double ***> &buffer_distribution_view,
                    Kokkos::View<double ***> &distribution_function,
                    int grid_width, int grid_height);