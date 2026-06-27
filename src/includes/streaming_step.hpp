#pragma once

void streaming_step(Kokkos::View<double ***> &previous_distribution,
                    Kokkos::View<double ***> &current_distribution,
                    int grid_width, int grid_height);