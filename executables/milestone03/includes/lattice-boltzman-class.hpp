#pragma once

#include <Kokkos_Core.hpp>
#include <fstream>

#define TOTAL_DIRECTIONS 9 // 8 + 1 for the looping
enum Direction {
    Center = 0,
    Right = 1,
    Up = 2,
    Left = 3,
    Down = 4,
    UpRight = 5,
    UpLeft = 6,
    DownLeft = 7,
    DownRight = 8,
};

class LatticeBoltzman {

  private:
    int grid_width, grid_height;
    int iteration_count = 0;

    std::ofstream distribution_output_file;
    std::ofstream density_output_file;

    Kokkos::View<double ***> current_distribution;
    Kokkos::View<double ***> previous_distribution;

    Kokkos::View<double **> current_density;

    void calculate_current_density();
    void current_density_to_file();

  public:
    LatticeBoltzman(int, int);

    ~LatticeBoltzman();

    void calculate_next_step();
    void current_distribution_to_file();
};