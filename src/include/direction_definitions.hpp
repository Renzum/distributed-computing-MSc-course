#pragma once

#include <Kokkos_Core.hpp>
#include <array>

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

// #define VELOCITY_VECTORS \
//     {{0, 0}, {1, 0},  {0, 1},   {-1, 0}, {0, -1}, \
//      {1, 1}, {-1, 1}, {-1, -1}, {1, -1}};

#define VELOCITY_VECTORS_X {0, 1, 0, -1, 0, 1, -1, -1, 1}
#define VELOCITY_VECTORS_Y {0, 0, 1, 0, -1, 1, 1, -1, -1}

#define VELOCITY_FRACTIONS                                                     \
    {                                                                          \
        4.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,             \
        1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,                        \
    };
