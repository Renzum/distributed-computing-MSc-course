#pragma once

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

constexpr std::tuple<int, int> velocity_vector[] = {{0, 0},  {1, 0},   {0, 1},
                                                    {-1, 0}, {0, -1},  {1, 1},
                                                    {-1, 1}, {-1, -1}, {1, -1}};
