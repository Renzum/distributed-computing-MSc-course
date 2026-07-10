#pragma once

#include <Kokkos_Macros.hpp>

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

KOKKOS_INLINE_FUNCTION
void get_velocity_vector(const Direction &dir, int &x, int &y) {
    switch (dir) {
    case Direction::Center:
        x = 0;
        y = 0;
        break;
    case Direction::Right:
        x = 1;
        y = 0;
        break;
    case Direction::Up:
        x = 0;
        y = 1;
        break;
    case Direction::Left:
        x = -1;
        y = 0;
        break;
    case Direction::Down:
        x = 0;
        y = -1;
        break;
    case Direction::UpRight:
        x = 1;
        y = 1;
        break;
    case Direction::UpLeft:
        x = -1;
        y = 1;
        break;
    case Direction::DownLeft:
        x = -1;
        y = -1;
        break;
    case Direction::DownRight:
        x = 1;
        y = -1;
        break;
    };
};

KOKKOS_INLINE_FUNCTION
void get_velocity_fraction(const Direction &dir, double &fraction) {
    switch (dir) {
    case Direction::Center:
        fraction = 4.0 / 9.0;
        break;
    case Direction::Right:
    case Direction::Up:
    case Direction::Left:
    case Direction::Down:
        fraction = 1.0 / 9.0;
        break;
    case Direction::UpRight:
    case Direction::UpLeft:
    case Direction::DownLeft:
    case Direction::DownRight:
        fraction = 1.0 / 36.0;
        break;
    };
}

KOKKOS_INLINE_FUNCTION Direction get_opposite_direction(const Direction &dir) {
    switch (dir) {
    case Direction::Center:
        return dir;
        break;
    case Direction::Right:
        return Direction::Left;
        break;
    case Direction::Up:
        return Direction::Down;
        break;
    case Direction::Left:
        return Direction::Right;
        break;
    case Direction::Down:
        return Direction::Up;
        break;
    case Direction::UpRight:
        return Direction::DownLeft;
        break;
    case Direction::UpLeft:
        return Direction::DownRight;
        break;
    case Direction::DownLeft:
        return Direction::UpRight;
        break;
    case Direction::DownRight:
        return Direction::UpLeft;
        break;
    default:
        return dir;
        break;
    }
};
