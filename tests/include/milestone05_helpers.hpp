#pragma once

#include <lattice_boltzmann_types.hpp>

double correction_left(const int &x, const int &y, const double &density,
                       const double &wall_vel_x, const double &wall_vel_y);

void check_bounce_back_left(
    const LatticeBoltzmann::DistributionFunction::HostMirror
        &distribution_function,
    const LatticeBoltzmann::DistributionFunction::HostMirror
        &previous_distribution_function,
    const int &x, const int &y, const double &correction_left = 0);

double correction_up_left(const int &x, const int &y, const double &density,
                          const double &wall_vel_x, const double &wall_vel_y);

void check_bounce_back_up_left(
    const LatticeBoltzmann::DistributionFunction::HostMirror
        &distribution_function,
    const LatticeBoltzmann::DistributionFunction::HostMirror
        &previous_distribution_function,
    const int &x, const int &y, const double &correction_up_left = 0);

double correction_up(const int &x, const int &y, const double &density,
                     const double &wall_vel_x, const double &wall_vel_y);

void check_bounce_back_up(
    const LatticeBoltzmann::DistributionFunction::HostMirror
        &distribution_function,
    const LatticeBoltzmann::DistributionFunction::HostMirror
        &previous_distribution_function,
    const int &x, const int &y, const double &correction_up = 0);

double correction_up_right(const int &x, const int &y, const double &density,
                           const double &wall_vel_x, const double &wall_vel_y);

void check_bounce_back_up_right(
    const LatticeBoltzmann::DistributionFunction::HostMirror
        &distribution_function,
    const LatticeBoltzmann::DistributionFunction::HostMirror
        &previous_distribution_function,
    const int &x, const int &y, const double &correction_up_right = 0);

double correction_right(const int &x, const int &y, const double &density,
                        const double &wall_vel_x, const double &wall_vel_y);

void check_bounce_back_right(
    const LatticeBoltzmann::DistributionFunction::HostMirror
        &distribution_function,
    const LatticeBoltzmann::DistributionFunction::HostMirror
        &previous_distribution_function,
    const int &x, const int &y, const double &correction_right = 0);

double correction_down_right(const int &x, const int &y, const double &density,
                             const double &wall_vel_x,
                             const double &wall_vel_y);

void check_bounce_back_down_right(
    const LatticeBoltzmann::DistributionFunction::HostMirror
        &distribution_function,
    const LatticeBoltzmann::DistributionFunction::HostMirror
        &previous_distribution_function,
    const int &x, const int &y, const double &correction_down_right = 0);

double correction_down(const int &x, const int &y, const double &density,
                       const double &wall_vel_x, const double &wall_vel_y);

void check_bounce_back_down(
    const LatticeBoltzmann::DistributionFunction::HostMirror
        &distribution_function,
    const LatticeBoltzmann::DistributionFunction::HostMirror
        &previous_distribution_function,
    const int &x, const int &y, const double &correction_down = 0);

double correction_down_left(const int &x, const int &y, const double &density,
                            const double &wall_vel_x, const double &wall_vel_y);

void check_bounce_back_down_left(
    const LatticeBoltzmann::DistributionFunction::HostMirror
        &distribution_function,
    const LatticeBoltzmann::DistributionFunction::HostMirror
        &previous_distribution_function,
    const int &x, const int &y, const double &correction_down_left = 0);
