#include <Kokkos_Core.hpp>

#include <fmt/printf.h>

#include "main.hpp"

int main(int argc, char *argv[]) {
    Kokkos::initialize(argc, argv);

    fmt::println("Hello, World! From Milestone {:d}.", 5);

    Kokkos::finalize();
}
