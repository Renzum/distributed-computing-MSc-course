add_library(lib STATIC
    lattice_boltzmann.cpp
    output_functions.cpp
    distribution_initializers.cpp
    lattice_boltzmann_types.cpp
)

target_include_directories(lib PUBLIC include)

target_link_libraries(lib PUBLIC MPI::MPI_CXX Eigen3::Eigen Kokkos::kokkos)
