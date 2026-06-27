add_library(lib STATIC hello.cpp lattice-boltzman-class.cpp streaming_step.cpp density_calculation.cpp)
target_include_directories(lib PUBLIC includes)
target_link_libraries(lib PUBLIC MPI::MPI_CXX Eigen3::Eigen Kokkos::kokkos)
