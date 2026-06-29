add_library(lib STATIC lattice_boltzmann_class.cpp lattice_boltzmann_impl.cpp)

target_include_directories(lib PUBLIC includes/public)
target_include_directories(lib PRIVATE includes/private)

target_link_libraries(lib PUBLIC MPI::MPI_CXX Eigen3::Eigen Kokkos::kokkos)
