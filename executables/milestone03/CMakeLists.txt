add_executable(milestone03 main.cpp lattice-boltzman-class.cpp)
target_include_directories(milestone03 PRIVATE includes)

target_link_libraries(milestone03 PUBLIC MPI::MPI_CXX Eigen3::Eigen Kokkos::kokkos)