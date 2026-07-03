add_executable(milestone04 main.cpp)

target_link_libraries(milestone04 PUBLIC lib MPI::MPI_CXX Eigen3::Eigen Kokkos::kokkos)