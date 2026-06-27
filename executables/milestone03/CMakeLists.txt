add_executable(milestone03 main.cpp)

target_link_libraries(milestone03 PUBLIC lib MPI::MPI_CXX Eigen3::Eigen Kokkos::kokkos)