add_executable(milestone02 main.cpp)

target_link_libraries(milestone02 PUBLIC lib MPI::MPI_CXX Eigen3::Eigen Kokkos::kokkos)