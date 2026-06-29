add_executable(milestone02 main.cpp)
target_include_directories(milestone02 PRIVATE includes)

target_link_libraries(milestone02 PUBLIC lib MPI::MPI_CXX Eigen3::Eigen Kokkos::kokkos)