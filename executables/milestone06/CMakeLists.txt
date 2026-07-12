add_executable(milestone06 main.cpp)

target_include_directories(milestone06 PRIVATE include)

target_link_libraries(milestone06 PUBLIC lib MPI::MPI_CXX Eigen3::Eigen Kokkos::kokkos fmt::fmt)
