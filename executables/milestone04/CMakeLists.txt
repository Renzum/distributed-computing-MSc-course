add_executable(milestone04 main.cpp)

target_include_directories(milestone04 PRIVATE include)

target_link_libraries(milestone04 PUBLIC lib MPI::MPI_CXX Eigen3::Eigen Kokkos::kokkos fmt::fmt)
