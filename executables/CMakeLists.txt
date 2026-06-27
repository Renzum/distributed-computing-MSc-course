add_executable(monte_carlo monte_carlo_pi.cpp)
target_link_libraries(monte_carlo PUBLIC MPI::MPI_CXX Eigen3::Eigen Kokkos::kokkos)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE lib)

add_subdirectory(milestone01)
add_subdirectory(milestone02)
add_subdirectory(milestone03)