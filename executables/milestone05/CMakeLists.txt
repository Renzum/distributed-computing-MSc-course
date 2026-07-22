add_executable(milestone05 main.cpp)

target_include_directories(milestone05 PRIVATE include)

target_link_libraries(milestone05 PUBLIC lib MPI::MPI_CXX Kokkos::kokkos fmt::fmt)
