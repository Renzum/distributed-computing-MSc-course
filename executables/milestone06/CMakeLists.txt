add_executable(milestone06 main.cpp mpi_node.cpp mpi_comm.cpp mpi_impl.cpp)

target_include_directories(milestone06 PRIVATE include)

target_link_libraries(milestone06 PUBLIC lib MPI::MPI_CXX Kokkos::kokkos fmt::fmt)
