add_executable(milestone06 main.cpp mpi_data.cpp mpi_impl.cpp mpi_layer.cpp
    mpi_impl_not_gpu_aware.cpp
    mpi_impl_gpu_aware.cpp)

target_include_directories(milestone06 PRIVATE include)

target_link_libraries(milestone06 PUBLIC lib MPI::MPI_CXX Kokkos::kokkos fmt::fmt)
