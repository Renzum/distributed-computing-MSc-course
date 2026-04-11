add_executable(main main.cpp)
target_link_libraries(main PRIVATE lib)

if(MPI_CXX_FOUND)
    target_link_libraries(main PRIVATE MPI::MPI_CXX)
endif()
