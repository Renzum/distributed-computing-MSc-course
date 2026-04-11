add_library(lib STATIC hello.cpp)
target_include_directories(lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(MPI_CXX_FOUND)
    target_link_libraries(lib PUBLIC MPI::MPI_CXX)
endif()
