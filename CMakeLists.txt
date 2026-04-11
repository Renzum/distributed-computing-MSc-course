cmake_minimum_required(VERSION 3.14)
project(my-fantastic-simulation-code LANGUAGES C CXX VERSION 0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optional dependency: MPI
find_package(MPI)

if(MPI_CXX_FOUND)
    add_compile_definitions(WITH_MPI)
    message(STATUS "MPI found")
else()
    message(STATUS "MPI not found, disabling MPI support")
endif()

enable_testing()

add_subdirectory(src)
add_subdirectory(executables)
add_subdirectory(tests)
