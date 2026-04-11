cmake_minimum_required(VERSION 3.14)
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()
project(my-fantastic-simulation-code LANGUAGES C CXX VERSION 0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Dependencies: MPI, Eigen3, Kokkos
find_package(MPI REQUIRED)
if(MPI_CXX_FOUND)
    add_compile_definitions(WITH_MPI)
    message(STATUS "MPI found")
endif()

find_package(Eigen3)
if(NOT Eigen3_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      eigen3
      URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    )
    set(BUILD_TESTING OFF CACHE BOOL "Disable Eigen tests" FORCE)
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "Disable Eigen docs" FORCE)
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "Disable Eigen tests" FORCE)
    FetchContent_MakeAvailable(eigen3)
    if(NOT TARGET Eigen3::Eigen)
        add_library(Eigen3::Eigen ALIAS eigen)
    endif()
endif()

find_package(Kokkos REQUIRED)

enable_testing()

add_subdirectory(src)
add_subdirectory(executables)
add_subdirectory(tests)
