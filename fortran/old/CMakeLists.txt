cmake_minimum_required(VERSION 3.10)

project(TL LANGUAGES Fortran)
message(STATUS "Detected OS: " ${CMAKE_SYSTEM_NAME})
message(STATUS "Compiling Fortran with: " ${CMAKE_Fortran_COMPILER_ID} " " ${CMAKE_Fortran_COMPILER_VERSION})
message(STATUS "Current source dir: " ${CMAKE_CURRENT_SOURCE_DIR})

set(src hello.f90)
add_executable(hello ${src})

# set(src Nconductors_3_losses.f90 fd1d_TL.f90 matrixUtils.f90)
# add_executable(Nconductors_3_losses ${src})
# link_directories(${CMAKE_CURRENT_SOURCE_DIR}/libs)
# target_link_libraries(Nconductors_3_losses ${CMAKE_CURRENT_SOURCE_DIR}/libs/LAPACK.lib ${CMAKE_CURRENT_SOURCE_DIR}/libs/BLAS.lib)
