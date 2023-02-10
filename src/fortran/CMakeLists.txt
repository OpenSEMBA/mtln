cmake_minimum_required(VERSION 3.10)

project(TL LANGUAGES Fortran)
message(STATUS "Detected OS: " ${CMAKE_SYSTEM_NAME})
message(STATUS "Compiling Fortran with: " ${CMAKE_Fortran_COMPILER_ID} " " ${CMAKE_Fortran_COMPILER_VERSION})

set(src coaxial.f90 fd1d_TL.f90)
add_executable(coaxial ${src})

