cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
project(cuda_hello LANGUAGES CXX CUDA)

set(CXX_STANDARD 17)
set(CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

add_executable(main main.cu)
