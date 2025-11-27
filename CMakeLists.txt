cmake_minimum_required(VERSION 3.10)
project(fastext_single_file)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall -pthread")

add_executable(fastext_single
    Fastext.cpp
)

