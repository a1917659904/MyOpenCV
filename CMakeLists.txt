cmake_minimum_required(VERSION 3.29)
project(MyOpenCV)

set(CMAKE_CXX_STANDARD 20)

add_library(MyOpenCV STATIC library.cpp
        cmake-build-debug/demo.h
        cmake-build-debug/MyOpenCV.h
        cmake-build-debug/demo.cpp
        cmake-build-debug/MyOpenCV.cpp
        cmake-build-debug/实验一.cpp
        cmake-build-debug/实验二.cpp
        cmake-build-debug/实验三.cpp
        cmake-build-debug/实验四.cpp
        cmake-build-debug/实验五.cpp)
