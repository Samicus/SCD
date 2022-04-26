# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(SCD)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(mobile_application mobile_application.cpp)
target_link_libraries(mobile_application ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET mobile_application PROPERTY CXX_STANDARD 14)