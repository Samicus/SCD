cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(SCD)

find_package(Torch REQUIRED)

add_executable(mobile_application mobile_application.cpp)
target_link_libraries(mobile_application "${TORCH_LIBRARIES}")
set_property(TARGET mobile_application PROPERTY CXX_STANDARD 14)