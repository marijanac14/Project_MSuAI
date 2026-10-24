cmake_minimum_required(VERSION 3.16)
project(lane_detection LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to a Release build 
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# OpenCV
# Provided by the system package `libopencv-dev` (OpenCV 4.x).
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# Executable
add_executable(lane_detector
  src/main.cpp
  src/SimpleLaneDetector.cpp
  src/Threshold.cpp
  src/Perspective.cpp
)

target_include_directories(lane_detector PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(lane_detector PRIVATE ${OpenCV_LIBS})

target_compile_options(lane_detector PRIVATE -Wall -Wextra)
