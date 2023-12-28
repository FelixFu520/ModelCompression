cmake_minimum_required(VERSION 3.1)

set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(infer LANGUAGES CXX CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_EXTENSIONS OFF)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)


# CUDA
find_package(CUDA REQUIRED)
get_filename_component(CUDA_LIB_DIR ${CUDA_LIBRARIES} DIRECTORY)
message(STATUS "CUDA Headers: \n${CUDA_INCLUDE_DIRS}\n")
message(STATUS "CUDA Libs: \n${CUDA_LIBRARIES}\n")

# TensorRT
set(TensorRT_INCLUDE_DIRS /usr/local/TensorRT-8.6.1.6/include)
set(TensorRT_LIBRARIES /usr/local/TensorRT-8.6.1.6/lib)
message(STATUS "TensorRT Libs: \n${TensorRT_LIBRARIES}\n")
message(STATUS "TensorRT Headers: \n${TensorRT_INCLUDE_DIRS}\n")

# OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV Libs: \n${OpenCV_LIBS}\n")
message(STATUS "OpenCV Libraries: \n${OpenCV_LIBRARIES}\n")
message(STATUS "OpenCV Headers: \n${OpenCV_INCLUDE_DIRS}\n")

list(APPEND INCLUDE_DIRS
        ${CUDA_INCLUDE_DIRS}
        ${TensorRT_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
)
message(STATUS "INCLUDE: \n${INCLUDE_DIRS}\n")

list(APPEND ALL_LIBS
        ${CUDA_LIBRARIES}
        ${CUDA_LIB_DIR}
        ${TensorRT_LIBRARIES}
        ${OpenCV_LIBRARIES}
)
message(STATUS "LIB: \n${ALL_LIBS}\n")

include_directories(${INCLUDE_DIRS})

add_executable(infer infer.cpp)
target_link_directories(infer PUBLIC ${ALL_LIBS})
target_link_libraries(${PROJECT_NAME} PRIVATE nvinfer nvinfer_plugin nvonnxparser nvonnxparser_static cudart ${OpenCV_LIBS})

