cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(erode)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

# Define our library target
add_library(erode SHARED op.cpp)
# Enable C++14
target_compile_features(erode PRIVATE cxx_std_14)
# Link against LibTorch
target_link_libraries(erode "${TORCH_LIBRARIES}")
#set(OpenCV_DIR /home/terence/libs/opencv/build)
# Link against OpenCV
target_link_libraries(erode opencv_core opencv_imgproc)
set(CUDA_HOME /usr/local/cuda)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
