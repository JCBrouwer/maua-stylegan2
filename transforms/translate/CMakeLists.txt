cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(translate)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

# Define our library target
add_library(translate SHARED op.cpp)
# Enable C++14
target_compile_features(translate PRIVATE cxx_std_14)
# Link against LibTorch
target_link_libraries(translate "${TORCH_LIBRARIES}")
# Link against OpenCV
target_link_libraries(translate opencv_core opencv_imgproc)
set(CUDA_HOME /usr/local/cuda)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)