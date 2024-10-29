cmake_minimum_required(VERSION 3.10)

project(ContourDetection LANGUAGES CXX CUDA)
# Set policy CMP0104 to NEW
if(POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW)
endif()

# Set C++17 standard for CUDA
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set C++17 standard for C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA architectures (adjust according to your GPU)
set(CMAKE_CUDA_ARCHITECTURES 60)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find CUDA
find_package(CUDA REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

# Set CUDA architecture flags (adjust according to your GPU)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_60,code=sm_60")

# Add the CUDA source file
add_executable(contour_detection contour_detection.cu)

# Set CUDA properties for the target
set_target_properties(contour_detection PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 17
)

# Link OpenCV and CUDA libraries (including NPP)
target_link_libraries(contour_detection ${OpenCV_LIBS} ${CUDA_LIBRARIES} nppicc nppig nppisu nppidei nppif )
