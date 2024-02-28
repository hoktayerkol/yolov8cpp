# cmake -DCMAKE_PREFIX_PATH=/your/path 
# Finally I solved this problem by adding /path/to/libtorch to PATH.
# export PATH=$PATH:/path/to/directory
#
# # steps
# in the project
# mkdir build
# cd bulild
# cmake -DCMAKE_PREFIX_PATH="/home/ws/libtorch" ..
# make
# have to select cpp v 17


cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(yolov8)

# add_library(Torch SHARED IMPORTED) # or STATIC instead of SHARED
# set_target_properties(bar PROPERTIES
#   IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/lib/libbar.so"
#   INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/include/libbar"
# )

# find_package(Torch REQUIRED)

# add_library(Torch SHARED IMPORTED)
# set_target_properties(Torch PROPERTIES
#         INTERFACE_INCLUDE_DIRECTORIES /home/ws/libtorch/include/lib)

# include_directories(include ${TORCH_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

add_executable(yolov8 main.cpp)
target_link_libraries(yolov8 ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET yolov8 PROPERTY CXX_STANDARD 17)