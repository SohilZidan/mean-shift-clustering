cmake_minimum_required(VERSION 3.9)

# set the project name and version
project( multi_model_multi_instance_fitting VERSION 1.0)

# 
# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


add_subdirectory(lib)
# add libs
# add_library(src distance.cpp)

###############
# packages #
###############

find_package( OpenCV REQUIRED )
find_package( Eigen3 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(OpenMP)

###############
# executables #
###############

add_executable( mean_shift main.cxx)
 #src/kernel.cpp src/meanshift.cpp)
target_link_libraries( mean_shift PUBLIC ${OpenCV_LIBS} Eigen3::Eigen lib)

if(OpenMP_CXX_FOUND)
    target_link_libraries( mean_shift PUBLIC OpenMP::OpenMP_CXX )
endif()

# target_include_directories(
#     mean_shift PUBLIC
#     "${PROJECT_SOURCE_DIR}/lib"
# )
