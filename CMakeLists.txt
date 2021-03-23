cmake_minimum_required(VERSION 3.9)

# set the project name and version
project( multi_model_multi_instance_fitting VERSION 1.0)

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

add_executable( mean_shift src/main.cpp src/distance.cpp src/kernel.cpp src/meanshift.cpp)
target_link_libraries( mean_shift ${OpenCV_LIBS} Eigen3::Eigen)

if(OpenMP_CXX_FOUND)
    target_link_libraries( mean_shift OpenMP::OpenMP_CXX )
endif()
