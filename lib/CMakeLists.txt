
find_package( OpenCV REQUIRED )
find_package( Eigen3 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_library(lib distance.cxx kernel.cxx meanshift.cxx)

target_include_directories(
    lib
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
)

