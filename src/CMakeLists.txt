cmake_minimum_required(VERSION 3.1)
project(ist)

# ## Compile as C++11
# add_compile_options(-std=c++11)

# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)


## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)

## Specify additional locations of header files
## Your package locations should be listed before other locations

SET( PROJECT_INCLUDES "../include" )
SET( OPENCV_ROOT "$ENV{OPENCV_ROOT}" )
SET( OPENCV_INCLUDES "${OPENCV_ROOT}/include/opencv4" )
SET( OPENCV_LIB_PATH "${OPENCV_ROOT}/lib" )
SET( OPENCV_PACKAGE "${OPENCV_ROOT}/lib/cmake/opencv4" )

# Show the above variables.
MESSAGE(STATUS "OPENCV_ROOT = ${OPENCV_ROOT}")
MESSAGE(STATUS "OPENCV_INCLUDES = ${OPENCV_INCLUDES}")
MESSAGE(STATUS "OPENCV_LIB_PATH = ${OPENCV_LIB_PATH}")

include_directories(
# include
  ${PROJECT_INCLUDES}
  # ${OPENCV_INCLUDES}
)

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/stereo_xi_camera.cpp
# )

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# >>>
# Search the ./src folder for .cpp files.

FILE(GLOB_RECURSE SRC_LIST *.cpp)

IF (SRC_LIST)
	FOREACH (SFILE ${SRC_LIST})
		MESSAGE(STATUS "CPP: ${SFILE}")
	ENDFOREACH (SFILE)
ENDIF (SRC_LIST) 

# Declare a C++ executable
# With catkin_make all packages are built within a single CMake context
# The recommended prefix ensures that target names across packages don't collide
add_executable(${PROJECT_NAME}_exe ${SRC_LIST})

# ## Add cmake target dependencies of the executable
# ## same as for the library above

# FIND_LIBRARY( LIB_OPENCV_CORE NAMES opencv_core PATHS ${OPENCV_LIB_PATH} NO_DEFAULT_PATH )
# FIND_LIBRARY( LIB_OPENCV_HIGHGUI NAMES opencv_highgui PATHS ${OPENCV_LIB_PATH} NO_DEFAULT_PATH )
# FIND_LIBRARY( LIB_OPENCV_IMGPROC NAMES opencv_imgproc PATHS ${OPENCV_LIB_PATH} NO_DEFAULT_PATH )
# FIND_LIBRARY( LIB_OPENCV_IMGCODECS NAMES opencv_imgcodecs PATHS ${OPENCV_LIB_PATH} NO_DEFAULT_PATH )
# FIND_LIBRARY( LIB_OPENCV_XFEATURE2D NAMES opencv_xfeatures2d PATHS ${OPENCV_LIB_PATH} NO_DEFAULT_PATH )
# FIND_LIBRARY( LIB_BOOST_EXCEPTION NAMES boost_exception )

# # Show the found libraries.
# MESSAGE( STATUS "LIB_OPENCV_CORE = ${LIB_OPENCV_CORE}" )
# MESSAGE( STATUS "LIB_OPENCV_HIGHGUI = ${LIB_OPENCV_HIGHGUI}" )
# MESSAGE( STATUS "LIB_OPENCV_IMGPROC = ${LIB_OPENCV_IMGPROC}" )
# MESSAGE( STATUS "LIB_OPENCV_IMGCODECS = ${LIB_OPENCV_IMGCODECS}" )
# MESSAGE( STATUS "LIB_OPENCV_XFEATURE2D = ${LIB_OPENCV_XFEATURE2D}" )
# MESSAGE( STATUS "LIB_BOOST_EXCEPTION = ${LIB_BOOST_EXCEPTION}" )

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED PATHS ${OPENCV_PACKAGE} NO_DEFAULT_PATH)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")


SET( CXX_FLAGS "-g" )
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${CXX_FLAGS}" )

# Specify libraries to link a library or executable target against
# target_link_libraries(${PROJECT_NAME}_exe
#   ${LIB_OPENCV_CORE} ${LIB_OPENCV_HIGHGUI} ${LIB_OPENCV_IMGPROC} ${LIB_OPENCV_IMGCODECS} ${LIB_OPENCV_XFEATURE2D} 
#   ${LIB_BOOST_EXCEPTION}
# )

target_link_libraries(${PROJECT_NAME}_exe ${OpenCV_LIBS}
)
