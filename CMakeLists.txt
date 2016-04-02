cmake_minimum_required(VERSION 2.8)
project(motionseg)

# Version
# ===================================================
set(MOTIONSEG_MAJOR_VERSION 0)
set(MOTIONSEG_MINOR_VERSION 5)
set(MOTIONSEG_VERSION ${MOTIONSEG_MAJOR_VERSION}.${MOTIONSEG_MINOR_VERSION})

# Global configurations
# ===================================================
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
if(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
	 SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX")
endif()

# motionseg build components
# ===================================================
option(BUILD_FACE_MOTIONSEG "Build face_motionseg application" ON)
option(BUILD_DOCS "Build documentation using Doxygen" ON)
option(BUILD_INTERFACE_MATLAB "Build interface for Matlab" ON)

# Find dependencies
# ===================================================
# Boost
if(MSVC)
	set(Boost_USE_STATIC_LIBS ON)	# only find static libs
endif(MSVC)
set(Boost_USE_MULTITHREADED ON)
find_package( Boost REQUIRED program_options filesystem)
if(Boost_FOUND)
	include_directories(${Boost_INCLUDE_DIRS})
	if(MSVC)
		# On Windows the Boost libraries are automatically linked
		link_directories(${Boost_LIBRARY_DIRS})
	endif(MSVC)	
endif()

find_package(OpenCV REQUIRED highgui imgproc imgcodecs video videoio cudaoptflow cudacodec cudastereo)
find_package(vsal REQUIRED)
find_package(dlib REQUIRED)

if(BUILD_DOCS)
	find_package(Doxygen)
endif()

# Process subdirectories
# ===================================================
add_subdirectory(motionseg)
set(MOTIONSEG_TARGETS motionseg)

if(BUILD_FACE_MOTIONSEG)
	add_subdirectory(face_motionseg)
	set(MOTIONSEG_TARGETS ${MOTIONSEG_TARGETS} face_motionseg)
endif()

if(BUILD_DOCS AND DOXYGEN_FOUND)
	add_subdirectory(doc)
endif()

# interfaces
if(BUILD_INTERFACE_MATLAB)
	add_subdirectory(interfaces/matlab)
endif()

# Export configuration
# ===================================================

# Add all targets to the build-tree export set
export(TARGETS ${MOTIONSEG_TARGETS}
  FILE "${PROJECT_BINARY_DIR}/motionseg-targets.cmake")
  
# Export the package for use from the build-tree
# (this registers the build-tree with a global CMake-registry)
export(PACKAGE motionseg)

# Create the motionseg-config.cmake and motionseg-config-version.cmake files
set(CONF_INCLUDE_DIRS "\${MOTIONSEG_CMAKE_DIR}/include")
configure_file(cmake/motionseg-config.cmake.in
  "${PROJECT_BINARY_DIR}/motionseg-config.cmake" @ONLY)
configure_file(cmake/motionseg-config-version.cmake.in
  "${PROJECT_BINARY_DIR}/motionseg-config-version.cmake" @ONLY)
   
# Install the motionseg-config.cmake and motionseg-config-version.cmake
install(FILES
  "${PROJECT_BINARY_DIR}/motionseg-config.cmake"
  "${PROJECT_BINARY_DIR}/motionseg-config-version.cmake"
  DESTINATION "cmake" COMPONENT dev)
 
# Install the export set for use with the install-tree
install(EXPORT motionseg-targets DESTINATION cmake COMPONENT dev)