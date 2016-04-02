# Target
add_executable(face_motionseg face_motionseg_app.cpp)
target_include_directories(face_motionseg PRIVATE ../motionseg/include ${VSAL_INCLUDE_DIRS})
target_link_libraries(face_motionseg motionseg vsal)
if(UNIX)
	target_link_libraries(face_motionseg ${Boost_LIBRARIES})
endif(UNIX)
set_target_properties(face_motionseg PROPERTIES DEBUG_POSTFIX _d)

# Installation
install(TARGETS face_motionseg EXPORT motionseg-targets DESTINATION bin COMPONENT bin)
if(BUILD_INTERFACE_MATLAB)
	install(DIRECTORY ${CMAKE_SOURCE_DIR}/interfaces/matlab/face_motionseg
			DESTINATION interfaces/matlab
			FILES_MATCHING PATTERN "*.m")
endif()
