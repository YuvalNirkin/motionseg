# Source
set(MOTIONSEG_INCLUDE include/motionseg/motionseg.h include/motionseg/face_motionseg.h)
set(MOTIONSEG_SRC src/thread_pool.h src/motionseg.cpp src/face_motionseg.cpp)
source_group("include\\motionseg" FILES ${MOTIONSEG_INCLUDE})

# Target
add_library(motionseg ${MOTIONSEG_INCLUDE} ${MOTIONSEG_SRC})
target_include_directories(motionseg PRIVATE include ${dlib_INCLUDE_DIRS})
target_include_directories(motionseg PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(motionseg PRIVATE ${dlib_LIBRARIES})
target_link_libraries(motionseg PUBLIC ${OpenCV_LIBS})
if(UNIX)
	target_link_libraries(motionseg PRIVATE ${Boost_LIBRARIES})
endif(UNIX)
set_target_properties(motionseg PROPERTIES DEBUG_POSTFIX _d)

# Installation
install(TARGETS motionseg
	 EXPORT motionseg-targets
     RUNTIME DESTINATION bin COMPONENT dev
     LIBRARY DESTINATION lib COMPONENT dev
     ARCHIVE DESTINATION lib COMPONENT dev)
install(FILES ${MOTIONSEG_INCLUDE} DESTINATION include/motionseg)
