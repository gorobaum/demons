
cmake_minimum_required (VERSION 2.8)

project (demons)
include_directories (src)
find_package(X11)
find_package(Threads)
find_package( OpenCV REQUIRED )

add_executable(demon 
	src/demon.cxx
	src/vectorfield.cxx
	src/interpolation.cxx
	src/image.cxx)

if (X11_FOUND)
include_directories(${X11_INCLUDE_DIRS})
target_link_libraries (demon ${X11_LIBRARIES})
endif (X11_FOUND)
target_link_libraries (demon ${CMAKE_THREAD_LIBS_INIT} ${OpenCV_LIBS})
list( APPEND CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS_DEBUG} -Wall -g")

add_custom_target (
	cleanShit
	COMMAND ${CMAKE_COMMAND} -E remove VFN-Iteration* Iteration* DFI-Iteration* VectorFieldInformation.info deformed* Gradient* DeltaFieldInformation.info
)