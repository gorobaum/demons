
cmake_minimum_required (VERSION 2.8)

project (demons)
include_directories (src)
find_package(X11)
find_package(Threads)

add_executable(main 
	src/main.cxx
	src/deform.cxx)

if (X11_FOUND)
include_directories(${X11_INCLUDE_DIRS})
target_link_libraries (main ${X11_LIBRARIES})
endif (X11_FOUND)
target_link_libraries (main ${CMAKE_THREAD_LIBS_INIT})


