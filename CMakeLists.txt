
cmake_minimum_required (VERSION 2.8)

project (demons)
include_directories (src)
add_executable(main src/main.cxx)
find_package(X11)
find_package(Threads)
if (X11_FOUND)
include_directories(${X11_INCLUDE_DIRS})
target_link_libraries (main ${X11_LIBRARIES})
endif (X11_FOUND)
target_link_libraries (main ${CMAKE_THREAD_LIBS_INIT})