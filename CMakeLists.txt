cmake_minimum_required(VERSION 3.7)
project(fake_code)

set(CMAKE_CXX_STANDARD 11)
find_package( OpenCV REQUIRED )
find_package(Armadillo REQUIRED)
find_package(GLUT REQUIRED)
include_directories(${GLUT_INCLUDE_DIRS})
include_directories(${ARMADILLO_INCLUDE_DIRS})

link_directories(${GLUT_LIBRARY_DIRS})
add_definitions(${GLUT_DEFINITIONS})

if(NOT GLUT_FOUND)
    message(ERROR " GLUT not found!")
endif(NOT GLUT_FOUND)


find_package(OpenGL REQUIRED)
include_directories(${OpenGL_INCLUDE_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})
add_definitions(${OpenGL_DEFINITIONS})
if(NOT OPENGL_FOUND)
    message(ERROR " OPENGL not found!")
endif(NOT OPENGL_FOUND)
set(SOURCE_FILES #[[NeuralNetwork.cpp]] DataReader.cpp )
add_executable(fake_code ${SOURCE_FILES})
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( fake_code ${OpenCV_LIBS} ${cvsba_LIBS} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${ARMADILLO_LIBRARIES})