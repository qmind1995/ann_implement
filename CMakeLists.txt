cmake_minimum_required(VERSION 3.7)
project(Demo_ann)

set(CMAKE_CXX_STANDARD 11)
find_package(Armadillo REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
include_directories(${GLUT_INCLUDE_DIRS})
include_directories(${ARMADILLO_INCLUDE_DIRS})

add_definitions(${GLUT_DEFINITIONS})

if(NOT GLUT_FOUND)
    message(ERROR " GLUT not found!")
endif(NOT GLUT_FOUND)

include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} )
link_directories(${OpenGL_LIBRARY_DIRS})
link_directories(${GLUT_LIBRARY_DIRS})
add_definitions(${OpenGL_DEFINITIONS})
if(NOT OPENGL_FOUND)
    message(ERROR " OPENGL not found!")
endif(NOT OPENGL_FOUND)
#set(SOURCE_FILES main.cpp NeuralNetwork.cpp DataReader.cpp Trainer.cpp BatchTrainer.cpp DataGenerator.cpp HOGFeature.cpp Layer.cpp)
set(SOURCE_FILES
        main.cpp
        ann/Trainer.cpp
        ann/BatchTrainer.cpp
        ann/NeuralNetwork.cpp
        dataReader/DataReader.cpp
        dataReader/DataGenerator.cpp
        dataReader/HOGFeature.cpp
        ann/Layer.cpp
        Utils.cpp
        visualize/Visualize.cpp)
add_executable(Demo_ann ${SOURCE_FILES})
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( Demo_ann ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${ARMADILLO_LIBRARIES} pthread)