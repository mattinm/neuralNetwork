cmake_minimum_required(VERSION 2.8)
project( ConvNetTest )
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
find_package( OpenCV REQUIRED )
find_package( MPI REQUIRED )
find_library( OPENCL_LIBRARY OpenCL.framework )
SET( SRCS src/ConvNetTest.cpp src/ConvNet.cpp )
SET( CMAKE_CXX_FLAGS "-std=c++11 -Wall")
include_directories( ${OpenCV_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS} ${MPI_INCLUDE_PATH})

######
###### MAKE EXECUTABLES
######

#add_executable( ConvNetTest ${SRCS} )
#add_executable( CLTest src/CLTest.cpp src/ConvNetCL.cpp )

add_executable( ConvNetDriver src/ConvNetDriver.cpp src/ConvNet.cpp )

add_executable( ConvNetTrainer src/ConvNetTrainer.cpp src/ConvNet.cpp )
add_executable( ConvNetTrainerCL src/ConvNetTrainerCL.cpp src/ConvNetCL.cpp )

add_executable( ConvNetContinuance src/ConvNetContinuance.cpp src/ConvNet.cpp )
add_executable( ConvNetContinuanceCL src/ConvNetContinuanceCL.cpp src/ConvNetCL.cpp )

add_executable( ConvNetTester src/ConvNetTester.cpp src/ConvNet.cpp )

add_executable( ConvNetFullImageDriver src/ConvNetFullImageDriver.cpp src/ConvNet.cpp)
add_executable( ConvNetFullImageDriverCL src/ConvNetFullImageDriverCL.cpp src/ConvNetCL.cpp)

#add_executable( ConvNetVideoDriver src/ConvNetVideoDriver.cpp src/ConvNet.cpp)
#add_executable( ConvNetVideoDriverParallel src/ConvNetVideoDriverParallel.cpp src/ConvNet.cpp)
add_executable( ConvNetVideoDriverParallelCL src/ConvNetVideoDriverParallelCL.cpp src/ConvNetCL.cpp)
add_executable( ConvNetVideoDriverMPI src/ConvNetVideoDriverMPI.cpp src/ConvNetCL.cpp)

add_executable( kernelBuilder src/kernelBuilder.cpp )

add_executable( ImageSplitter src/ImageSplitter.cpp )

add_executable( ImageFileCreator src/ImageFileCreator.cpp )
add_executable( TrainingImageFileCreator src/TrainingImageFileCreator.cpp )
add_executable( ImageSplitterFileCreator src/ImageSplitterFileCreator.cpp )
add_executable( TrainingImageSplitterFileCreator src/TrainingImageSplitterFileCreator.cpp )

######
###### LINK LIBRARIES
######

#target_link_libraries( ConvNetTest ${OpenCV_LIBS} ${OPENCL_LIBRARY})
#target_link_libraries( CLTest ${OPENCL_LIBRARY} )

target_link_libraries( ConvNetDriver ${OPENCL_LIBRARY} )

target_link_libraries( ConvNetTrainer ${OPENCL_LIBRARY} )
target_link_libraries( ConvNetTrainerCL ${OPENCL_LIBRARY} )

target_link_libraries( ConvNetContinuance ${OPENCL_LIBRARY} )
target_link_libraries( ConvNetContinuanceCL ${OPENCL_LIBRARY} )

target_link_libraries( ConvNetTester ${OPENCL_LIBRARY} )

target_link_libraries( ConvNetFullImageDriver ${OpenCV_LIBS} ${OPENCL_LIBRARY})
target_link_libraries( ConvNetFullImageDriverCL ${OpenCV_LIBS} ${OPENCL_LIBRARY})

#target_link_libraries( ConvNetVideoDriver ${OpenCV_LIBS} ${OPENCL_LIBRARY})
#target_link_libraries( ConvNetVideoDriverParallel ${OpenCV_LIBS} ${OPENCL_LIBRARY})
target_link_libraries( ConvNetVideoDriverParallelCL ${OpenCV_LIBS} ${OPENCL_LIBRARY})
target_link_libraries( ConvNetVideoDriverMPI ${OpenCV_LIBS} ${OPENCL_LIBRARY} ${MPI_LIBRARY} ${MPI_EXTRA_LIBRARY})

target_link_libraries( kernelBuilder ${OPENCL_LIBRARY} )

target_link_libraries( ImageSplitter ${OpenCV_LIBS} )

target_link_libraries( ImageFileCreator ${OpenCV_LIBS} )
target_link_libraries( ImageSplitterFileCreator ${OpenCV_LIBS} )
target_link_libraries( TrainingImageFileCreator ${OpenCV_LIBS} )
target_link_libraries( TrainingImageSplitterFileCreator ${OpenCV_LIBS} )
