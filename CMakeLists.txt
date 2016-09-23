cmake_minimum_required(VERSION 2.8)
project( ConvNetTest )
#set variables
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
SET( SRCS src/ConvNetTest.cpp src/ConvNet.cpp )
SET( CMAKE_CXX_FLAGS "-std=c++11 -Wall -O3")
SET( OpenCV_STATIC ON)

list(APPEND CMAKE_PREFIX_PATH /usr/local)




#find packages and libraries
find_package( OpenCV REQUIRED )
find_package( MPI REQUIRED )
find_package( BOINC REQUIRED )
find_package( OpenCL REQUIRED )
find_package( MySQL REQUIRED )
#find_library( OPENCL_LIBRARY OpenCL.framework )


#set include directories
include_directories( ${OpenCV_INCLUDE_DIRS} 
	${OpenCL_INCLUDE_DIRS} 
	${MPI_INCLUDE_PATH} 
	${BOINC_INCLUDE_DIRS} 
	${CMAKE_SOURCE_DIR}/src
	${MYSQL_INCLUDE_DIR}
	${BOINC_INCLUDE_DIR}/boinc 
	# "../test/boinc" 
	# "/usr/local/Cellar/mysql/5.7.11/include/mysql" 
	# "/usr/local/Cellar/"
	# "/usr/local/opt/openssl/include" 
	# "/usr/local/opt/openssl/lib" 
	# "/Users/connorbowley/Development/test/boinc/sched"
	# "/Users/connorbowley/Development/test/boinc/db"
	)

# SET(MYSQL_SRCS "/usr/local/Cellar/mysql/5.7.11/include/mysql/mysql.h")

######
###### MAKE EXECUTABLES
######

#non CL (still used OpenCL, but is the old version of this code)
# add_executable( ConvNetDriver src/ConvNetDriver.cpp src/ConvNet.cpp )
# add_executable( ConvNetTrainer src/ConvNetTrainer.cpp src/ConvNet.cpp )
# add_executable( ConvNetContinuance src/ConvNetContinuance.cpp src/ConvNet.cpp )
add_executable( ConvNetTester src/ConvNetTester.cpp src/ConvNet.cpp )
# add_executable( ConvNetFullImageDriver src/ConvNetFullImageDriver.cpp src/ConvNet.cpp)


#CL
add_executable( ConvNetTrainerCL src/ConvNetTrainerCL.cpp src/ConvNetCL.cpp )
add_executable( ConvNetContinuanceCL src/ConvNetContinuanceCL.cpp src/ConvNetCL.cpp )
# add_executable( ConvNetFullImageDriverCL_newOutput src/ConvNetFullImageDriverCL_newOutput.cpp src/ConvNetCL.cpp)
add_executable( ConvNetFullImageDriverCL src/ConvNetFullImageDriverCL.cpp src/ConvNetCL.cpp)
add_executable( ConvNetFullImageDriverParallelCL src/ConvNetFullImageDriverParallelCL.cpp src/ConvNetCL.cpp)
add_executable( ConvNetFullImageDriverCL_BOINC src/ConvNetFullImageDriverCL_BOINC.cpp src/ConvNetCL.cpp)
# add_executable( ConvNetVideoDriverCL src/ConvNetVideoDriverCL.cpp src/ConvNetCL.cpp)
add_executable( ConvNetVideoDriverParallelCL src/ConvNetVideoDriverParallelCL.cpp src/ConvNetCL.cpp)
add_executable( ConvNetVideoDriverParallelCL_multiNet src/ConvNetVideoDriverParallelCL_multiNet.cpp src/ConvNetCL.cpp)
add_executable( ConvNetVideoDriverMPI src/ConvNetVideoDriverMPI.cpp src/ConvNetCL.cpp)
add_executable( ConvNetVideoDriverCL_BOINC src/ConvNetVideoDriverCL_BOINC.cpp src/ConvNetCL.cpp)

#Seam
# add_executable( ConvNetSeamTrain src/ConvNetSeamTrain.cpp src/ConvNetSeam.cpp src/ConvNetCL.cpp )
# target_link_libraries( ConvNetSeamTrain ${OPENCL_LIBRARY})# ${OpenCV_LIBS} ${MYSQL_LIBRARIES} )
add_executable( SeamTest src/SeamTest.cpp src/ConvNetSeam )
target_link_libraries(SeamTest ${OpenCV_LIBS})
add_executable( ConvNetSeamFile src/ConvNetSeamFile.cpp src/ConvNetSeam.cpp src/ConvNetCL.cpp)
target_link_libraries(ConvNetSeamFile ${OpenCV_LIBS} ${MYSQL_LIBRARIES} ${OPENCL_LIBRARY})


#Image/Binary
# add_executable( ImageSplitter src/ImageSplitter.cpp )
# add_executable( ImageFileCreator src/ImageFileCreator.cpp )
add_executable( TrainingImageFileCreator src/TrainingImageFileCreator.cpp )
add_executable( ImageSplitterFileCreator src/ImageSplitterFileCreator.cpp )
add_executable( TrainingImageSplitterFileCreator src/TrainingImageSplitterFileCreator.cpp )

#BOINC Specific
# add_executable( ConvNet_Video_work_generator src/ConvNet_Video_work_generator.cxx /usr/local/Cellar/mysql/5.7.11/include/mysql/mysql.h)

#Misc
add_executable( kernelBuilder src/kernelBuilder.cpp )
# add_executable( FileStorageOperations src/FileStorageOperations)


######
###### LINK LIBRARIES
######

#non CL
# target_link_libraries( ConvNetDriver ${OPENCL_LIBRARY} )
# target_link_libraries( ConvNetTrainer ${OPENCL_LIBRARY} )
# target_link_libraries( ConvNetContinuance ${OPENCL_LIBRARY} )
target_link_libraries( ConvNetTester ${OPENCL_LIBRARY} )
# target_link_libraries( ConvNetFullImageDriver ${OpenCV_LIBS} ${OPENCL_LIBRARY})


#CL
target_link_libraries( ConvNetTrainerCL ${OPENCL_LIBRARY} )
target_link_libraries( ConvNetContinuanceCL ${OPENCL_LIBRARY} )
# target_link_libraries( ConvNetFullImageDriverCL_newOutput ${OpenCV_LIBS} ${OPENCL_LIBRARY})
target_link_libraries( ConvNetFullImageDriverCL ${OpenCV_LIBS} ${OPENCL_LIBRARY})
target_link_libraries( ConvNetFullImageDriverParallelCL ${OpenCV_LIBS} ${OPENCL_LIBRARY})
target_link_libraries( ConvNetFullImageDriverCL_BOINC ${OpenCV_LIBS} ${OPENCL_LIBRARY} ${BOINC_APP_LIBRARIES} ${BOINC_SERVER_LIBRARIES})
# target_link_libraries( ConvNetVideoDriverCL ${OpenCV_LIBS} ${OPENCL_LIBRARY})
target_link_libraries( ConvNetVideoDriverParallelCL ${OpenCV_LIBS} ${OPENCL_LIBRARY})
target_link_libraries( ConvNetVideoDriverParallelCL_multiNet ${OpenCV_LIBS} ${OPENCL_LIBRARY})
target_link_libraries( ConvNetVideoDriverMPI ${OpenCV_LIBS} ${OPENCL_LIBRARY} ${MPI_LIBRARY} ${MPI_EXTRA_LIBRARY})
target_link_libraries( ConvNetVideoDriverCL_BOINC ${OpenCV_LIBS} ${OPENCL_LIBRARY} ${BOINC_APP_LIBRARIES})




#Image/Binary
# target_link_libraries( ImageSplitter ${OpenCV_LIBS} )
# target_link_libraries( ImageFileCreator ${OpenCV_LIBS} )
target_link_libraries( ImageSplitterFileCreator ${OpenCV_LIBS} )
target_link_libraries( TrainingImageFileCreator ${OpenCV_LIBS} )
target_link_libraries( TrainingImageSplitterFileCreator ${OpenCV_LIBS} )

#BOINC Specific
# target_link_libraries( ConvNet_Video_work_generator ${BOINC_LIBRARIES} /usr/local/Cellar/mysql/5.7.11/include/mysql )

#Misc
target_link_libraries( kernelBuilder ${OPENCL_LIBRARY} )
# target_link_libraries( FileStorageOperations ${OpenCV_LIBS} )
