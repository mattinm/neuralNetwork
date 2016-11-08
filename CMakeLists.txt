cmake_minimum_required(VERSION 2.8)
project( ConvNetTest )
#set variables
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
SET( SRCS src/ConvNetTest.cpp src/ConvNet.cpp )
SET( CMAKE_CXX_FLAGS "-std=c++11 -Wall -O3")
SET( OpenCV_STATIC ON)

list(APPEND CMAKE_PREFIX_PATH /usr/local)




#find packages and libraries
find_package( OpenCV )
find_package( BOINC )
find_package( OpenCL )
find_package( MySQL )

#set include directories
include_directories( ${CMAKE_SOURCE_DIR}/src )
IF(OpenCV_FOUND)
	include_directories(${OpenCV_INCLUDE_DIRS})
ENDIF(OpenCV_FOUND)

IF(OpenCL_FOUND)
	include_directories(${OpenCL_INCLUDE_DIRS})
ENDIF(OpenCL_FOUND)

IF(BOINC_APP_FOUND OR BOINC_SERVER_FOUND)
	include_directories(
		${BOINC_INCLUDE_DIR}/boinc
		${BOINC_INCLUDE_DIR}
		)
ENDIF(BOINC_APP_FOUND OR BOINC_SERVER_FOUND)

IF(MYSQL_FOUND)
	include_directories(${MYSQL_INCLUDE_DIR})
ENDIF(MYSQL_FOUND)

IF(OpenCV_FOUND AND OpenCL_FOUND)
	add_executable( ConvNetTrainerCL src/ConvNetTrainerCL.cpp src/ConvNetCL.cpp )
	add_executable( ConvNetContinuanceCL src/ConvNetContinuanceCL.cpp src/ConvNetCL.cpp )
	add_executable( ConvNetFullImageDriverCL src/ConvNetFullImageDriverCL.cpp src/ConvNetCL.cpp)
	add_executable( ConvNetFullImageDriverParallelCL src/ConvNetFullImageDriverParallelCL.cpp src/ConvNetCL.cpp)
	add_executable( ConvNetVideoDriverParallelCL src/ConvNetVideoDriverParallelCL.cpp src/ConvNetCL.cpp)
	add_executable( ConvNetVideoDriverParallelCL_multiNet src/ConvNetVideoDriverParallelCL_multiNet.cpp src/ConvNetCL.cpp)

	target_link_libraries( ConvNetTrainerCL ${OPENCL_LIBRARY} )
	target_link_libraries( ConvNetContinuanceCL ${OPENCL_LIBRARY} )
	target_link_libraries( ConvNetFullImageDriverCL ${OpenCV_LIBS} ${OPENCL_LIBRARY})
	target_link_libraries( ConvNetFullImageDriverParallelCL ${OpenCV_LIBS} ${OPENCL_LIBRARY})
	target_link_libraries( ConvNetVideoDriverParallelCL ${OpenCV_LIBS} ${OPENCL_LIBRARY})
	target_link_libraries( ConvNetVideoDriverParallelCL_multiNet ${OpenCV_LIBS} ${OPENCL_LIBRARY})

	add_executable( SeamTest src/SeamTest.cpp src/ConvNetSeam )
	target_link_libraries(SeamTest ${OpenCV_LIBS} ${OPENCL_LIBRARY})
ENDIF(OpenCV_FOUND AND OpenCL_FOUND)

IF(OpenCV_FOUND AND OpenCL_FOUND AND BOINC_APP_FOUND)
	add_executable( ConvNetVideoDriverCL_BOINC src/ConvNetVideoDriverCL_BOINC.cpp src/ConvNetCL.cpp)
	add_executable( ConvNetFullImageDriverCL_BOINC src/ConvNetFullImageDriverCL_BOINC.cpp src/ConvNetCL.cpp)

	target_link_libraries( ConvNetFullImageDriverCL_BOINC ${OpenCV_LIBS} ${OPENCL_LIBRARY} ${BOINC_APP_LIBRARIES} ${BOINC_SERVER_LIBRARIES})
	target_link_libraries( ConvNetVideoDriverCL_BOINC ${OpenCV_LIBS} ${OPENCL_LIBRARY} ${BOINC_APP_LIBRARIES})

	add_executable( ConvNetSeamRun_BOINC src/ConvNetSeamRun_BOINC.cpp src/ConvNetSeam.cpp src/ConvNetCL.cpp src/ConvNetEvent)
target_link_libraries(ConvNetSeamRun_BOINC ${OPENCL_LIBRARY} ${OpenCV_LIBS} ${BOINC_APP_LIBRARIES})
ENDIF(OpenCV_FOUND AND OpenCL_FOUND AND BOINC_APP_FOUND)

IF(BOINC_SERVER_FOUND AND MYSQL_FOUND)
	add_executable(ConvNetSeam_work_generator
		ConvNetSeam_work_generator
		ConvNetCL_light
		)
	target_link_libraries(ConvNetSeam_work_generator ${UNDVC_COMMON_LIBRARY} ${BOINC_SERVER_LIBRARIES} ${MYSQL_LIBRARIES})

	add_executable(ConvNetSeam_validator
		${BOINC_INCLUDE_DIR}/sched/validator
		${BOINC_INCLUDE_DIR}/sched/validate_util
		${BOINC_INCLUDE_DIR}/sched/validate_util2
		ConvNetSeam_validator
		)
	target_link_libraries(ConvNetSeam_validator ${UNDVC_COMMON_LIBRARY} ${BOINC_SERVER_LIBRARIES} ${MYSQL_LIBRARIES})

	add_executable(ConvNetSeam_assimilator
		${BOINC_INCLUDE_DIR}/sched/assimilator
		${BOINC_INCLUDE_DIR}/sched/validate_util
		ConvNetSeam_validator
		ConvNetSeam_assimilator
		)
	target_link_libraries(ConvNetSeam_assimilator ${UNDVC_COMMON_LIBRARY} ${BOINC_SERVER_LIBRARIES} ${MYSQL_LIBRARIES})
ENDIF(BOINC_SERVER_FOUND AND MYSQL_FOUND)



IF(OpenCV_FOUND AND OpenCL_FOUND AND MYSQL_FOUND)
	add_executable( ConvNetSeamTrain_AllInOne src/ConvNetSeamTrain_AllInOne.cpp src/ConvNetSeam.cpp src/ConvNetCL.cpp )
	target_link_libraries( ConvNetSeamTrain_AllInOne ${OPENCL_LIBRARY} ${OpenCV_LIBS} ${MYSQL_LIBRARIES} )
ENDIF(OpenCV_FOUND AND OpenCL_FOUND AND MYSQL_FOUND)

#Image/Binary
IF(OpenCV_FOUND)
	add_executable( TrainingImageFileCreator src/TrainingImageFileCreator.cpp )
	add_executable( ImageSplitterFileCreator src/ImageSplitterFileCreator.cpp )
	add_executable( TrainingImageSplitterFileCreator src/TrainingImageSplitterFileCreator.cpp )

	target_link_libraries( ImageSplitterFileCreator ${OpenCV_LIBS} )
	target_link_libraries( TrainingImageFileCreator ${OpenCV_LIBS} )
	target_link_libraries( TrainingImageSplitterFileCreator ${OpenCV_LIBS} )
ENDIF(OpenCV_FOUND)

#Misc
IF(OpenCL_FOUND)
	add_executable( kernelBuilder src/kernelBuilder.cpp )
	target_link_libraries( kernelBuilder ${OPENCL_LIBRARY} )
ENDIF(OpenCL_FOUND)



#testing ones
add_executable( CLTest src/CLTest.cpp src/ConvNetCL_light)

