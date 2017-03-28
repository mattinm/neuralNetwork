cmake_minimum_required (VERSION 2.8)
project (NeuralNetwork_BASIC)

# make the ConvNetCL library
if (BUILD_DYNAMIC) 
    add_library (ConvNetCL SHARED ConvNetCL.cpp ConvNetCommon.cpp)
    #add_library (ConvNet SHARED ConvNet.cpp ConvNetCommon.cpp)
    add_library (ConvNetEvent SHARED ConvNetEvent.cpp ConvNetCommon.cpp)
    add_library (ConvNetSeam SHARED ConvNetSeam.cpp ConvNetCommon.cpp)
    add_library (Seamcarver SHARED Seamcarver.cpp)
else ()
    add_library (ConvNetCL STATIC ConvNetCL.cpp ConvNetCommon.cpp)
    #add_library (ConvNet STATIC ConvNet.cpp ConvNetCommon.cpp)
    add_library (ConvNetEvent STATIC ConvNetEvent.cpp ConvNetCommon.cpp)
    add_library (ConvNetSeam STATIC ConvNetSeam.cpp ConvNetCommon.cpp)
    add_library (Seamcarver STATIC Seamcarver.cpp)
endif ()

######
###### MAKE EXECUTABLES
######

add_executable (BlobCounter BlobCounter.cpp)
target_link_libraries(BlobCounter ${OpenCV_LIBS})

add_executable (ConvNetTrainerCL ConvNetTrainerCL.cpp)
target_link_libraries(ConvNetTrainerCL ${OpenCL_LIBRARIES} ConvNetCL)

add_executable (ConvNetTrainerCL_idx ConvNetTrainerCL_idx.cpp)
target_link_libraries(ConvNetTrainerCL_idx ${OpenCL_LIBRARIES} ConvNetCL ${OpenCV_LIBS})

#add_executable (ConvNetContinuanceCL ConvNetContinuanceCL.cpp)
#target_link_libraries(ConvNetContinuanceCL ${OpenCL_LIBRARIES} ConvNetCL)

#add_executable (ConvNetTester ConvNetTester.cpp)
#target_link_libraries(ConvNetTester ${OpenCL_LIBRARIES} ConvNet)

add_executable (ConvNetFullImageDriverCL ConvNetFullImageDriverCL.cpp)
target_link_libraries(ConvNetFullImageDriverCL ${OpenCV_LIBS} ${OpenCL_LIBRARIES} ConvNetCL)

# TODO: pthread
add_executable (ConvNetFullImageDriverParallelCL ConvNetFullImageDriverParallelCL.cpp)
target_link_libraries(ConvNetFullImageDriverParallelCL ${OpenCV_LIBS} ${OpenCL_LIBRARIES} ConvNetCL)

add_executable (ConvNetVideoDriverCL ConvNetVideoDriverCL.cpp)
target_link_libraries(ConvNetVideoDriverCL ${OpenCV_LIBS} ${OpenCL_LIBRARIES} ConvNetCL)

# TODO: pthread
#add_executable (ConvNetVideoDriverParallelCL ConvNetVideoDriverParallelCL.cpp)
#target_link_libraries(ConvNetVideoDriverParallelCL ${OpenCV_LIBS} ${OpenCL_LIBRARIES} ConvNetCL)

# TODO: pthread
#add_executable (ConvNetVideoDriverParallelCL_multiNet ConvNetVideoDriverParallelCL_multiNet.cpp)
#target_link_libraries(ConvNetVideoDriverParallelCL_multiNet ${OpenCV_LIBS} ${OpenCL_LIBRARIES} ConvNetCL)

add_executable(CNNtoObserver_comparator CNNtoObserver_comparator.cpp)
target_link_libraries(CNNtoObserver_comparator ConvNetEvent ${OpenCV_LIBS})

add_executable (kernelBuilder kernelBuilder.cpp)
target_link_libraries(kernelBuilder ${OpenCL_LIBRARIES})

add_executable (ImageSplitter ImageSplitter.cpp)
target_link_libraries(ImageSplitter ${OpenCV_LIBS})

add_executable (ImageFileCreator ImageFileCreator.cpp)
target_link_libraries(ImageFileCreator ${OpenCV_LIBS})

add_executable (MNIST_test MNIST_test.cpp)
target_link_libraries (MNIST_test ConvNetCL ${OpenCL_LIBRARIES} ${OpenCV_LIBS})

add_executable (Overlay Overlay.cpp)
target_link_libraries(Overlay ConvNetCL ${OpenCL_LIBRARIES} ${OpenCV_LIBS})

add_executable (TrainingImageFileCreator TrainingImageFileCreator.cpp)
target_link_libraries(TrainingImageFileCreator ${OpenCV_LIBS})

add_executable (ImageSplitterFileCreator ImageSplitterFileCreator.cpp)
target_link_libraries(ImageSplitterFileCreator ${OpenCV_LIBS})

add_executable (TrainingImageSplitterFileCreator TrainingImageSplitterFileCreator.cpp)
target_link_libraries(TrainingImageSplitterFileCreator ${OpenCV_LIBS})

IF(MYSQL_FOUND)
	add_executable(ConvNetSeam_ToIDX ConvNetSeam_ToIDX.cpp)
	target_link_libraries(ConvNetSeam_ToIDX ${OpenCV_LIBS} ${OpenCL_LIBRARIES} ConvNetEvent Seamcarver ${MYSQL_LIBRARIES})
ELSE()
	MESSAGE("Not building ConvNetSeam_ToIDX due to MySQL not found")
ENDIF()
