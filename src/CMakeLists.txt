cmake_minimum_required (VERSION 2.8)
project (NeuralNetwork_BASIC)

# make the ConvNetCL library
if (BUILD_DYNAMIC) 
    add_library (ConvNetCL_LIBRARY SHARED ConvNetCL.cpp)
else ()
    add_library (ConvNetCL_LIBRARY STATIC ConvNetCL.cpp)
endif ()

######
###### MAKE EXECUTABLES
######

add_executable (ConvNetTrainerCL ConvNetTrainerCL.cpp)
target_link_libraries(ConvNetTrainerCL ${OPENCL_LIBRARY} ${ConvNetCL_LIBRARY})

add_executable (ConvNetContinuanceCL ConvNetContinuanceCL.cpp)
target_link_libraries(ConvNetContinuanceCL ${OPENCL_LIBRARY} ${ConvNetCL_LIBRARY})

add_executable (ConvNetTester ConvNetTester.cpp ConvNet.cpp)
target_link_libraries(ConvNetTester ${OPENCL_LIBRARY})

add_executable (ConvNetFullImageDriverCL ConvNetFullImageDriverCL.cpp)
target_link_libraries(ConvNetFullImageDriverCL ${OpenCV_LIBS} ${OPENCL_LIBRARY} ${ConvNetCL_LIBRARY})

add_executable (ConvNetFullImageDriverParallelCL ConvNetFullImageDriverParallelCL.cpp)
target_link_libraries(ConvNetFullImageDriverParallelCL ${OpenCV_LIBS} ${OPENCL_LIBRARY} ${ConvNetCL_LIBRARY})

add_executable (ConvNetVideoDriverCL ConvNetVideoDriverCL.cpp)
target_link_libraries(ConvNetVideoDriverCL ${OpenCV_LIBS} ${OPENCL_LIBRARY} ${ConvNetCL_LIBRARY})

add_executable (ConvNetVideoDriverParallelCL ConvNetVideoDriverParallelCL.cpp)
target_link_libraries(ConvNetVideoDriverParallelCL ${OpenCV_LIBS} ${OPENCL_LIBRARY} ${ConvNetCL_LIBRARY})

add_executable (ConvNetVideoDriverParallelCL_multiNet ConvNetVideoDriverParallelCL_multiNet.cpp)
target_link_libraries(ConvNetVideoDriverParallelCL_multiNet ${OpenCV_LIBS} ${OPENCL_LIBRARY} ${ConvNetCL_LIBRARY})

add_executable (kernelBuilder kernelBuilder.cpp)
target_link_libraries(kernelBuilder ${OPENCL_LIBRARY})

add_executable (ImageSplitter ImageSplitter.cpp)
target_link_libraries(ImageSplitter ${OpenCV_LIBS})

add_executable (ImageFileCreator ImageFileCreator.cpp)
target_link_libraries(ImageFileCreator ${OpenCV_LIBS})

add_executable (TrainingImageFileCreator TrainingImageFileCreator.cpp)
target_link_libraries(ImageSplitterFileCreator ${OpenCV_LIBS})

add_executable (ImageSplitterFileCreator ImageSplitterFileCreator.cpp)
target_link_libraries(TrainingImageFileCreator ${OpenCV_LIBS})

add_executable (TrainingImageSplitterFileCreator TrainingImageSplitterFileCreator.cpp)
target_link_libraries(TrainingImageSplitterFileCreator ${OpenCV_LIBS})
